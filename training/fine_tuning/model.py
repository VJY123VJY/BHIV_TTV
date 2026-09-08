import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class TextEncoder(nn.Module):
    """Encodes tokenized prompt text into semantic conditioning vectors."""
    def __init__(self, vocab_size: int = 10000, max_length: int = 32, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embeddings = nn.Parameter(torch.randn(1, max_length, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L)
        b, seq_len = token_ids.shape
        safe_tokens = token_ids.clamp(0, self.token_embeddings.num_embeddings - 1)
        max_pos = self.pos_embeddings.shape[1]
        if seq_len > max_pos:
            safe_tokens = safe_tokens[:, :max_pos]
            seq_len = max_pos
        x = self.token_embeddings(safe_tokens) + self.pos_embeddings[:, :seq_len, :]
        x = self.norm(x)
        # Global average pool over sequence length
        mask = (safe_tokens != 0).unsqueeze(-1).float() # (B, seq_len, 1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0) # (B, D)
        return self.proj(pooled) # (B, embed_dim)


class TemporalResidualBlock(nn.Module):
    """Temporal residual convolution and cross-attention block."""
    def __init__(self, channels: int = 64, text_dim: int = 128):
        super().__init__()
        self.channels = channels
        self.norm1 = nn.GroupNorm(8, channels)
        # 1D temporal convolution across time dimension
        self.temporal_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, channels)
        # Cross-attention projections
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(text_dim, channels)
        self.to_v = nn.Linear(text_dim, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, text_context: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        # 1. Temporal 1D convolution
        # Reshape to (B * H * W, C, T)
        x_perm = x.permute(0, 3, 4, 2, 1).contiguous().view(b * h * w, c, t)
        h_temp = F.relu(self.temporal_conv(x_perm))
        # Reshape back to (B, T, C, H, W)
        h_temp = h_temp.view(b, h, w, c, t).permute(0, 4, 3, 1, 2)
        x = x + h_temp

        # 2. Text cross-attention
        # Query from spatial-temporal features, Key/Value from text context
        # (B, T, H, W, C)
        x_flat = x.permute(0, 1, 3, 4, 2).contiguous().view(b, t * h * w, c)
        q = self.to_q(x_flat) # (B, t*h*w, C)
        k = self.to_k(text_context).unsqueeze(1) # (B, 1, C)
        v = self.to_v(text_context).unsqueeze(1) # (B, 1, C)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(c) # (B, t*h*w, 1)
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, v) # (B, t*h*w, C)
        out = self.proj(out).view(b, t, h, w, c).permute(0, 1, 4, 2, 3)

        return x + out


class SpatialTemporalTTVModel(nn.Module):
    """
    Genuine PyTorch Spatial-Temporal Neural Text-to-Video Architecture.
    Consists of:
    - Text Conditioning Encoder
    - Spatial Frame Feature Extractor
    - Temporal Motion Synthesizer
    - Frame Decoder
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 64,
        num_frames: int = 8,
        vocab_size: int = 10000,
        max_prompt_length: int = 32,
        text_embed_dim: int = 128
    ):
        super().__init__()
        self.num_frames = num_frames
        self.latent_channels = latent_channels
        self.text_embed_dim = text_embed_dim

        # 1. Text conditioning
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_length=max_prompt_length,
            embed_dim=text_embed_dim
        )

        # 2. Spatial Keyframe Encoder (downsamples 4x)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # /2
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, latent_channels, kernel_size=3, stride=2, padding=1), # /4
            nn.GroupNorm(8, latent_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. Temporal Position Embedding
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames, latent_channels, 1, 1) * 0.02)

        # 4. Temporal Motion Synthesizer Blocks
        self.temp_block1 = TemporalResidualBlock(latent_channels, text_embed_dim)
        self.temp_block2 = TemporalResidualBlock(latent_channels, text_embed_dim)

        # 5. Spatial Frame Decoder (upsamples 4x back to image space)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=4, stride=2, padding=1), # x2
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # x4
            nn.Tanh() # Normalizes output to [-1.0, 1.0]
        )

        # 6. Alignment projection head for semantic similarity
        self.video_semantic_proj = nn.Linear(latent_channels, text_embed_dim)

    def forward(
        self,
        keyframe: torch.Tensor,
        prompt_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Inputs:
          keyframe: Initial frame tensor (B, C, H, W)
          prompt_tokens: Tokenized prompt IDs (B, L)
        Returns:
          generated_frames: (B, T, C, H, W) normalized to [-1.0, 1.0]
          video_embedding: (B, text_embed_dim) for semantic alignment loss
          text_embedding: (B, text_embed_dim)
        """
        b = keyframe.shape[0]

        # 1. Encode text context
        text_context = self.text_encoder(prompt_tokens) # (B, text_embed_dim)

        # 2. Encode initial keyframe
        z0 = self.spatial_encoder(keyframe) # (B, latent_channels, H/4, W/4)

        # 3. Expand across time dimension
        z_seq = z0.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1) # (B, T, C, h, w)
        z_seq = z_seq + self.temporal_pos_embed[:, :self.num_frames, :, :, :]

        # 4. Synthesize motion
        z_seq = self.temp_block1(z_seq, text_context)
        z_seq = self.temp_block2(z_seq, text_context)

        # 5. Extract global video representation for semantic alignment
        # Average pool over (T, H, W)
        video_emb = self.video_semantic_proj(z_seq.mean(dim=(1, 3, 4)))

        # 6. Decode each frame
        # Flatten B and T to decode in parallel
        b, t, c, h, w = z_seq.shape
        z_flat = z_seq.view(b * t, c, h, w)
        out_flat = self.decoder(z_flat) # (B*T, 3, H, W)
        generated_frames = out_flat.view(b, t, 3, out_flat.shape[2], out_flat.shape[3])

        return generated_frames, video_emb, text_context
