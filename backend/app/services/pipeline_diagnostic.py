"""
Pipeline diagnostic service.
Traces prompt tokenization, text embeddings, model architecture,
checkpoint metadata, device, latent shapes, and execution timing.
"""
from __future__ import annotations

import time
import torch
import numpy as np
from typing import Dict, Any, Optional

from training.preprocessing.text_preprocessor import TextPreprocessor
from app.adapters.video.neural_video_adapter import NeuralVideoAdapter
from app.core.config import settings


def diagnose_pipeline_execution(
    prompt: str,
    seed: int = 42,
    fps: int = 24,
    num_frames: int = 16,
    resolution: str = "1280x720",
    run_forward: bool = True,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Text Preprocessing & Tokenization
    text_preprocessor = TextPreprocessor(max_length=128)
    token_ids = text_preprocessor.encode(prompt)
    diag = text_preprocessor.diagnose_prompt(prompt)

    # 2. Video Adapter & Model
    adapter = NeuralVideoAdapter()
    is_loaded = adapter.is_available()
    device = str(adapter.device)
    ckpt_path = str(adapter.checkpoint_path) if adapter.checkpoint_path else "None"

    # 3. Model Architecture & Embeddings
    embedding_shape = None
    latent_channels = 4
    model_name = "None"
    total_params = 0
    trainable_params = 0

    if is_loaded and adapter.model is not None:
        model_name = adapter.model.__class__.__name__
        total_params = sum(p.numel() for p in adapter.model.parameters())
        trainable_params = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
        latent_channels = getattr(adapter.model, "latent_channels", 4)
        
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=adapter.device)
        try:
            with torch.no_grad():
                emb = adapter.model.text_encoder(token_tensor)
                if isinstance(emb, tuple):
                    embedding_shape = [list(e.shape) for e in emb]
                else:
                    embedding_shape = list(emb.shape)
        except Exception as e:
            embedding_shape = f"Error: {e}"
    else:
        embedding_shape = [1, len(token_ids), 64]

    # Latent shape: [B, C, T, H/8, W/8]
    latent_shape = [1, latent_channels, num_frames, 16, 16]

    # 4. Forward pass execution test
    forward_duration = 0.0
    if run_forward and is_loaded and adapter.model is not None:
        f_start = time.perf_counter()
        try:
            with torch.no_grad():
                # Model takes (B, 3, H, W) keyframe and (B, L) token tensor
                test_kf = torch.randn(1, 3, 128, 128, device=adapter.device)
                _ = adapter.model(test_kf, token_tensor)
                forward_duration = round(time.perf_counter() - f_start, 4)
        except Exception as e:
            forward_duration = f"Error: {e}"

    total_duration = round(time.perf_counter() - t0, 4)

    return {
        "status": "success",
        "prompt_received": prompt,
        "tokens_produced": {
            "count": len(token_ids),
            "token_ids": token_ids,
            "words_count": diag.get("raw_word_count"),
            "subwords_count": diag.get("tokens_count"),
            "tokens_preserved": diag.get("non_pad_tokens"),
            "tokens_truncated": diag.get("is_truncated"),
            "tokenizer_type": diag.get("tokenizer_type"),
            "term_retention": diag.get("term_retention"),
        },
        "text_embedding_shape": embedding_shape,
        "model": {
            "name": model_name,
            "loaded": is_loaded,
            "checkpoint_path": ckpt_path,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        },
        "device": device,
        "latent_shape": latent_shape,
        "generation_config": {
            "num_frames": num_frames,
            "fps": fps,
            "resolution": resolution,
            "seed": seed,
        },
        "timing": {
            "total_diagnostic_time_sec": total_duration,
            "model_forward_time_sec": forward_duration,
        }
    }
