import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import torch
from torch.utils.data import Dataset, DataLoader

from training.preprocessing.video_preprocessor import VideoPreprocessor
from training.preprocessing.text_preprocessor import TextPreprocessor


class TextToVideoDataset(Dataset):
    """
    PyTorch Dataset for Text-to-Video training.
    Loads video files from manifest, extracts normalized frame sequences,
    and tokenizes prompt text into tensor representations.
    """
    def __init__(
        self,
        manifest_path: str,
        num_frames: int = 8,
        height: int = 128,
        width: int = 128,
        max_prompt_length: int = 32,
        vocab_size: int = 10000,
        transform: Optional[Callable] = None
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.samples: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        self.video_preprocessor = VideoPreprocessor(
            num_frames=num_frames,
            height=height,
            width=width
        )
        self.text_preprocessor = TextPreprocessor(max_length=max_prompt_length, vocab_size=vocab_size)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        video_path = sample["video_path"]
        prompt = sample.get("prompt", "")

        # 1. Process video frames -> Tensor shape: (T, C, H, W) normalized to [-1, 1]
        frames_tensor = self.video_preprocessor.process_video(video_path)
        if self.transform is not None:
            frames_tensor = self.transform(frames_tensor)

        # 2. Process text prompt -> Tensor shape: (L,) int64 tokens
        tokens_tensor = self.text_preprocessor.tokenize_to_tensor(prompt)

        return {
            "id": sample.get("id", f"sample_{idx}"),
            "prompt": prompt,
            "prompt_tokens": tokens_tensor,
            "frames": frames_tensor,  # Shape: (T, C, H, W)
            "duration": float(sample.get("duration", 0.0)),
            "quality_score": torch.tensor(float(sample.get("quality_score", 1.0)), dtype=torch.float32)
        }


def collate_ttv_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator for Text-to-Video batching."""
    frames = torch.stack([item["frames"] for item in batch], dim=0) # (B, T, C, H, W)
    tokens = torch.stack([item["prompt_tokens"] for item in batch], dim=0) # (B, L)
    quality_scores = torch.stack([item["quality_score"] for item in batch], dim=0)
    prompts = [item["prompt"] for item in batch]
    ids = [item["id"] for item in batch]

    return {
        "ids": ids,
        "prompts": prompts,
        "prompt_tokens": tokens,
        "frames": frames,
        "quality_scores": quality_scores
    }


def create_dataloader(
    manifest_path: str,
    batch_size: int = 1,
    num_frames: int = 8,
    height: int = 128,
    width: int = 128,
    max_prompt_length: int = 32,
    vocab_size: int = 10000,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Factory helper to build a PyTorch DataLoader for TTV dataset."""
    dataset = TextToVideoDataset(
        manifest_path=manifest_path,
        num_frames=num_frames,
        height=height,
        width=width,
        max_prompt_length=max_prompt_length,
        vocab_size=vocab_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_ttv_batch
    )
