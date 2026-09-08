import os
import sys
import tempfile
import json
import numpy as np
import cv2
import pytest
from pathlib import Path

# Add project root and backend to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))

from training.datasets.validator import (
    DatasetValidator,
    probe_video_integrity,
    split_dataset,
    compute_video_hash
)
from training.datasets.manifest_builder import build_manifest_from_directory
from training.datasets.loader import TextToVideoDataset, create_dataloader


def create_dummy_video(path: str, duration_sec: float = 2.0, fps: int = 24, width: int = 128, height: int = 128):
    """Helper to create a small valid MP4 video for tests."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        # Create shifting color frame
        frame = np.full((height, width, 3), (i * 5) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_probe_video_integrity():
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "valid_video.mp4")
        create_dummy_video(vid_path, duration_sec=1.5, fps=24)

        is_valid, info = probe_video_integrity(vid_path)
        assert is_valid is True
        assert info["width"] == 128
        assert info["height"] == 128
        assert info["frame_count"] > 0
        assert info["duration_seconds"] >= 1.0


def test_corrupted_and_missing_video_detection():
    validator = DatasetValidator(min_duration=0.5, max_duration=10.0, min_prompt_len=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Missing video
        missing_sample = {"prompt": "Valid prompt", "video_path": os.path.join(tmpdir, "missing.mp4")}

        # 2. Corrupted (0 byte) video
        corrupted_path = os.path.join(tmpdir, "corrupted.mp4")
        with open(corrupted_path, "wb") as f:
            f.write(b"not a valid video stream")
        corrupted_sample = {"prompt": "Valid prompt", "video_path": corrupted_path}

        # 3. Valid video
        valid_path = os.path.join(tmpdir, "valid.mp4")
        create_dummy_video(valid_path, duration_sec=1.0)
        valid_sample = {"prompt": "A running dog in park", "video_path": valid_path}

        res = validator.validate_dataset([missing_sample, corrupted_sample, valid_sample])
        assert res["total_evaluated"] == 3
        assert res["valid_count"] == 1
        assert res["rejected_count"] == 2
        assert res["valid_samples"][0]["prompt"] == "A running dog in park"


def test_split_dataset_reproducibility():
    dummy_samples = [{"id": f"s_{i}", "prompt": f"prompt {i}"} for i in range(10)]
    split_1 = split_dataset(dummy_samples, train_ratio=0.8, val_ratio=0.1, seed=42)
    split_2 = split_dataset(dummy_samples, train_ratio=0.8, val_ratio=0.1, seed=42)

    assert len(split_1["train"]) == len(split_2["train"])
    assert [s["id"] for s in split_1["train"]] == [s["id"] for s in split_2["train"]]


def test_manifest_builder_and_dataloader():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 2 sample videos
        v1 = os.path.join(tmpdir, "test1.mp4")
        v2 = os.path.join(tmpdir, "test2.mp4")
        create_dummy_video(v1, duration_sec=1.0)
        create_dummy_video(v2, duration_sec=1.0)

        # Create sidecar json for v1
        meta1 = {"prompt": "A robot walking forward", "duration": 1.0, "fps": 24}
        with open(os.path.join(tmpdir, "test1_metadata.json"), "w") as mf:
            json.dump(meta1, mf)

        manifest_out = os.path.join(tmpdir, "test_manifest.jsonl")
        summary = build_manifest_from_directory(tmpdir, manifest_out, do_split=True)

        assert summary["valid_samples_count"] == 2
        assert os.path.exists(manifest_out)

        # Test PyTorch DataLoader
        loader = create_dataloader(manifest_out, batch_size=2, num_frames=4, height=64, width=64)
        for batch in loader:
            assert batch["frames"].shape == (2, 4, 3, 64, 64)
            assert batch["prompt_tokens"].shape == (2, 32)
            assert len(batch["prompts"]) == 2
            break
