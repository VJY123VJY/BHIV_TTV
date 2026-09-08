import os
import sys
import tempfile
import numpy as np
import cv2
import torch
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from training.preprocessing.video_preprocessor import VideoPreprocessor
from training.preprocessing.text_preprocessor import TextPreprocessor


def test_video_preprocessor():
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "sample.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vid_path, fourcc, 24.0, (160, 120))
        for i in range(24):
            frame = np.full((120, 160, 3), i * 10, dtype=np.uint8)
            out.write(frame)
        out.release()

        preprocessor = VideoPreprocessor(num_frames=6, height=64, width=64, normalize_range=(-1.0, 1.0))
        tensor = preprocessor.process_video(vid_path)

        assert tensor.shape == (6, 3, 64, 64)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

        # Test denormalization
        denorm = preprocessor.denormalize_tensor(tensor)
        assert denorm.shape == (6, 64, 64, 3)
        assert denorm.dtype == np.uint8


def test_text_preprocessor():
    tp = TextPreprocessor(max_length=16, vocab_size=1000)
    prompt = "A Small Robot  Explores A Cyberpunk City at Sunset!!  "

    clean = tp.clean_text(prompt)
    assert clean == "a small robot explores a cyberpunk city at sunset!!"

    tokens = tp.tokenize(clean)
    assert "robot" in tokens
    assert "cyberpunk" in tokens

    token_ids = tp.tokenize_to_ids(prompt)
    assert len(token_ids) == 16
    assert token_ids[0] == tp.bos_token_id
    assert tp.eos_token_id in token_ids

    tensor = tp.tokenize_to_tensor(prompt)
    assert tensor.shape == (16,)
    assert tensor.dtype == torch.long
