import os
import sys
import tempfile
import numpy as np
import cv2
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.adapters.video import get_video_adapter
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.adapters.video.neural_video_adapter import NeuralVideoAdapter
from app.models.scene import Scene
from training.datasets.validator import probe_video_integrity


def test_video_adapter_mode_switching():
    # 1. Base Mode
    settings.MODEL_MODE = "base"
    adapter_base = get_video_adapter()
    assert isinstance(adapter_base, OpenCVVideoAdapter)

    # 2. Fine-Tuned Mode
    settings.MODEL_MODE = "finetuned"
    adapter_finetuned = get_video_adapter()
    assert isinstance(adapter_finetuned, NeuralVideoAdapter)

    # Restore default
    settings.MODEL_MODE = "base"


@pytest.mark.asyncio
async def test_neural_video_adapter_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample keyframe
        kf_path = os.path.join(tmpdir, "keyframe.jpg")
        kf_img = np.full((128, 128, 3), 150, dtype=np.uint8)
        cv2.imwrite(kf_path, kf_img)

        out_clip = os.path.join(tmpdir, "output_clip.mp4")
        adapter = NeuralVideoAdapter()

        scene = Scene(
            index=1,
            title="Scene 1",
            narrative="A small rover moves across the dunes.",
            visual_description="Cinematic establishing shot of Mars rover",
            duration=2.0,
            image_path=kf_path
        )

        result_path = await adapter.generate_scene_video(
            scene=scene,
            keyframe_path=kf_path,
            output_path=out_clip,
            width=128,
            height=128,
            fps=24
        )

        assert os.path.exists(result_path)
        is_valid, info = probe_video_integrity(result_path)
        assert is_valid is True
        assert info["frame_count"] > 0
        assert info["duration_seconds"] >= 1.5
