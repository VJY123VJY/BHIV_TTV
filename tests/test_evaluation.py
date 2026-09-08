import os
import sys
import tempfile
import numpy as np
import cv2
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from training.evaluation.metrics import (
    calculate_sharpness,
    calculate_contrast,
    calculate_frame_ssim,
    evaluate_temporal_consistency,
    evaluate_motion_consistency,
    evaluate_visual_quality,
    evaluate_prompt_adherence
)
from training.evaluation.evaluator import TTVEvaluator


def test_individual_metrics():
    # Identical frames should yield SSIM 1.0
    f1 = np.full((64, 64, 3), 128, dtype=np.uint8)
    f2 = np.full((64, 64, 3), 128, dtype=np.uint8)
    ssim = calculate_frame_ssim(f1, f2)
    assert ssim == 1.0

    frames = [np.full((64, 64, 3), int(100 + i * 5), dtype=np.uint8) for i in range(5)]
    temp_score = evaluate_temporal_consistency(frames)
    assert 0.0 <= temp_score <= 1.0

    mot_score = evaluate_motion_consistency(frames)
    assert 0.0 <= mot_score <= 1.0

    vis_score = evaluate_visual_quality(frames)
    assert 0.0 <= vis_score <= 1.0

    prompt_score = evaluate_prompt_adherence("Mars rover dunes", frames)
    assert 0.0 <= prompt_score <= 1.0


def test_evaluator_video_and_report_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "eval_sample.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vid_path, fourcc, 24.0, (64, 64))
        for i in range(24):
            frame = np.full((64, 64, 3), (i * 10) % 255, dtype=np.uint8)
            out.write(frame)
        out.release()

        evaluator = TTVEvaluator(num_eval_frames=6, height=64, width=64)
        res = evaluator.evaluate_video(vid_path, "A colourful abstract video")

        assert "metrics" in res
        m = res["metrics"]
        assert "prompt_adherence" in m
        assert "temporal_consistency" in m
        assert "visual_quality" in m
        assert "composite_score" in m

        # Test Markdown report generation
        md_out = os.path.join(tmpdir, "report.md")
        dataset_res = {
            "timestamp": "2026-09-07T12:00:00Z",
            "evaluated_samples": 1,
            "summary": {
                "prompt_adherence_mean": m["prompt_adherence"],
                "prompt_adherence_std": 0.0,
                "temporal_consistency_mean": m["temporal_consistency"],
                "temporal_consistency_std": 0.0,
                "visual_quality_mean": m["visual_quality"],
                "visual_quality_std": 0.0,
                "motion_consistency_mean": m["motion_consistency"],
                "motion_consistency_std": 0.0,
                "composite_score_mean": m["composite_score"],
                "composite_score_std": 0.0
            },
            "per_sample": [res]
        }
        md_content = evaluator.generate_markdown_report(dataset_res, md_out)
        assert os.path.exists(md_out)
        assert "Composite Score" in md_content
