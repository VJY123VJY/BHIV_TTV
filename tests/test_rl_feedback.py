import os
import sys
import tempfile
import numpy as np
import torch
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from training.fine_tuning.model import SpatialTemporalTTVModel
from training.rl.rewards import (
    PromptAdherenceRewardModel,
    TemporalSmoothnessRewardModel,
    AestheticVisualRewardModel,
    HumanPreferenceRewardModel,
    CompositeRewardModel
)
from training.rl.feedback_loop import RLFeedbackLoop


def test_reward_models():
    frames = [np.full((64, 64, 3), 120, dtype=np.uint8) for _ in range(4)]
    prompt = "A robot walking on sunny beach"

    r_prompt = PromptAdherenceRewardModel().calculate_reward(prompt, frames)
    assert 0.0 <= r_prompt <= 1.0

    r_temp = TemporalSmoothnessRewardModel().calculate_reward(prompt, frames)
    assert 0.0 <= r_temp <= 1.0

    r_aes = AestheticVisualRewardModel().calculate_reward(prompt, frames)
    assert 0.0 <= r_aes <= 1.0

    r_human = HumanPreferenceRewardModel().calculate_reward(prompt, frames, {"human_rating": 4.5})
    assert r_human == (4.5 - 1.0) / 4.0

    comp = CompositeRewardModel()
    r_total = comp.calculate_reward(prompt, frames, {"human_rating": 4.0})
    assert 0.0 <= r_total <= 1.0


def test_rl_feedback_loop_step():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SpatialTemporalTTVModel(
            in_channels=3,
            latent_channels=16,
            num_frames=4,
            vocab_size=500,
            max_prompt_length=16,
            text_embed_dim=32
        )

        rl_loop = RLFeedbackLoop(
            model=model,
            config={
                "num_candidates": 2,
                "beta": 0.1,
                "learning_rate": 1e-4,
                "lora_rank": 4,
                "max_prompt_length": 16,
                "num_frames": 4,
                "height": 64,
                "width": 64
            },
            output_dir=tmpdir
        )

        prompts = ["A robot exploring city."]
        keyframes = [torch.randn(3, 64, 64)]

        res = rl_loop.run_iteration(prompts, keyframes, iteration=1)

        assert res["iteration"] == 1
        assert len(res["trajectories"]) == 1
        assert "winner_reward" in res["trajectories"][0]
        assert "loser_reward" in res["trajectories"][0]
        assert os.path.exists(res["saved_checkpoint"])
