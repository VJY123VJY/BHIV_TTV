import os
import sys
import yaml
import argparse
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.fine_tuning.model import SpatialTemporalTTVModel
from training.rl.feedback_loop import RLFeedbackLoop
from training.rl.rewards import CompositeRewardModel
from training.preprocessing.video_preprocessor import VideoPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Run RL / Direct Preference Feedback Loop")
    parser.add_argument("--config", type=str, default="training/configs/rl_loop.yaml", help="Path to RL YAML config")
    parser.add_argument("--iterations", type=int, default=2, help="Number of feedback loop iterations")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("STARTING RL / DIRECT PREFERENCE FEEDBACK LOOP")
    print("=" * 60)

    # Initialize model
    model = SpatialTemporalTTVModel(
        in_channels=3,
        latent_channels=32,
        num_frames=4,
        vocab_size=1000,
        max_prompt_length=16,
        text_embed_dim=64
    )

    reward_model = CompositeRewardModel(weights=cfg.get("rewards", {}).get("weights"))
    rl_loop = RLFeedbackLoop(
        model=model,
        reward_model=reward_model,
        config=cfg.get("rl", {}),
        output_dir=str(PROJECT_ROOT / cfg.get("output_dir", "training/rl/runs"))
    )

    # Sample prompts and synthetic keyframes for feedback loop execution
    sample_prompts = [
        "A small robot explores a futuristic city at sunset.",
        "A golden retriever dog running on a sunny beach.",
        "An astronaut walking slowly on red sands of Mars."
    ]
    # Synthetic normalized keyframe tensors (3, 128, 128)
    sample_keyframes = [torch.randn(3, 128, 128) for _ in sample_prompts]

    for it in range(1, args.iterations + 1):
        print(f"\n--- Running RL Iteration {it}/{args.iterations} ---")
        res = rl_loop.run_iteration(sample_prompts, sample_keyframes, iteration=it)
        print(f"Iteration {it} Complete:")
        print(f"  Version:       {res['model_version']}")
        print(f"  Winner Reward: {res['mean_reward_winner']:.4f}")
        print(f"  Loser Reward:  {res['mean_reward_loser']:.4f}")
        print(f"  Checkpoint:    {res['saved_checkpoint']}")

    print("\n" + "=" * 60)
    print("RL FEEDBACK LOOP COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
