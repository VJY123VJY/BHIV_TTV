"""
CLI tool for evaluating fine-tuned models on test manifests.
Usage:
  python -m training.evaluate --config training/configs/smoke_test.yaml
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.run_evaluation import main as run_eval_main


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned TTV Checkpoints")
    parser.add_argument("--config", type=str, default="training/configs/smoke_test.yaml", help="Path to evaluation config")
    parser.add_argument("--checkpoint", type=str, default="models/lora/checkpoint_final.pt", help="Path to checkpoint")

    args = parser.parse_args()

    sys.argv = [sys.argv[0], "--config", args.config, "--checkpoint", args.checkpoint]
    print(f"=== Evaluating Checkpoint {args.checkpoint} ===")
    run_eval_main()


if __name__ == "__main__":
    main()
