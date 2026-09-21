"""
CLI tool for running LoRA adapter fine-tuning on prepared dataset.
Usage:
  python -m training.train_lora --config training/configs/smoke_test.yaml --epochs 1
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.run_training import main as run_training_main


def main():
    parser = argparse.ArgumentParser(description="BHIV TTV LoRA Adapter Training Tool")
    parser.add_argument("--config", type=str, default="training/configs/smoke_test.yaml", help="Path to training YAML config")
    parser.add_argument("--version-name", type=str, default="ttv_lora_v001", help="Version identifier for registered checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()

    # Pass through to existing trainer
    sys.argv = [sys.argv[0], "--config", args.config, "--version-name", args.version_name]
    if args.resume:
        sys.argv.extend(["--resume", args.resume])

    print(f"=== Starting LoRA Fine-Tuning with config: {args.config} ===")
    run_training_main()


if __name__ == "__main__":
    main()
