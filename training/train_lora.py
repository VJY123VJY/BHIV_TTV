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
    parser.add_argument("--reference-id", type=str, default=None, help="Reference ID to condition/fine-tune with")
    parser.add_argument("--reference-path", type=str, default=None, help="Direct path to reference image/video")
    parser.add_argument("--reference-url", type=str, default=None, help="Reference public media URL")

    args = parser.parse_args()

    # Pass through to existing trainer
    sys.argv = [sys.argv[0], "--config", args.config, "--version-name", args.version_name]
    if args.resume:
        sys.argv.extend(["--resume", args.resume])
    if args.reference_id:
        sys.argv.extend(["--reference-id", args.reference_id])
    if args.reference_path:
        sys.argv.extend(["--reference-path", args.reference_path])
    if args.reference_url:
        sys.argv.extend(["--reference-url", args.reference_url])

    print(f"=== Starting LoRA Fine-Tuning with config: {args.config} ===")
    run_training_main()


if __name__ == "__main__":
    main()
