"""
CLI pipeline diagnostic tool:
python -m backend.tools.diagnose_pipeline "Your prompt here"
"""
import sys
import json
import argparse
from pathlib import Path

# Add project root and backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.pipeline_diagnostic import diagnose_pipeline_execution


def main():
    parser = argparse.ArgumentParser(description="TTV Studio Pipeline Diagnostic Tool")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="A highly realistic cinematic video of an Indian farmer walking slowly through a lush green agricultural field during sunrise.",
        help="Prompt text to diagnose",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed value")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--resolution", type=str, default="1280x720", help="Resolution")

    args = parser.parse_args()

    print(f"\n==================================================")
    print(f" Diagnosing TTV Pipeline for Prompt:")
    print(f" '{args.prompt}'")
    print(f"==================================================\n")

    result = diagnose_pipeline_execution(
        prompt=args.prompt,
        seed=args.seed,
        fps=args.fps,
        num_frames=args.num_frames,
        resolution=args.resolution,
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
