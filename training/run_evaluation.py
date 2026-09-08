import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluation.evaluator import TTVEvaluator
from training.datasets.manifest_builder import build_manifest_from_directory


def main():
    parser = argparse.ArgumentParser(description="Evaluate TTV Generation Performance")
    parser.add_argument("--manifest", type=str, default="training/datasets/dataset.jsonl", help="Dataset manifest path")
    parser.add_argument("--video", type=str, default=None, help="Evaluate single video file")
    parser.add_argument("--prompt", type=str, default="Cinematic scene", help="Prompt for single video evaluation")
    parser.add_argument("--output-dir", type=str, default="training/evaluation/reports", help="Directory to store reports")
    parser.add_argument("--compare-baseline", type=str, default=None, help="Path to baseline evaluation JSON to compare against")
    args = parser.parse_args()

    evaluator = TTVEvaluator(num_eval_frames=16, height=256, width=256)
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        res = evaluator.evaluate_video(args.video, args.prompt)
        print("Single Video Evaluation Results:")
        print(json.dumps(res, indent=2))
        return

    # Dataset evaluation
    manifest_path = PROJECT_ROOT / args.manifest
    if not manifest_path.exists():
        print(f"[Dataset] Manifest {manifest_path} not found. Automatically building from generated/videos...")
        build_manifest_from_directory(
            videos_dir=str(PROJECT_ROOT / "generated" / "videos"),
            output_manifest_path=str(manifest_path)
        )

    print(f"Starting dataset evaluation on {manifest_path}...")
    results = evaluator.evaluate_dataset(str(manifest_path))

    # Save JSON report
    json_path = out_dir / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown report
    md_path = out_dir / "evaluation_report.md"
    evaluator.generate_markdown_report(results, str(md_path))

    print(f"Evaluation complete. Reports generated:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    # Optional baseline comparison
    if args.compare_baseline and os.path.exists(args.compare_baseline):
        with open(args.compare_baseline, "r", encoding="utf-8") as bf:
            base_data = json.load(bf)
        comp_md = out_dir / "model_comparison.md"
        evaluator.compare_evaluations(base_data, results, str(comp_md))
        print(f"  Comparison Report: {comp_md}")


if __name__ == "__main__":
    main()
