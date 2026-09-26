"""Side-by-side fixed benchmark evaluator for real generated videos.

It does not fabricate a semantic win. Automated frame-quality metrics are
reported separately from human/VLM rubric scores, which remain null until
provided by a reviewer or a configured evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import cv2

from training.evaluation.evaluator import TTVEvaluator


ROOT = Path(__file__).resolve().parents[2]


def dimensions(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path)); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0); cap.release(); return width, height


def expected_orientation(aspect_ratio: str) -> bool:
    return aspect_ratio == "16:9"


def score_run(video_root: Path, benchmark: Dict[str, Any]) -> Dict[str, Any]:
    evaluator = TTVEvaluator(num_eval_frames=16, height=256, width=256)
    rows = []
    for item in benchmark["prompts"]:
        candidate = next((p for p in video_root.glob(f"{item['id']}.*") if p.suffix.lower() in {".mp4", ".mov", ".webm"}), None)
        if not candidate:
            rows.append({"id": item["id"], "status": "missing"}); continue
        result = evaluator.evaluate_video(str(candidate), item["prompt"])
        width, height = dimensions(candidate)
        rows.append({"id": item["id"], "status": "evaluated", "video": str(candidate.resolve()), "dimensions": [width, height], "aspect_ratio_correct": (width >= height) == expected_orientation(item["aspect_ratio"]), "automated_metrics": result["metrics"], "human_or_vlm_rubric": {key: None for key in benchmark["rating_rubric"]}})
    return {"video_root": str(video_root.resolve()), "samples": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned TTV output with a fixed benchmark")
    parser.add_argument("--base-dir", required=True); parser.add_argument("--fine-tuned-dir", required=True)
    parser.add_argument("--benchmark", default=str(ROOT / "data" / "benchmarks" / "ttv_fixed_v001.json")); parser.add_argument("--output", default=str(ROOT / "training" / "evaluation" / "reports" / "fixed_benchmark_comparison.json"))
    args = parser.parse_args(); benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    report = {"benchmark_version": benchmark["benchmark_version"], "base_model": score_run(Path(args.base_dir), benchmark), "fine_tuned_model": score_run(Path(args.fine_tuned_dir), benchmark), "improvement_claim": "not established: complete the shared 1–5 human/VLM rubric before claiming improvement"}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
