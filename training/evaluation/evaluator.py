import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from training.preprocessing.video_preprocessor import VideoPreprocessor
from training.evaluation.metrics import (
    evaluate_temporal_consistency,
    evaluate_motion_consistency,
    evaluate_visual_quality,
    evaluate_prompt_adherence
)


class TTVEvaluator:
    """
    Comprehensive Evaluator for Text-to-Video generation models.
    Measures prompt adherence, temporal consistency, visual quality, and motion smoothness.
    Produces machine-readable JSON reports and formatted Markdown comparison tables.
    """
    def __init__(self, num_eval_frames: int = 16, height: int = 256, width: int = 256):
        self.preprocessor = VideoPreprocessor(num_frames=num_eval_frames, height=height, width=width)

    def evaluate_video(self, video_path: str, prompt: str) -> Dict[str, Any]:
        """Runs full metric evaluation suite on a single video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        raw_frames = self.preprocessor.extract_raw_frames(video_path)
        if not raw_frames:
            raise RuntimeError(f"Could not extract frames from {video_path}")

        prompt_score = evaluate_prompt_adherence(prompt, raw_frames)
        temporal_score = evaluate_temporal_consistency(raw_frames)
        visual_score = evaluate_visual_quality(raw_frames)
        motion_score = evaluate_motion_consistency(raw_frames)

        # Composite weighted quality score
        composite = (
            0.30 * prompt_score +
            0.30 * temporal_score +
            0.20 * visual_score +
            0.20 * motion_score
        )

        return {
            "video_path": video_path,
            "prompt": prompt,
            "frame_count_evaluated": len(raw_frames),
            "metrics": {
                "prompt_adherence": round(prompt_score, 4),
                "temporal_consistency": round(temporal_score, 4),
                "visual_quality": round(visual_score, 4),
                "motion_consistency": round(motion_score, 4),
                "composite_score": round(composite, 4)
            }
        }

    def evaluate_dataset(self, manifest_path: str) -> Dict[str, Any]:
        """Evaluates all samples listed in a dataset manifest."""
        samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))

        per_sample_results = []
        metric_accumulators = {
            "prompt_adherence": [],
            "temporal_consistency": [],
            "visual_quality": [],
            "motion_consistency": [],
            "composite_score": []
        }

        for sample in samples:
            video_path = sample.get("video_path")
            prompt = sample.get("prompt", "")
            if not video_path or not os.path.exists(video_path):
                continue

            try:
                res = self.evaluate_video(video_path, prompt)
                per_sample_results.append(res)
                for k, v in res["metrics"].items():
                    metric_accumulators[k].append(v)
            except Exception as e:
                print(f"[Warning] Failed to evaluate {video_path}: {e}")

        summary = {}
        for k, vals in metric_accumulators.items():
            summary[f"{k}_mean"] = round(float(np.mean(vals)), 4) if vals else 0.0
            summary[f"{k}_std"] = round(float(np.std(vals)), 4) if vals else 0.0

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "evaluated_samples": len(per_sample_results),
            "summary": summary,
            "per_sample": per_sample_results
        }

    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        output_path: str,
        title: str = "TTV Model Evaluation Report"
    ) -> str:
        """Generates formatted GitHub-style Markdown report."""
        summary = results.get("summary", {})
        md = []
        md.append(f"# {title}\n")
        md.append(f"**Generated:** {results.get('timestamp')}")
        md.append(f"**Total Evaluated Samples:** {results.get('evaluated_samples')}\n")
        md.append("## Aggregate Summary Metrics\n")
        md.append("| Metric | Mean Score | Std Dev |")
        md.append("|---|---|---|")
        md.append(f"| **Prompt Adherence** | {summary.get('prompt_adherence_mean', 0.0):.4f} | {summary.get('prompt_adherence_std', 0.0):.4f} |")
        md.append(f"| **Temporal Consistency** | {summary.get('temporal_consistency_mean', 0.0):.4f} | {summary.get('temporal_consistency_std', 0.0):.4f} |")
        md.append(f"| **Visual Quality** | {summary.get('visual_quality_mean', 0.0):.4f} | {summary.get('visual_quality_std', 0.0):.4f} |")
        md.append(f"| **Motion Consistency** | {summary.get('motion_consistency_mean', 0.0):.4f} | {summary.get('motion_consistency_std', 0.0):.4f} |")
        md.append(f"| **Composite Score** | **{summary.get('composite_score_mean', 0.0):.4f}** | {summary.get('composite_score_std', 0.0):.4f} |\n")

        md.append("## Per-Sample Breakdown\n")
        md.append("| Sample Video | Prompt | Prompt Adherence | Temporal | Visual Quality | Composite |")
        md.append("|---|---|---|---|---|---|")
        for s in results.get("per_sample", [])[:15]:
            p = Path(s["video_path"]).name
            prompt_short = (s["prompt"][:40] + "...") if len(s["prompt"]) > 40 else s["prompt"]
            m = s["metrics"]
            md.append(f"| `{p}` | {prompt_short} | {m['prompt_adherence']} | {m['temporal_consistency']} | {m['visual_quality']} | **{m['composite_score']}** |")

        content = "\n".join(md)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return content

    @staticmethod
    def compare_evaluations(
        baseline_report: Dict[str, Any],
        finetuned_report: Dict[str, Any],
        output_path: str
    ) -> str:
        """Compares baseline vs fine-tuned performance in a clear before-and-after table."""
        b_sum = baseline_report.get("summary", {})
        f_sum = finetuned_report.get("summary", {})

        metrics = ["prompt_adherence", "temporal_consistency", "visual_quality", "motion_consistency", "composite_score"]
        labels = ["Prompt Adherence", "Temporal Consistency", "Visual Quality", "Motion Consistency", "Overall Composite Score"]

        md = []
        md.append("# Model Fine-Tuning Performance Comparison\n")
        md.append("| Evaluation Dimension | Baseline Model | Fine-Tuned Model | Delta (Absolute) | Relative Change |")
        md.append("|---|---|---|---|---|")

        for m, lbl in zip(metrics, labels):
            b_val = b_sum.get(f"{m}_mean", 0.0)
            f_val = f_sum.get(f"{m}_mean", 0.0)
            delta = f_val - b_val
            pct = (delta / b_val * 100.0) if b_val > 0 else 0.0
            sign = "+" if delta >= 0 else ""
            md.append(f"| **{lbl}** | {b_val:.4f} | {f_val:.4f} | {sign}{delta:.4f} | {sign}{pct:.2f}% |")

        content = "\n".join(md)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return content
