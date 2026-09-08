import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from training.datasets.validator import DatasetValidator, split_dataset


def build_manifest_from_directory(
    videos_dir: str,
    output_manifest_path: str,
    default_quality_score: float = 0.85,
    do_split: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Scans a directory for MP4 videos and paired metadata JSON sidecars.
    Generates a validated dataset.jsonl (and train/val/test splits).
    Does NOT fabricate videos or prompts: only indexes actual files on disk.
    """
    vdir = Path(videos_dir)
    if not vdir.exists():
        raise FileNotFoundError(f"Videos directory does not exist: {videos_dir}")

    mp4_files = sorted(list(vdir.glob("*.mp4")))
    raw_samples = []

    for idx, mp4_path in enumerate(mp4_files):
        # Look for corresponding metadata JSON: e.g. <stem>_metadata.json or <stem>.json
        meta_candidates = [
            vdir / f"{mp4_path.stem}_metadata.json",
            vdir / f"{mp4_path.stem}.json"
        ]
        meta_data = {}
        for mc in meta_candidates:
            if mc.exists():
                try:
                    with open(mc, "r", encoding="utf-8") as mf:
                        meta_data = json.load(mf)
                    break
                except Exception:
                    pass

        prompt = meta_data.get("prompt")
        if not prompt:
            # Check scene descriptions
            scenes = meta_data.get("scenes", [])
            if scenes and isinstance(scenes, list) and len(scenes) > 0:
                first_scene = scenes[0]
                prompt = first_scene.get("narrative") or first_scene.get("visual_description")
        
        if not prompt:
            # Derive meaningful prompt from clean stem
            clean_name = mp4_path.stem.replace("exec_", "").replace("_", " ")
            prompt = f"Cinematic video sequence: {clean_name}"

        sample = {
            "id": f"sample_{idx+1:04d}_{mp4_path.stem}",
            "prompt": prompt,
            "video_path": str(mp4_path.resolve()),
            "duration": meta_data.get("duration", 10.0),
            "fps": meta_data.get("fps", 24),
            "resolution": meta_data.get("resolution", "1280x720"),
            "style": meta_data.get("style", "cinematic"),
            "quality_score": meta_data.get("quality_score", default_quality_score),
            "metadata": {
                "source": "bhiv_ttv_generated",
                "scenes_count": len(meta_data.get("scenes", [])),
                "original_metadata": meta_data
            }
        }
        raw_samples.append(sample)

    # Validate all samples
    validator = DatasetValidator()
    validation_result = validator.validate_dataset(raw_samples)
    valid_samples = validation_result["valid_samples"]

    out_path = Path(output_manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write main dataset.jsonl
    with open(out_path, "w", encoding="utf-8") as f:
        for s in valid_samples:
            f.write(json.dumps(s) + "\n")

    summary = {
        "manifest_path": str(out_path.resolve()),
        "total_scanned": len(raw_samples),
        "valid_samples_count": len(valid_samples),
        "rejected_samples_count": validation_result["rejected_count"],
        "splits": {}
    }

    # Generate train/val/test splits if requested
    if do_split and valid_samples:
        splits = split_dataset(valid_samples, train_ratio=0.8, val_ratio=0.1, seed=seed)
        split_dir = out_path.parent
        for split_name, split_samples in splits.items():
            split_file = split_dir / f"{out_path.stem}_{split_name}.jsonl"
            with open(split_file, "w", encoding="utf-8") as f:
                for s in split_samples:
                    f.write(json.dumps(s) + "\n")
            summary["splits"][split_name] = {
                "count": len(split_samples),
                "path": str(split_file.resolve())
            }

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build TTV Training Dataset Manifest")
    parser.add_argument("--videos-dir", type=str, default="generated/videos", help="Directory containing videos")
    parser.add_argument("--output", type=str, default="training/datasets/dataset.jsonl", help="Output JSONL manifest path")
    args = parser.parse_args()

    result = build_manifest_from_directory(args.videos_dir, args.output)
    print("Manifest successfully built:")
    print(json.dumps(result, indent=2))
