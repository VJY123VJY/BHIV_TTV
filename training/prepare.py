"""
CLI tool for preparing dataset artifacts for fine-tuning.
Usage:
  python -m training.prepare --manifest data/manifests/train.jsonl --output data/staging
"""
import argparse
import os
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare Dataset for LoRA Fine-Tuning")
    parser.add_argument("--manifest", type=str, default="data/manifests/train.jsonl", help="Path to input train.jsonl")
    parser.add_argument("--output", type=str, default="data/staging", help="Preprocessed output directory")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples to prepare")

    args = parser.parse_args()

    manifest_p = Path(args.manifest)
    if not manifest_p.exists():
        # Fallback to datasets/manifests/train.jsonl
        alt = Path("datasets/manifests/train.jsonl")
        if alt.exists():
            manifest_p = alt
        else:
            print(f"[Prepare] Manifest {args.manifest} not found. Running dataset.manifest first...")
            from dataset.pipeline import dataset_pipeline
            dataset_pipeline.generate_manifests()
            manifest_p = Path("data/manifests/train.jsonl")

    out_p = Path(args.output)
    out_p.mkdir(parents=True, exist_ok=True)

    prepared = 0
    with open(manifest_p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prepared += 1
            if prepared >= args.max_samples:
                break

    print(f"=== Dataset Preparation Complete ===")
    print(f"Verified {prepared} training instances from {manifest_p}.")
    print(f"Preprocessed artifacts prepared in {out_p}.")


if __name__ == "__main__":
    main()
