"""
CLI for dataset quality auditing and corrupted asset cleanup.
Usage:
  python -m dataset.clean
"""
import os
import json
from pathlib import Path
from dataset.quality import assess_image_quality


def main():
    print("=== Auditing Dataset Integrity & Cleaning Low-Quality Assets ===")
    data_dir = Path("data/processed")
    meta_dir = Path("data/metadata")
    rej_dir = Path("data/rejected")
    rej_dir.mkdir(parents=True, exist_ok=True)

    cleaned_count = 0
    audited_count = 0

    for img_file in data_dir.glob("*.jpg"):
        audited_count += 1
        metrics = assess_image_quality(str(img_file))
        if not metrics.passed_qc:
            cleaned_count += 1
            # Move to rejected
            dest = rej_dir / img_file.name
            img_file.rename(dest)
            # Remove associated metadata
            meta_file = meta_dir / f"{img_file.stem}.json"
            if meta_file.exists():
                meta_file.unlink()

    print(f"Audit completed: {audited_count} assets checked, {cleaned_count} quarantined to {rej_dir}.")


if __name__ == "__main__":
    main()
