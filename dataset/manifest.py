"""
CLI for generating versioned train/validation/test JSONL manifests and quality reports.
Usage:
  python -m dataset.manifest
"""
from dataset.pipeline import dataset_pipeline


def main():
    print("=== Generating Dataset Manifests (train.jsonl, validation.jsonl, test.jsonl) ===")
    report = dataset_pipeline.generate_manifests()
    print("=== Dataset Manifest Summary ===")
    print(f"Total Assets:        {report.total_downloaded}")
    print(f"Training Eligible:   {report.training_ready}")
    print(f"Quality Filtered:    {report.low_quality}")
    print(f"Categories Breakdown: {report.categories}")
    print("Manifests generated in data/manifests/ and datasets/manifests/")
    print("Quality report written to dataset_report.json")


if __name__ == "__main__":
    main()
