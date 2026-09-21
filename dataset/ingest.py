"""
CLI for dataset ingestion.
Usage:
  python -m dataset.ingest --source wikimedia --query "farmer speaking" --limit 10 --license CC-BY
"""
import argparse
import sys
from dataset.pipeline import dataset_pipeline


def main():
    parser = argparse.ArgumentParser(description="BHIV TTV Dataset Ingestion Tool")
    parser.add_argument("--source", type=str, default="synthetic", choices=["wikimedia", "archive_org", "synthetic"], help="Data source")
    parser.add_argument("--query", type=str, default="people speaking nature farm", help="Search query or visual theme")
    parser.add_argument("--limit", type=int, default=10, help="Maximum items to ingest")
    parser.add_argument("--license", type=str, default=None, help="Optional exact license family filter (e.g. CC-BY, CC0); omitted means any approved license")
    parser.add_argument("--min-resolution", type=str, default="512x512", help="Minimum resolution WIDTHxHEIGHT")
    parser.add_argument("--media-type", type=str, default="image", choices=["image", "video"], help="Media type")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")

    args = parser.parse_args()

    print(f"=== Starting Ingestion: source={args.source}, query='{args.query}', limit={args.limit}, license={args.license} ===")
    try:
        min_width, min_height = (int(v) for v in args.min_resolution.lower().split("x", 1))
    except ValueError:
        parser.error("--min-resolution must be WIDTHxHEIGHT, for example 1280x720")
    assets = dataset_pipeline.ingest_assets(
        source=args.source,
        query=args.query,
        limit=args.limit,
        license_filter=args.license,
        media_type=args.media_type,
        min_resolution=(min_width, min_height),
    )
    print(f"Successfully processed and committed {len(assets)} verified assets to {args.output}.")

    # Update manifests
    report = dataset_pipeline.generate_manifests()
    print(f"Manifests updated. Total training-ready assets: {report.training_ready}")


if __name__ == "__main__":
    main()
