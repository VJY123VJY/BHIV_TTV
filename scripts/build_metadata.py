"""Make a canonical processed metadata JSONL after caption review."""
from __future__ import annotations
import argparse
from pathlib import Path
from dataset.ttv_prepare import _jsonl, _write_jsonl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True); parser.add_argument("--output", default="data/processed/metadata.jsonl")
    args = parser.parse_args()
    rows = [row for row in _jsonl(Path(args.input)) if row.get("caption_status") == "approved" and row.get("caption")]
    _write_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} reviewed records to {args.output}")
