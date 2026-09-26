"""Collect only registry-approved Wikimedia candidates; no arbitrary URL mode."""
from __future__ import annotations
import sys
from dataset.ingest import main

if __name__ == "__main__":
    if "--source" not in sys.argv:
        sys.argv.extend(["--source", "wikimedia"])
    main()
