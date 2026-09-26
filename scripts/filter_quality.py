"""Quality filtering is part of strict provenance validation."""
from __future__ import annotations
import sys
from dataset.ttv_prepare import main
if __name__ == "__main__":
    sys.argv.insert(1, "validate"); main()
