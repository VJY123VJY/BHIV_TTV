"""Compatibility package so repository-root CLI commands can use ``python -m app``.

The implementation remains in ``backend/app``; this package only exposes that
directory as the package search path and does not duplicate application code.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "backend" / "app")]
