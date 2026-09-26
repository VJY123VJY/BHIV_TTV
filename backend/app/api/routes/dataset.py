"""Read-only dataset/training dashboard data; it never starts a training job."""
from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/api/v1/dataset/dashboard")
async def dataset_dashboard():
    root = settings.get_absolute_path(settings.DATASET_STORAGE)
    report_path = root / "dataset_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    except (OSError, ValueError):
        report = {}
    try:
        from training.utils.hardware import detect_hardware
        hardware = detect_hardware()
    except Exception:
        hardware = {"cuda_available": False, "recommended_device": "unknown"}
    split_counts = {}
    for split in ("train", "validation", "test"):
        manifest = root / split / "metadata.jsonl"
        split_counts[split] = sum(1 for _ in manifest.open(encoding="utf-8")) if manifest.exists() else 0
    return {
        "dataset": report,
        "splits": split_counts,
        "hardware": hardware,
        "training_enabled": settings.ENABLE_DATASET_TRAINING,
        "note": "Dashboard values are read from the versioned report; no metrics imply model improvement."
    }
