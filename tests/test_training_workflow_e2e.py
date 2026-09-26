import os
import sys
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend and project root are in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))

from backend.app.main import app
from backend.app.services.training_service import (
    parse_manifest_records,
    validate_manifest_records,
    check_cuda_availability,
    training_job_manager
)

client = TestClient(app)

MANIFEST_120_PATH = BASE_DIR / "data" / "manifests" / "ttv_training_manifest_120.jsonl"
if not MANIFEST_120_PATH.exists():
    MANIFEST_120_PATH = BASE_DIR.parent / "ttv_training_manifest_120.jsonl"


def test_120_record_manifest_file_exists_and_unmodified():
    """Ensure the 120-record manifest is preserved intact with exactly 120 lines."""
    assert MANIFEST_120_PATH.exists(), f"120 manifest not found at {MANIFEST_120_PATH}"
    lines = [l for l in MANIFEST_120_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 120, f"Expected 120 lines, got {len(lines)}"


def test_complete_training_workflow_with_120_manifest():
    """
    Test the complete production training workflow with the 120-record manifest:
    1. Validate manifest via API
    2. Submit training configuration
    3. Verify CUDA_UNAVAILABLE status on CPU host
    4. Verify reviewed CLI fallback command
    5. Poll status and logs
    """
    os.environ.pop("TTV_FORCE_CUDA_AVAILABLE", None)

    # 1. Validation check
    with open(MANIFEST_120_PATH, "rb") as f:
        val_resp = client.post(
            "/api/v1/training/manifest/validate",
            files={"manifest_file": ("ttv_training_manifest_120.jsonl", f, "application/json")}
        )
    assert val_resp.status_code == 200
    val_data = val_resp.json()
    assert val_data["valid"] is True
    assert val_data["total_records"] == 120
    assert val_data["valid_records"] == 120
    assert val_data["splits"]["train"] == 90
    assert val_data["splits"]["val"] == 18
    assert val_data["splits"]["test"] == 12

    # 2. Launch / Queue job via API
    with open(MANIFEST_120_PATH, "rb") as f:
        launch_resp = client.post(
            "/api/v1/training/jobs",
            files={"manifest_file": ("ttv_training_manifest_120.jsonl", f, "application/json")},
            data={
                "base_model": "SpatialTemporalTTVModel",
                "method": "lora",
                "epochs": "10",
                "learning_rate": "0.0001",
                "batch_size": "2",
                "version_name": "prod_lora_120_v001"
            }
        )
    assert launch_resp.status_code == 200
    job = launch_resp.json()
    job_id = job["job_id"]

    # Verify status is clearly CUDA_UNAVAILABLE
    assert job["status"] == "CUDA_UNAVAILABLE"
    assert job["cuda_available"] is False
    assert job["progress"] == 0
    assert "cli_fallback_command" in job
    assert "python training/run_training.py" in job["cli_fallback_command"]
    assert job["manifest_stats"]["total_records"] == 120

    # 3. Poll job status
    status_resp = client.get(f"/api/v1/training/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] == "CUDA_UNAVAILABLE"
    assert status_data["progress"] == 0

    # 4. Poll job logs
    logs_resp = client.get(f"/api/v1/training/jobs/{job_id}/logs")
    assert logs_resp.status_code == 200
    logs_data = logs_resp.json()
    assert logs_data["job_id"] == job_id
    assert logs_data["status"] == "CUDA_UNAVAILABLE"
    logs = logs_data["logs"]
    assert any("CUDA_UNAVAILABLE" in line for line in logs)
    assert any("Manual Fallback" in line for line in logs)
    assert any("120 records" in line for line in logs)


def test_ui_distinguishes_all_four_training_states():
    """
    Verify backend status outputs that correspond to the four required UI states:
    1. 'job queued'
    2. 'training running'
    3. 'CUDA unavailable'
    4. 'manual CLI required'
    """
    # Test CUDA_UNAVAILABLE and manual CLI fallback
    os.environ.pop("TTV_FORCE_CUDA_AVAILABLE", None)
    res_cuda_off = client.post("/api/v1/training/jobs", json={
        "manifest_path": str(MANIFEST_120_PATH),
        "epochs": 2
    })
    assert res_cuda_off.status_code == 200
    job_off = res_cuda_off.json()
    assert job_off["status"] == "CUDA_UNAVAILABLE"
    assert "python training/run_training.py" in job_off["cli_fallback_command"]

    # Test CUDA available (queued / running)
    os.environ["TTV_FORCE_CUDA_AVAILABLE"] = "1"
    try:
        res_cuda_on = client.post("/api/v1/training/jobs", json={
            "manifest_path": "training/datasets/dataset.jsonl",
            "epochs": 1
        })
        assert res_cuda_on.status_code == 200
        job_on = res_cuda_on.json()
        assert job_on["status"] in ["queued", "running", "completed"]
        assert job_on["cuda_available"] is True
    finally:
        os.environ.pop("TTV_FORCE_CUDA_AVAILABLE", None)


def test_frontend_serves_updated_training_interface():
    """Verify that root / serves index.html with the training controls and fallback UI."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "start-training-btn" in html
    assert "Launch Training" in html
    assert "training-session-status" in html
    assert "cli-fallback-card" in html
    assert "training-terminal-log" in html
