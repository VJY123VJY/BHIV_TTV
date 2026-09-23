import io
import os
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.services.reference_service import reference_service
from training.datasets.manifest_builder import prepare_reference_dataset_entry
from training.datasets.loader import create_dataloader

client = TestClient(app)


def _create_test_image_bytes() -> bytes:
    img = Image.new("RGB", (640, 480), color=(255, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_reference_upload_image():
    content = _create_test_image_bytes()
    response = client.post(
        "/api/v1/references/upload",
        files={"file": ("test_ref.jpg", content, "image/jpeg")},
        data={"reference_type": "image"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reference_id"].startswith("ref_")
    assert data["media_type"] == "image"
    assert data["source"] == "upload"
    assert data["preview_url"] is not None


def test_reference_upload_invalid_type():
    response = client.post(
        "/api/v1/references/upload",
        files={"file": ("malicious.exe", b"binary content", "application/x-msdownload")}
    )
    assert response.status_code == 400
    assert "Unsupported reference file type" in response.json().get("message", "")


def test_reference_url_invalid():
    response = client.post(
        "/api/v1/references/url",
        json={"url": "http://127.0.0.1:8080/internal-secret"}
    )
    assert response.status_code == 400


def test_training_start_without_reference():
    payload = {
        "base_model": "SpatialTemporalTTVModel",
        "method": "lora",
        "epochs": 1,
        "learning_rate": "0.0001",
        "batch_size": 1
    }
    response = client.post("/api/v1/training/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"].startswith("train_")
    assert data["status"] in {"running", "queued"}


def test_training_start_with_image_reference():
    content = _create_test_image_bytes()
    upload_res = client.post(
        "/api/v1/references/upload",
        files={"file": ("train_ref.jpg", content, "image/jpeg")},
        data={"reference_type": "image"}
    )
    assert upload_res.status_code == 200
    ref_id = upload_res.json()["reference_id"]

    payload = {
        "base_model": "SpatialTemporalTTVModel",
        "method": "lora",
        "epochs": 1,
        "learning_rate": "0.0001",
        "batch_size": 1,
        "reference_id": ref_id,
        "reference_type": "image"
    }
    response = client.post("/api/v1/training/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"].startswith("train_")
    assert data["reference_id"] == ref_id
    assert any("Attached reference" in log for log in data.get("logs", []))


def test_training_prepare_reference_dataset_manifest(tmp_path):
    content = _create_test_image_bytes()
    saved = reference_service.save_upload("ref_test.jpg", content, "image/jpeg")
    
    out_manifest = str(tmp_path / "reference_train_manifest.jsonl")
    entry = prepare_reference_dataset_entry(saved, output_manifest_path=out_manifest)
    
    assert entry["id"].startswith("sample_ref_")
    assert os.path.exists(out_manifest)
    
    # Test DataLoader compatibility with the reference manifest
    loader = create_dataloader(
        manifest_path=out_manifest,
        batch_size=1,
        num_frames=8,
        height=128,
        width=128,
        shuffle=False
    )
    assert len(loader) >= 1
    batch = next(iter(loader))
    assert batch["frames"].shape == (1, 8, 3, 128, 128)
    assert batch["prompt_tokens"].shape[0] == 1
