import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "providers" in data

def test_generate_validation_fail():
    response = client.post("/api/v1/generate", json={"prompt": ""})
    assert response.status_code == 422 # Pydantic schema validation error for min_length

def test_generate_queued_success():
    payload = {
        "prompt": "A small rover roaming the dunes of Mars.",
        "duration": 10,
        "style": "cinematic",
        "voice": True
    }
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert "job_id" in data

    job_id = data["job_id"]
    # Check job status endpoint
    job_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["job_id"] == job_id
    assert job_data["status"] in ["queued", "processing", "completed"]

def test_video_not_found():
    response = client.get("/api/v1/videos/non_existent_video_id")
    assert response.status_code == 404
