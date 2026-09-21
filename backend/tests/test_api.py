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
    assert data["settings"]["aspect_ratio"] == "16:9"
    assert data["settings"]["quality"] == "standard"
    assert data["settings"]["language"] == "en"

    job_id = data["job_id"]
    job_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["job_id"] == job_id
    assert job_data["status"] in ["queued", "processing", "completed"]
    assert job_data["settings"]["resolution"] == "1280x720"


@pytest.mark.parametrize("field,value", [
    ("aspect_ratio", "1:1"),
    ("quality", "8k"),
    ("language", "klingon"),
    ("reference_url", "http://127.0.0.1/x.jpg"),
])
def test_generate_validation_errors(field, value):
    payload = {"prompt": "A farmer walking through a green vegetable farm", field: value}
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 422


def test_complete_generation_request_is_accepted():
    payload = {
        "prompt": "A farmer walking through a green vegetable farm",
        "duration": 15,
        "aspect_ratio": "9:16",
        "quality": "high",
        "style": "realistic",
        "language": "mr",
        "voice": True,
        "reference_url": None,
        "reference_type": None,
        "fps": 24
    }
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200
    settings = response.json()["settings"]
    assert settings["aspect_ratio"] == "9:16"
    assert settings["quality"] == "high"
    assert settings["resolution"] == "1080x1920"
    assert settings["language"] == "mr"
    assert settings["style"] == "realistic"


def test_generation_options_catalog():
    response = client.get("/api/v1/generation-options")
    assert response.status_code == 200
    data = response.json()
    assert data["resolution_matrix"]["9:16"]["ultra"] == "2160x3840"
    assert any(lang["id"] == "mr" for lang in data["languages"])

def test_video_not_found():
    response = client.get("/api/v1/videos/non_existent_video_id")
    assert response.status_code == 404
