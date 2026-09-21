import asyncio
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np
import cv2

from app.main import app
from app.adapters.reference.validator import validate_public_url, ReferenceValidationError
from app.services.reference_service import reference_service

client = TestClient(app)


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 80, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_reject_private_and_invalid_urls():
    for url in ("file:///etc/passwd", "http://127.0.0.1/secret", "ftp://example.com/a.jpg"):
        try:
            validate_public_url(url, resolve_dns=False)
            raise AssertionError(f"should reject {url}")
        except ReferenceValidationError:
            pass


def test_detect_supported_public_hosts():
    _, source = validate_public_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ", resolve_dns=False)
    assert source == "youtube"
    _, source = validate_public_url("https://instagram.com/p/public123/", resolve_dns=False)
    assert source == "instagram"
    _, source = validate_public_url("https://example.com/photo.jpg", resolve_dns=False)
    assert source == "direct"


def test_reference_image_upload():
    response = client.post(
        "/api/v1/references/upload",
        files={"file": ("farm.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["media_type"] == "image"
    assert data["reference_id"].startswith("ref_")


def test_reference_video_upload(tmp_path):
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(8):
        writer.write(frame)
    writer.release()
    response = client.post(
        "/api/v1/references/upload",
        files={"file": ("clip.mp4", video_path.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 200
    assert response.json()["media_type"] == "video"


def test_reference_upload_rejects_bad_type():
    response = client.post(
        "/api/v1/references/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_generate_rejects_invalid_reference_url():
    response = client.post("/api/v1/generate", json={
        "prompt": "A farmer walking through a green vegetable farm",
        "reference_url": "http://localhost/private.jpg",
    })
    assert response.status_code == 422


@patch.object(reference_service, "ingest_url", new_callable=AsyncMock)
def test_valid_public_url_is_ingested(mock_ingest, tmp_path):
    dest = tmp_path / "still.jpg"
    Image.new("RGB", (64, 64), (10, 120, 30)).save(dest)
    mock_ingest.return_value = {
        "reference_id": "ref_abc123abc123",
        "path": str(dest),
        "media_type": "image",
        "source": "direct",
        "width": 64,
        "height": 64,
    }
    payload = asyncio.run(reference_service.resolve(reference_url="https://example.com/photo.jpg"))
    assert payload["still_path"]
    assert mock_ingest.called
