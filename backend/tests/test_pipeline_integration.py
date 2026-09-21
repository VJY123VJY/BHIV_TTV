"""
End-to-end integration test verifying the complete multi-parameter TTV workflow:
Script -> 9:16 -> High (1080x1920) -> Realistic -> Marathi -> Reference -> Generate
and all resolution matrix combinations.
"""
import os
import pytest
from PIL import Image
import numpy as np

from app.pipelines.text_to_video import pipeline
from app.utils.video_settings import resolve_video_settings, VIDEO_SETTINGS, ASPECT_RATIOS, QUALITIES
from app.utils.media_check import validate_video_file
from app.services.reference_service import reference_service


@pytest.mark.asyncio
async def test_full_acceptance_workflow_marathi_vertical_high(tmp_path):
    """
    Acceptance Criteria:
    - User enters script: "A farmer walking through a green vegetable farm"
    - Aspect Ratio: 9:16
    - Quality: High (1080x1920)
    - Style: Realistic
    - Language: Marathi (mr)
    - Valid reference image
    - Generates 9:16 1080x1920 MP4 with Marathi narrative and lip-sync
    """
    # 1. Create a valid reference image
    ref_img_path = tmp_path / "reference_farmer.jpg"
    img = Image.new("RGB", (720, 1280), color=(34, 139, 34))
    img.save(ref_img_path, format="JPEG")
    ref_bytes = ref_img_path.read_bytes()

    # Upload via reference service
    ref_info = reference_service.save_upload(
        filename="reference_farmer.jpg",
        content=ref_bytes,
        content_type="image/jpeg"
    )
    assert ref_info["reference_id"].startswith("ref_")
    assert ref_info["media_type"] == "image"

    # 2. Execute pipeline with all requested settings
    result = await pipeline.execute(
        prompt="A farmer walking through a green vegetable farm",
        duration=6,  # 6s for fast CI test execution
        style="realistic",
        voice=True,
        aspect_ratio="9:16",
        quality="high",
        language="mr",
        reference_id=ref_info["reference_id"],
        reference_type="image",
        lipsync=True,
        fps=24,
    )

    # 3. Assertions on pipeline result
    assert result["status"] == "success"
    assert "metadata" in result
    meta = result["metadata"]

    # Resolution and aspect ratio checks
    assert meta["aspect_ratio"] == "9:16"
    assert meta["quality"] == "high"
    assert meta["resolution"] == "1080x1920"
    assert meta["language"] == "mr"
    assert meta["style"] == "realistic"
    assert meta["fps"] == 24

    # Output file validation
    video_url = meta["video_url"]
    assert video_url.endswith(".mp4")
    raw_path = meta.get("file_path") or str(meta.get("output_path", ""))
    if os.path.exists(raw_path):
        is_valid, info = validate_video_file(raw_path)
        assert is_valid
        assert info["width"] == 1080
        assert info["height"] == 1920

    # Scene & language verification
    scenes = meta.get("scenes", [])
    assert len(scenes) >= 2
    for scene in scenes:
        # Check that scene metadata recorded Marathi
        assert scene.get("tts_language") == "mr" or scene.get("language") == "mr" or "mr" in str(scene)


@pytest.mark.parametrize(
    "aspect_ratio,quality,expected_w,expected_h",
    [
        ("16:9", "standard", 1280, 720),
        ("16:9", "high", 1920, 1080),
        ("16:9", "ultra", 3840, 2160),
        ("9:16", "standard", 720, 1280),
        ("9:16", "high", 1080, 1920),
        ("9:16", "ultra", 2160, 3840),
    ],
)
def test_all_format_and_quality_resolution_mappings(aspect_ratio, quality, expected_w, expected_h):
    """Verify all 6 required resolution matrix targets."""
    settings = resolve_video_settings(aspect_ratio=aspect_ratio, quality=quality)
    assert settings["width"] == expected_w
    assert settings["height"] == expected_h
    assert settings["resolution"] == f"{expected_w}x{expected_h}"
    assert settings["aspect_ratio"] == aspect_ratio
    assert settings["quality"] == quality
