"""
Comprehensive test suite verifying all 15 required features:
- Aspect Ratio (16:9 Landscape - YouTube, 9:16 Vertical - Reels / Shorts)
- Reference Material (Upload and Direct Link with SSRF security)
- Visual Style Selection (including Documentary and Custom)
- Multi-Language TTS & Dialogue Processing
- Automatic Lip Sync Integration
- Extended API parameters & Job Metadata
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.request import GenerationRequest, SUPPORTED_STYLES
from app.utils.video_settings import resolve_video_settings, ASPECT_RATIOS
from app.utils.languages import normalize_language, SUPPORTED_LANGUAGES
from app.adapters.reference.validator import validate_public_url, ReferenceValidationError
from app.adapters.tts import get_tts_adapter
from app.adapters.lipsync import get_lipsync_adapter
from app.pipelines.text_to_video import pipeline
from app.core.queue import job_manager

client = TestClient(app)


# =========================================================================
# FEATURE 1: VIDEO FORMAT / ASPECT RATIO TESTS
# =========================================================================

def test_aspect_ratio_16_9_accepted():
    payload = {
        "prompt": "Cinematic mountain landscape at sunrise",
        "aspect_ratio": "16:9",
        "quality": "standard"
    }
    req = GenerationRequest(**payload)
    assert req.aspect_ratio == "16:9"
    cfg = resolve_video_settings(req.aspect_ratio, req.quality)
    assert cfg["aspect_ratio"] == "16:9"
    assert cfg["width"] == 1280
    assert cfg["height"] == 720
    assert "Landscape" in cfg["aspect_label"]


def test_aspect_ratio_9_16_accepted():
    payload = {
        "prompt": "Cinematic vertical portrait shot of a runner",
        "aspect_ratio": "9:16",
        "quality": "high"
    }
    req = GenerationRequest(**payload)
    assert req.aspect_ratio == "9:16"
    cfg = resolve_video_settings(req.aspect_ratio, req.quality)
    assert cfg["aspect_ratio"] == "9:16"
    assert cfg["width"] == 1080
    assert cfg["height"] == 1920
    assert "Vertical" in cfg["aspect_label"]


def test_invalid_aspect_ratio_rejected():
    with pytest.raises(Exception):
        GenerationRequest(prompt="A test video prompt", aspect_ratio="4:3")
    
    with pytest.raises(Exception):
        GenerationRequest(prompt="A test video prompt", aspect_ratio="1:1")

    res = client.post("/api/v1/generate", json={"prompt": "Valid test prompt", "aspect_ratio": "4:3"})
    assert res.status_code == 422


# =========================================================================
# FEATURE 2 & 13: REFERENCE MATERIAL & SSRF SECURITY TESTS
# =========================================================================

def test_valid_public_media_url_accepted():
    url, source = validate_public_url("https://example.com/assets/sample.mp4", resolve_dns=False)
    assert url == "https://example.com/assets/sample.mp4"
    assert source == "direct"


def test_valid_social_urls_detected():
    _, yt_src = validate_public_url("https://www.youtube.com/watch?v=sample123", resolve_dns=False)
    assert yt_src == "youtube"
    _, ig_src = validate_public_url("https://instagram.com/reel/sample123", resolve_dns=False)
    assert ig_src == "instagram"


@pytest.mark.parametrize("bad_url", [
    "http://127.0.0.1/private.jpg",
    "http://localhost/secret.mp4",
    "http://10.0.0.1/data.png",
    "http://192.168.1.1/video.mp4",
    "http://169.254.169.254/latest/meta-data/",
    "ftp://example.com/file.mp4",
    "file:///etc/passwd",
])
def test_private_and_ssrf_urls_rejected(bad_url):
    with pytest.raises(ReferenceValidationError):
        validate_public_url(bad_url, resolve_dns=False)

    res = client.post("/api/v1/generate", json={
        "prompt": "Test video generation prompt",
        "reference_url": bad_url
    })
    assert res.status_code == 422


# =========================================================================
# FEATURE 3: VISUAL STYLE SELECTION TESTS
# =========================================================================

@pytest.mark.parametrize("style_name,expected", [
    ("realistic", "realistic"),
    ("Realistic / Photorealistic", "realistic"),
    ("photorealistic", "realistic"),
    ("cartoon", "cartoon"),
    ("Cartoon / Cartoonish", "cartoon"),
    ("cinematic", "cinematic"),
    ("Cinematic / Film Look", "cinematic"),
    ("3d", "3d"),
    ("3D / Stylized", "3d"),
    ("anime", "anime"),
    ("documentary", "documentary"),
    ("custom", "custom"),
])
def test_visual_styles_accepted(style_name, expected):
    req = GenerationRequest(prompt="Valid prompt here", style=style_name)
    assert req.style == expected
    assert req.style in SUPPORTED_STYLES

    # Test visual_style parameter alias
    req_alias = GenerationRequest(prompt="Valid prompt here", visual_style=style_name)
    assert req_alias.style == expected
    assert req_alias.visual_style == expected


def test_invalid_visual_style_rejected():
    with pytest.raises(Exception):
        GenerationRequest(prompt="Valid prompt", style="non_existent_style_xyz")

    res = client.post("/api/v1/generate", json={"prompt": "Valid prompt", "visual_style": "invalid_style"})
    assert res.status_code == 422


# =========================================================================
# FEATURE 4: MULTI-LANGUAGE AUDIO / TTS TESTS
# =========================================================================

@pytest.mark.parametrize("lang_input,expected_code", [
    ("en", "en"),
    ("english", "en"),
    ("en-US", "en"),
    ("hi", "hi"),
    ("hindi", "hi"),
    ("hi-IN", "hi"),
    ("mr", "mr"),
    ("marathi", "mr"),
    ("mr-IN", "mr"),
    ("gu", "gu"),
    ("gujarati", "gu"),
    ("gu-IN", "gu"),
    ("fr", "fr"),
    ("french", "fr"),
    ("fr-FR", "fr"),
    ("es", "es"),
    ("spanish", "es"),
    ("es-ES", "es"),
    ("de", "de"),
    ("german", "de"),
])
def test_multi_language_normalization(lang_input, expected_code):
    assert normalize_language(lang_input) == expected_code
    req = GenerationRequest(prompt="Valid prompt here", language=lang_input)
    assert req.language == expected_code


def test_unsupported_language_rejected():
    with pytest.raises(Exception):
        normalize_language("klingon")

    res = client.post("/api/v1/generate", json={"prompt": "Valid prompt", "language": "unsupported_xyz"})
    assert res.status_code == 422


def test_tts_provider_interface():
    tts = get_tts_adapter("local")
    assert hasattr(tts, "synthesize_speech")
    assert hasattr(tts, "generate_audio")
    assert hasattr(tts, "get_supported_languages")
    assert hasattr(tts, "validate_language")
    languages = tts.get_supported_languages()
    assert "en" in languages
    assert "hi" in languages
    assert "mr" in languages
    assert tts.validate_language("mr-IN") == "mr"


# =========================================================================
# FEATURE 5 & 6: SCRIPT / DIALOGUE PROCESSING & LIP SYNC TESTS
# =========================================================================

def test_lipsync_provider_interface():
    lipsync = get_lipsync_adapter("viseme")
    assert hasattr(lipsync, "sync_clip")
    assert hasattr(lipsync, "generate_lipsync")
    assert hasattr(lipsync, "validate_input")
    assert hasattr(lipsync, "get_status")
    status = lipsync.get_status()
    assert status["available"] is True


def test_generation_request_supports_dialogue_and_visual_style():
    payload = {
        "prompt": "A young farmer standing in a green field during sunrise.",
        "dialogue": "आज आपण आपल्या शेतातील नवीन पिकाबद्दल माहिती घेणार आहोत.",
        "aspect_ratio": "16:9",
        "visual_style": "cinematic",
        "language": "mr-IN"
    }
    req = GenerationRequest(**payload)
    assert req.dialogue == "आज आपण आपल्या शेतातील नवीन पिकाबद्दल माहिती घेणार आहोत."
    assert req.visual_style == "cinematic"
    assert req.style == "cinematic"
    assert req.language == "mr"
    assert req.aspect_ratio == "16:9"


# =========================================================================
# FEATURE 8 & 11: EXTENDED API & JOB METADATA TESTS
# =========================================================================

def test_generate_api_with_all_new_parameters():
    payload = {
        "prompt": "A young farmer standing in a green field during sunrise.",
        "dialogue": "Welcome to our green farm at early dawn.",
        "aspect_ratio": "16:9",
        "visual_style": "documentary",
        "language": "en",
        "fps": 24,
        "duration": 10
    }
    res = client.post("/api/v1/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "queued"
    job_id = data["job_id"]
    settings = data["settings"]
    assert settings["aspect_ratio"] == "16:9"
    assert settings["style"] == "documentary"
    assert settings["visual_style"] == "documentary"
    assert settings["dialogue"] == "Welcome to our green farm at early dawn."

    # Test job status contains tracked metadata
    job_res = client.get(f"/api/v1/jobs/{job_id}")
    assert job_res.status_code == 200
    job_info = job_res.json()
    assert job_info["aspect_ratio"] == "16:9"
    assert job_info["visual_style"] == "documentary"
    assert job_info["language"] == "en"
    assert job_info["tts_status"] in ["pending", "processing", "completed"]
    assert job_info["video_status"] in ["pending", "processing", "completed"]
    assert job_info["lip_sync_status"] in ["pending", "processing", "completed"]


# =========================================================================
# FEATURE 9 & 14: END-TO-END PIPELINE WORKFLOW TESTS
# =========================================================================

@pytest.mark.asyncio
async def test_pipeline_16_9_cinematic_english_with_dialogue():
    """Test 1: Prompt + 16:9 + Cinematic + English + Dialogue"""
    result = await pipeline.execute(
        prompt="A scientist in a laboratory observing glowing crystals.",
        dialogue="The experiment is showing promising results today.",
        duration=6,
        aspect_ratio="16:9",
        quality="standard",
        style="cinematic",
        language="en",
        lipsync=True,
        voice=True,
    )
    assert result["status"] == "success"
    meta = result["metadata"]
    assert meta["aspect_ratio"] == "16:9"
    assert meta["resolution"] == "1280x720"
    assert meta["validation"]["width"] == 1280
    assert meta["validation"]["height"] == 720
    assert meta["dialogue"] == "The experiment is showing promising results today."
    assert len(meta["scenes"]) >= 1


@pytest.mark.asyncio
async def test_pipeline_9_16_realistic_marathi_with_dialogue():
    """Test 2: Prompt + 9:16 + Realistic + Marathi + Dialogue"""
    result = await pipeline.execute(
        prompt="A young farmer standing in a green field during sunrise.",
        dialogue="आज आपण आपल्या शेतातील नवीन पिकाबद्दल माहिती घेणार आहोत.",
        duration=6,
        aspect_ratio="9:16",
        quality="high",
        style="realistic",
        language="mr",
        lipsync=True,
        voice=True,
    )
    assert result["status"] == "success"
    meta = result["metadata"]
    assert meta["aspect_ratio"] == "9:16"
    assert meta["resolution"] == "1080x1920"
    assert meta["validation"]["width"] == 1080
    assert meta["validation"]["height"] == 1920
    assert meta["language"] == "mr"
    assert meta["dialogue"] == "आज आपण आपल्या शेतातील नवीन पिकाबद्दल माहिती घेणार आहोत."
