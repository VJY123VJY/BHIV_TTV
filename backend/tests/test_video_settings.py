import pytest
from app.core.exceptions import ValidationError
from app.utils.video_settings import get_resolution, resolve_video_settings, format_resolution


@pytest.mark.parametrize(
    "aspect,quality,expected",
    [
        ("16:9", "standard", (1280, 720)),
        ("16:9", "high", (1920, 1080)),
        ("16:9", "ultra", (3840, 2160)),
        ("9:16", "standard", (720, 1280)),
        ("9:16", "high", (1080, 1920)),
        ("9:16", "ultra", (2160, 3840)),
    ],
)
def test_resolution_matrix(aspect, quality, expected):
    assert get_resolution(aspect, quality) == expected
    resolved = resolve_video_settings(aspect, quality)
    assert (resolved["width"], resolved["height"]) == expected
    assert resolved["resolution"] == format_resolution(*expected)
    assert resolved["aspect_ratio"] == aspect
    assert resolved["quality"] == quality


def test_quality_wins_over_legacy_resolution():
    resolved = resolve_video_settings("9:16", "high", "1280x720")
    assert resolved["resolution"] == "1080x1920"


def test_legacy_resolution_when_new_fields_omitted():
    resolved = resolve_video_settings(None, None, "1920x1080")
    assert resolved["aspect_ratio"] == "16:9"
    assert resolved["quality"] == "high"


def test_defaults_when_nothing_provided():
    resolved = resolve_video_settings()
    assert resolved["aspect_ratio"] == "16:9"
    assert resolved["quality"] == "standard"
    assert resolved["resolution"] == "1280x720"


def test_invalid_aspect_and_quality():
    with pytest.raises(ValidationError):
        get_resolution("1:1", "standard")
    with pytest.raises(ValidationError):
        get_resolution("16:9", "8k")
