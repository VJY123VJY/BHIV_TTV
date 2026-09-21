import pytest
from app.core.exceptions import ValidationError
from app.utils.languages import normalize_language, resolve_voice, localize_narrative


def test_supported_language_codes():
    assert normalize_language("English") == "en"
    assert normalize_language("hi") == "hi"
    assert normalize_language("Marathi") == "mr"
    assert normalize_language("gu") == "gu"
    assert normalize_language("fr") == "fr"


def test_unsupported_language():
    with pytest.raises(ValidationError):
        normalize_language("swahili")


def test_voice_mismatch_is_rejected():
    with pytest.raises(ValidationError):
        resolve_voice("mr", "en-US-GuyNeural")


def test_marathi_narrative_is_not_english():
    text = localize_narrative("mr", 1, "farmer", "farm", "walking", fallback="A farmer walks")
    assert "मध्ये" in text or "मराठी" in text or "कथा" in text
    assert text != "A farmer walks"
