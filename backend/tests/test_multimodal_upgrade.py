import asyncio

from app.schemas.request import GenerationRequest
from app.services.realtime_data_service import realtime_data_service
from app.services.character_service import character_service


def test_generation_request_carries_multimodal_controls():
    request = GenerationRequest(
        prompt="A farmer explains organic farming.",
        language="mr-IN",
        character_id="rahul_farmer",
        subtitles=True,
        realtime_data=True,
    )
    assert request.language == "mr"
    assert request.character_id == "rahul_farmer"
    assert request.subtitles is True
    assert request.realtime_data is True


def test_builtin_character_has_identity_and_consent_metadata():
    profile = character_service.get_profile("rahul_farmer")
    assert profile is not None
    assert "Consistent facial structure" in profile.get_visual_anchor()
    assert profile.voice_profile and profile.voice_profile.consent_status == "confirmed"


def test_realtime_data_fails_closed_when_not_configured(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "ENABLE_REALTIME_DATA", False)
    assert asyncio.run(realtime_data_service.fetch_realtime_context("today's tomato price")) is None
