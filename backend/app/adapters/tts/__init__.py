"""
Text-to-Speech (TTS) unified provider router.
Supports edge, google (gtts), local, xtts, elevenlabs, azure, and auto routing.
"""
from typing import Optional
from app.adapters.base import BaseTTSAdapter
from app.adapters.tts.local_tts_adapter import LocalTTSAdapter
from app.adapters.tts.edge_tts_adapter import EdgeTTSAdapter
from app.adapters.tts.google_tts_adapter import GoogleTTSAdapter
from app.adapters.tts.xtts_adapter import XTTSAdapter
from app.adapters.tts.elevenlabs_adapter import ElevenLabsAdapter
from app.adapters.tts.azure_tts_adapter import AzureTTSAdapter
from app.core.config import settings


def get_tts_adapter(provider: Optional[str] = None) -> BaseTTSAdapter:
    prov = (provider or settings.TTS_PROVIDER or "auto").lower().strip()

    if prov in ("edge", "edgetts"):
        return EdgeTTSAdapter()
    if prov in ("google", "gtts"):
        return GoogleTTSAdapter()
    if prov == "xtts":
        return XTTSAdapter()
    if prov == "elevenlabs":
        return ElevenLabsAdapter()
    if prov in ("azure", "azuretts"):
        return AzureTTSAdapter()
    if prov in ("auto", "local"):
        return LocalTTSAdapter()

    return LocalTTSAdapter()
