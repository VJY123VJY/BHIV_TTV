"""Lip-sync adapter factory."""
from typing import Optional
from app.core.config import settings
from app.adapters.lipsync.viseme_adapter import VisemeLipSyncAdapter
from app.adapters.lipsync.wav2lip_adapter import Wav2LipAdapter
from app.adapters.lipsync.musetalk_adapter import MuseTalkAdapter


def get_lipsync_adapter(provider: Optional[str] = None):
    name = (provider or getattr(settings, "LIPSYNC_PROVIDER", "auto") or "auto").lower().strip()
    if name == "musetalk":
        return MuseTalkAdapter()
    if name in {"wav2lip", "neural"}:
        return Wav2LipAdapter()
    if name == "viseme":
        return VisemeLipSyncAdapter()
    # Auto: default to MuseTalk if configured, else Wav2Lip if configured, else Viseme
    if getattr(settings, "MUSETALK_CHECKPOINT_PATH", None):
        return MuseTalkAdapter()
    if getattr(settings, "LIPSYNC_MODEL_PATH", None):
        return Wav2LipAdapter()
    return VisemeLipSyncAdapter()
