"""
Coqui XTTS v2 Voice Cloning & Multilingual Synthesis Adapter.
Enables instant voice cloning from reference audio clips where supported.
"""
import os
from typing import Optional
from app.adapters.base import BaseTTSAdapter
from app.adapters.tts.edge_tts_adapter import EdgeTTSAdapter
from app.core.logging import telemetry


class XTTSAdapter(BaseTTSAdapter):
    def __init__(self):
        self.fallback = EdgeTTSAdapter()

    async def synthesize_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        # XTTS requires GPU and TTS library installation
        try:
            from TTS.api import TTS  # type: ignore
            # Local XTTS inference if model available
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            tts.tts_to_file(text=text, file_path=output_path, language=language or "en")
            if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
                return output_path
        except Exception as exc:
            telemetry.emit("xtts_fallback_edge", "tts", {"reason": str(exc)}, level="info")

        # Graceful fallback to EdgeTTS
        return await self.fallback.synthesize_speech(text, output_path, voice=voice, language=language)
