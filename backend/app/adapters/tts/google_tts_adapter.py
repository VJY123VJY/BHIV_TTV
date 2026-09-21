"""
Google Translate TTS (gTTS) Adapter.
Reliable multilingual speech generation across 100+ languages.
"""
import os
from typing import Optional
from app.adapters.base import BaseTTSAdapter
from app.utils.languages import get_language_config, normalize_language
from app.core.exceptions import AudioGenerationError
from app.core.logging import telemetry


class GoogleTTSAdapter(BaseTTSAdapter):
    async def synthesize_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lang_cfg = get_language_config(language or "en")

        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=lang_cfg["gtts"])
            tts.save(output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
                telemetry.emit("gtts_synthesized", "tts", {"language": lang_cfg["code"]})
                return output_path
        except Exception as e:
            telemetry.emit("gtts_failed", "tts", {"error": str(e), "language": lang_cfg["code"]}, level="warning")
            raise AudioGenerationError(f"Google TTS synthesis failed for {lang_cfg['name']}: {e}") from e

        raise AudioGenerationError(f"Google TTS produced an invalid audio file for {lang_cfg['name']}.")
