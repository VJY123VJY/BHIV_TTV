"""
Microsoft Edge Neural TTS Adapter.
High-fidelity neural voice synthesis with extensive multilingual support.
"""
import os
from typing import Optional
from app.adapters.base import BaseTTSAdapter
from app.utils.languages import resolve_voice, normalize_language
from app.core.exceptions import AudioGenerationError
from app.core.logging import telemetry


class EdgeTTSAdapter(BaseTTSAdapter):
    async def synthesize_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lang_code = normalize_language(language or "en")
        selected_voice = resolve_voice(lang_code, voice)

        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, selected_voice)
            await communicate.save(output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
                telemetry.emit("edgetts_synthesized", "tts", {"language": lang_code, "voice": selected_voice})
                return output_path
        except Exception as e:
            telemetry.emit("edgetts_failed", "tts", {"error": str(e), "language": lang_code, "voice": selected_voice}, level="warning")
            raise AudioGenerationError(f"EdgeTTS synthesis failed for {lang_code}: {e}") from e

        raise AudioGenerationError(f"EdgeTTS produced an empty or invalid audio file for {lang_code}.")
