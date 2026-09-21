"""
ElevenLabs Multilingual Speech Synthesis Adapter.
Connects to ElevenLabs REST API when ELEVENLABS_API_KEY is configured.
"""
import os
from typing import Optional
import httpx
from app.adapters.base import BaseTTSAdapter
from app.adapters.tts.edge_tts_adapter import EdgeTTSAdapter
from app.core.logging import telemetry


class ElevenLabsAdapter(BaseTTSAdapter):
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.fallback = EdgeTTSAdapter()

    async def synthesize_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        if not self.api_key:
            return await self.fallback.synthesize_speech(text, output_path, voice=voice, language=language)

        voice_id = voice or "21m00Tcm4TlvDq8ikWAM"  # Default Rachel
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(res.content)
                    return output_path
        except Exception as exc:
            telemetry.emit("elevenlabs_fallback", "tts", {"reason": str(exc)}, level="warning")

        return await self.fallback.synthesize_speech(text, output_path, voice=voice, language=language)
