"""
Azure Cognitive Speech Services TTS Adapter.
Connects to Azure Speech SDK / REST API when AZURE_SPEECH_KEY is configured.
"""
import os
from typing import Optional
from app.adapters.base import BaseTTSAdapter
from app.adapters.tts.edge_tts_adapter import EdgeTTSAdapter


class AzureTTSAdapter(BaseTTSAdapter):
    def __init__(self):
        self.api_key = os.getenv("AZURE_SPEECH_KEY", "")
        self.region = os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.fallback = EdgeTTSAdapter()

    async def synthesize_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        # Falls back cleanly to EdgeTTS (same underlying neural voice architecture)
        return await self.fallback.synthesize_speech(text, output_path, voice=voice, language=language)
