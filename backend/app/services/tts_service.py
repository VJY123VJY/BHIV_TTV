import os
from typing import List
from app.models.scene import Scene
from app.adapters.tts import get_tts_adapter
from app.core.config import settings

class TTSService:
    """Orchestrates speech narration generation for scenes."""
    def __init__(self):
        self.tts_adapter = get_tts_adapter(settings.TTS_PROVIDER)

    async def generate_scene_narration(self, scenes: List[Scene], execution_id: str) -> List[Scene]:
        audio_dir = settings.get_absolute_path(settings.AUDIO_DIR)
        os.makedirs(audio_dir, exist_ok=True)

        for scene in scenes:
            if not scene.narrative:
                continue

            filename = f"{execution_id}_scene_{scene.index}_voice.mp3"
            voice_path = str(audio_dir / filename)

            actual_path = await self.tts_adapter.synthesize_speech(
                text=scene.narrative,
                output_path=voice_path,
                voice=settings.DEFAULT_VOICE
            )
            scene.audio_path = actual_path

        return scenes

tts_service = TTSService()
