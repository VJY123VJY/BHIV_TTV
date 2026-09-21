import os
from typing import List, Optional
from app.models.scene import Scene
from app.adapters.tts import get_tts_adapter
from app.core.config import settings
from app.utils.languages import resolve_voice, normalize_language

class TTSService:
    """Orchestrates speech narration generation for scenes."""
    def __init__(self):
        self.tts_adapter = get_tts_adapter(settings.TTS_PROVIDER)

    async def generate_scene_narration(
        self,
        scenes: List[Scene],
        execution_id: str,
        language: str = "en",
        voice: Optional[str] = None,
    ) -> List[Scene]:
        audio_dir = settings.get_absolute_path(settings.AUDIO_DIR)
        os.makedirs(audio_dir, exist_ok=True)
        lang = normalize_language(language)
        selected_voice = resolve_voice(lang, voice)

        for scene in scenes:
            if not scene.narrative:
                continue

            filename = f"{execution_id}_scene_{scene.index}_voice.mp3"
            voice_path = str(audio_dir / filename)

            actual_path = await self.tts_adapter.synthesize_speech(
                text=scene.narrative,
                output_path=voice_path,
                voice=selected_voice,
                language=lang,
            )
            scene.audio_path = actual_path
            scene.metadata["tts_language"] = lang
            scene.metadata["tts_voice"] = selected_voice

        return scenes

tts_service = TTSService()
