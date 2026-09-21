"""Apply the selected narration language to scene dialogue before TTS."""
from typing import List, Optional, Dict, Any

from app.models.scene import Scene
from app.utils.languages import localize_narrative, normalize_language


class LocalizationService:
    def apply(
        self,
        scenes: List[Scene],
        language: Optional[str],
        analysis: Optional[Dict[str, Any]] = None,
    ) -> List[Scene]:
        code = normalize_language(language)
        subject = (analysis or {}).get("subject") or "character"
        setting = (analysis or {}).get("setting") or "landscape"
        action = (analysis or {}).get("action") or "moving"

        for scene in scenes:
            original = scene.narrative or ""
            scene.metadata["source_narrative"] = original
            scene.narrative = localize_narrative(
                code,
                scene.index,
                str(subject),
                str(setting),
                str(action),
                fallback=original,
            )
            scene.metadata["tts_language"] = code
        return scenes


localization_service = LocalizationService()
