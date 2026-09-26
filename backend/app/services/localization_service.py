"""Apply the selected narration language to scene dialogue before TTS."""
import re
from typing import List, Optional, Dict, Any

from app.models.scene import Scene
from app.utils.languages import localize_narrative, normalize_language


class LocalizationService:
    def apply(
        self,
        scenes: List[Scene],
        language: Optional[str],
        analysis: Optional[Dict[str, Any]] = None,
        dialogue: Optional[str] = None,
    ) -> List[Scene]:
        code = normalize_language(language)
        subject = (analysis or {}).get("subject") or "character"
        setting = (analysis or {}).get("setting") or "landscape"
        action = (analysis or {}).get("action") or "moving"

        # If user explicitly supplied spoken dialogue, prioritize it over generic template narration
        if dialogue and str(dialogue).strip():
            raw_text = str(dialogue).strip()
            # Split into meaningful sentences / dialogue segments
            parts = [p.strip() for p in re.split(r'[\r\n]+|[।!?.]+', raw_text) if p.strip()]
            if not parts:
                parts = [raw_text]

            for idx, scene in enumerate(scenes):
                original = scene.narrative or ""
                scene.metadata["source_narrative"] = original
                assigned_line = parts[idx] if idx < len(parts) else (parts[-1] if len(parts) == 1 else "")
                scene.narrative = assigned_line or original
                scene.metadata["tts_language"] = code
                scene.metadata["user_dialogue"] = True
            return scenes

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
