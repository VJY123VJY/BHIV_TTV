from typing import List, Dict, Any, Optional
from app.models.scene import Scene
from app.adapters.vision import get_vision_adapter
from app.core.config import settings

class VisionService:
    """
    Manages visual consistency across scenes.
    Integrates prompt consistency, character anchors, and color palette alignment.
    """
    def __init__(self):
        self.vision_adapter = get_vision_adapter(settings.VISION_PROVIDER)

    def apply_visual_consistency(
        self,
        scenes: List[Scene],
        global_style: str,
        analysis: Optional[Dict[str, Any]] = None,
        aspect_ratio: str = "16:9",
        reference: Optional[Dict[str, Any]] = None,
    ) -> List[Scene]:
        char_refs = None
        if analysis:
            char_refs = {
                "subject_description": f"{analysis.get('subject', 'subject')}, {analysis.get('mood', 'atmospheric')}",
                "lighting": analysis.get("lighting", "natural lighting"),
                "setting": analysis.get("setting", "environment")
            }
        if reference:
            char_refs = char_refs or {}
            char_refs["subject_description"] = (
                f"{char_refs.get('subject_description', 'subject')}, "
                "match identity, wardrobe, and palette from the user reference still"
            )
        return self.vision_adapter.enrich_scene_prompts(scenes, global_style, char_refs, aspect_ratio=aspect_ratio)

vision_service = VisionService()
