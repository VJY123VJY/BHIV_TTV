from typing import List, Dict, Any, Optional
from app.adapters.base import BaseVisionConsistencyAdapter
from app.models.scene import Scene

class VisualConsistencyAdapter(BaseVisionConsistencyAdapter):
    """
    Enforces cross-scene visual consistency:
    - Character and subject identity retention
    - Color palette & lighting continuity derived from prompt
    - Environmental coherency
    - Consistent cinematic styling
    """
    STYLE_PRESETS = {
        "cinematic": "cinematic film still, 35mm photography, dramatic atmospheric lighting, natural textures, photorealistic depth of field, anamorphic bokeh",
        "anime": "high-end modern anime aesthetic, Makoto Shinkai style, vibrant luminous sky, detailed background art, cel shaded character",
        "realistic": "documentary style, natural daylight, uncompressed 8k, sharp focus, true-to-life textures, National Geographic quality",
        "cyberpunk": "cyberpunk metropolis, neon lighting reflections, dark wet asphalt, holographic billboards, high tech urban atmosphere",
        "fantasy": "epic fantasy world, magical ethereal glow, painted concept art, detailed matte painting, mystical wonder"
    }

    def enrich_scene_prompts(self, scenes: List[Scene], global_style: str, character_refs: Optional[Dict[str, Any]] = None) -> List[Scene]:
        style_key = global_style.lower()
        style_prompt = self.STYLE_PRESETS.get(style_key, self.STYLE_PRESETS["cinematic"])

        char_anchor = "consistent subject appearance and proportions"
        lighting_tag = "natural ambient lighting"
        setting_tag = "cohesive landscape"

        if character_refs:
            if "subject_description" in character_refs:
                char_anchor = character_refs["subject_description"]
            if "lighting" in character_refs:
                lighting_tag = character_refs["lighting"]
            if "setting" in character_refs:
                setting_tag = character_refs["setting"]

        for i, scene in enumerate(scenes):
            continuity_tag = f"Scene {scene.index} continuity: matches environment of {setting_tag}; {char_anchor}"
            
            enriched = (
                f"{scene.visual_description} "
                f"Style: {style_prompt}. "
                f"Lighting: {lighting_tag}. "
                f"Continuity: {continuity_tag}."
            )
            scene.metadata["enriched_prompt"] = enriched
            scene.metadata["style_preset"] = style_key
            scene.metadata["detected_setting"] = setting_tag

        return scenes
