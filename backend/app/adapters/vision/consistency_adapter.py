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
        "realistic": "documentary style, photorealistic, natural daylight, uncompressed 8k, sharp focus, true-to-life textures, National Geographic quality",
        "cyberpunk": "cyberpunk metropolis, neon lighting reflections, dark wet asphalt, holographic billboards, high tech urban atmosphere",
        "fantasy": "epic fantasy world, magical ethereal glow, painted concept art, detailed matte painting, mystical wonder",
        "cartoon": "stylized cartoon illustration, bold outlines, saturated colors, expressive simplified shapes, animated feature look",
        "3d": "stylized 3D render, octane lighting, subsurface scattering, cinematic CGI, clean topology, Pixar-like materials",
    }

    def enrich_scene_prompts(
        self,
        scenes: List[Scene],
        global_style: str,
        character_refs: Optional[Dict[str, Any]] = None,
        aspect_ratio: str = "16:9",
    ) -> List[Scene]:
        style_key = global_style.lower()
        style_prompt = self.STYLE_PRESETS.get(style_key, self.STYLE_PRESETS["cinematic"])
        composition = (
            "9:16 vertical full-height composition, Reels/Shorts framing, subject centered"
            if str(aspect_ratio) == "9:16"
            else "16:9 widescreen landscape composition, cinematic framing"
        )

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
                f"Framing: {composition}. "
                f"Lighting: {lighting_tag}. "
                f"Continuity: {continuity_tag}."
            )
            scene.metadata["enriched_prompt"] = enriched
            scene.metadata["style_preset"] = style_key
            scene.metadata["detected_setting"] = setting_tag
            scene.metadata["aspect_ratio"] = aspect_ratio

        return scenes
