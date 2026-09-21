import os
from typing import List, Optional, Dict, Any
from PIL import Image
from app.models.scene import Scene
from app.adapters.image import get_image_adapter
from app.core.config import settings

class ImageService:
    """Orchestrates scene keyframe image generation."""
    def __init__(self):
        self.image_adapter = get_image_adapter(settings.IMAGE_PROVIDER)

    async def generate_scene_keyframes(
        self,
        scenes: List[Scene],
        execution_id: str,
        width: int = 1280,
        height: int = 720,
        reference: Optional[Dict[str, Any]] = None,
    ) -> List[Scene]:
        images_dir = settings.get_absolute_path(settings.IMAGES_DIR)
        os.makedirs(images_dir, exist_ok=True)
        still_path = reference.get("still_path") if reference else None

        for scene in scenes:
            filename = f"{execution_id}_scene_{scene.index}_keyframe.jpg"
            keyframe_path = str(images_dir / filename)
            
            prompt_to_use = scene.metadata.get("enriched_prompt", scene.visual_description)
            style_to_use = scene.metadata.get("style_preset", "cinematic")
            if still_path:
                prompt_to_use = f"{prompt_to_use} Match composition, subject identity, wardrobe, and color palette of the provided reference still."
            
            await self.image_adapter.generate_image(
                prompt=prompt_to_use,
                output_path=keyframe_path,
                style=style_to_use,
                width=width,
                height=height,
            )
            if still_path and os.path.exists(still_path):
                self._blend_reference(keyframe_path, still_path, strength=0.28 if scene.index == 1 else 0.14)
            scene.image_path = keyframe_path
            scene.metadata["frame_size"] = f"{width}x{height}"

        return scenes

    def _blend_reference(self, keyframe_path: str, still_path: str, strength: float) -> None:
        try:
            with Image.open(keyframe_path).convert("RGB") as base, Image.open(still_path).convert("RGB") as ref:
                ref = ref.resize(base.size, Image.Resampling.LANCZOS)
                blended = Image.blend(base, ref, max(0.0, min(0.5, strength)))
                blended.save(keyframe_path, quality=92)
        except Exception:
            return

image_service = ImageService()
