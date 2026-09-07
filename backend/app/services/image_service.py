import os
from typing import List
from app.models.scene import Scene
from app.adapters.image import get_image_adapter
from app.core.config import settings

class ImageService:
    """Orchestrates scene keyframe image generation."""
    def __init__(self):
        self.image_adapter = get_image_adapter(settings.IMAGE_PROVIDER)

    async def generate_scene_keyframes(self, scenes: List[Scene], execution_id: str) -> List[Scene]:
        images_dir = settings.get_absolute_path(settings.IMAGES_DIR)
        os.makedirs(images_dir, exist_ok=True)

        for scene in scenes:
            filename = f"{execution_id}_scene_{scene.index}_keyframe.jpg"
            keyframe_path = str(images_dir / filename)
            
            prompt_to_use = scene.metadata.get("enriched_prompt", scene.visual_description)
            style_to_use = scene.metadata.get("style_preset", "cinematic")
            
            await self.image_adapter.generate_image(
                prompt=prompt_to_use,
                output_path=keyframe_path,
                style=style_to_use
            )
            scene.image_path = keyframe_path

        return scenes

image_service = ImageService()
