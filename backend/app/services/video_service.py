import os
from typing import List
from app.models.scene import Scene
from app.adapters.video import get_video_adapter
from app.core.config import settings
from app.core.exceptions import VisualGenerationError

class VideoService:
    """Orchestrates scene video rendering from keyframes."""
    @property
    def video_adapter(self):
        return get_video_adapter(settings.VIDEO_PROVIDER)

    async def generate_scene_videos(
        self,
        scenes: List[Scene],
        execution_id: str,
        fps: int = 24,
        width: int = 1280,
        height: int = 720,
    ) -> List[Scene]:
        scenes_dir = settings.get_absolute_path(settings.SCENES_DIR)
        os.makedirs(scenes_dir, exist_ok=True)

        for scene in scenes:
            if not scene.image_path or not os.path.exists(scene.image_path):
                raise VisualGenerationError(f"Missing image keyframe for scene {scene.index}")

            filename = f"{execution_id}_scene_{scene.index}_clip.mp4"
            clip_path = str(scenes_dir / filename)

            await self.video_adapter.generate_scene_video(
                scene=scene,
                keyframe_path=scene.image_path,
                output_path=clip_path,
                width=width,
                height=height,
                fps=fps
            )
            scene.video_path = clip_path
            scene.metadata["output_size"] = f"{width}x{height}"

        return scenes

video_service = VideoService()
