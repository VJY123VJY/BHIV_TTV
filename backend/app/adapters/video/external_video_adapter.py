from app.adapters.base import BaseVideoAdapter
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.models.scene import Scene
from app.core.logging import telemetry

class ExternalVideoAdapter(BaseVideoAdapter):
    """External video generation adapter with fallback to OpenCVVideoAdapter."""
    def __init__(self):
        self.fallback = OpenCVVideoAdapter()

    async def generate_scene_video(self, scene: Scene, keyframe_path: str, output_path: str, width: int = 1280, height: int = 720, fps: int = 24) -> str:
        # Fallback to high-performance local OpenCV engine
        telemetry.emit("external_video_route", scene.title, {"engine": "local_opencv_fallback"})
        return await self.fallback.generate_scene_video(scene, keyframe_path, output_path, width, height, fps)
