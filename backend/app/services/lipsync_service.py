"""Lip-sync orchestration for generated scene clips."""
from typing import List

from app.adapters.lipsync import get_lipsync_adapter
from app.core.config import settings
from app.core.logging import telemetry
from app.models.scene import Scene


class LipSyncService:
    def __init__(self):
        self.adapter = get_lipsync_adapter(settings.LIPSYNC_PROVIDER)

    async def apply_to_scenes(self, scenes: List[Scene], execution_id: str, enabled: bool = True) -> List[Scene]:
        if not enabled or not settings.LIPSYNC_ENABLED:
            for scene in scenes:
                scene.metadata["lipsync"] = {"applied": False, "reason": "disabled"}
            return scenes

        for scene in scenes:
            if not scene.video_path or not scene.audio_path:
                scene.metadata["lipsync"] = {"applied": False, "reason": "missing_audio_or_video"}
                continue
            output_path = scene.video_path.replace(".mp4", "_lipsync.mp4")
            result = await self.adapter.sync_clip(scene.video_path, scene.audio_path, output_path)
            scene.metadata["lipsync"] = result
            if result.get("applied") and result.get("output_path"):
                scene.video_path = result["output_path"]
            else:
                telemetry.emit(
                    "lipsync_scene_skipped",
                    execution_id,
                    {"scene": scene.index, "reason": result.get("reason")},
                    level="info",
                )
        return scenes


lipsync_service = LipSyncService()
