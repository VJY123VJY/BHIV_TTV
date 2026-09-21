"""
MuseTalk real-time high-quality lip synchronization adapter.
Supports neural face inpainting with viseme mouth animation fallback.
"""
from __future__ import annotations

import os
from typing import Dict, Any

from app.adapters.lipsync.base import BaseLipSyncAdapter
from app.adapters.lipsync.viseme_adapter import VisemeLipSyncAdapter
from app.core.config import settings
from app.core.logging import telemetry


class MuseTalkAdapter(BaseLipSyncAdapter):
    """
    MuseTalk neural lip-sync engine.
    Uses latent face inpainting when GPU and model weights are provisioned;
    gracefully cascades to the viseme engine in local/CPU environments.
    """

    def __init__(self):
        self.fallback = VisemeLipSyncAdapter()
        self.model_path = getattr(settings, "MUSETALK_CHECKPOINT_PATH", None)

    async def sync_clip(self, video_path: str, audio_path: str, output_path: str) -> Dict[str, Any]:
        if not self.model_path or not os.path.exists(self.model_path):
            telemetry.emit("musetalk_fallback", "lipsync", {"reason": "checkpoint_missing"}, level="info")
            result = await self.fallback.sync_clip(video_path, audio_path, output_path)
            result["engine"] = result.get("engine", "viseme")
            result["fallback_from"] = "musetalk"
            return result

        try:
            # Neural inference entrypoint if musetalk library installed
            from musetalk.utils.inference import inference as musetalk_infer  # type: ignore
            musetalk_infer(audio_path=audio_path, video_path=video_path, output_path=output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return {
                    "applied": True,
                    "reason": "musetalk_neural_sync",
                    "output_path": output_path,
                    "engine": "musetalk",
                }
        except Exception as exc:
            telemetry.emit("musetalk_failed", "lipsync", {"error": str(exc)}, level="warning")

        result = await self.fallback.sync_clip(video_path, audio_path, output_path)
        result["fallback_from"] = "musetalk"
        return result
