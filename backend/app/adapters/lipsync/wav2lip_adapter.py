"""Optional neural Wav2Lip adapter with viseme fallback."""
from __future__ import annotations

import os
from typing import Dict, Any

from app.adapters.lipsync.base import BaseLipSyncAdapter
from app.adapters.lipsync.viseme_adapter import VisemeLipSyncAdapter
from app.core.config import settings
from app.core.logging import telemetry


class Wav2LipAdapter(BaseLipSyncAdapter):
    """
    Uses a local Wav2Lip checkpoint when LIPSYNC_MODEL_PATH is configured.
    Falls back to the viseme engine so generation never silently skips mouth motion.
    """

    def __init__(self):
        self.fallback = VisemeLipSyncAdapter()
        self.model_path = settings.LIPSYNC_MODEL_PATH

    async def sync_clip(self, video_path: str, audio_path: str, output_path: str) -> Dict[str, Any]:
        if not self.model_path or not os.path.exists(self.model_path):
            telemetry.emit("wav2lip_unavailable", "lipsync", {"reason": "checkpoint_missing"}, level="warning")
            result = await self.fallback.sync_clip(video_path, audio_path, output_path)
            result["engine"] = result.get("engine", "viseme")
            result["fallback_from"] = "wav2lip"
            return result

        try:
            # Optional local inference entrypoint. Projects that vendor Wav2Lip can
            # expose `inference.main` without changing the TTV pipeline.
            from inference import main as wav2lip_main  # type: ignore
            wav2lip_main(checkpoint_path=self.model_path, face=video_path, audio=audio_path, outfile=output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return {"applied": True, "reason": "wav2lip", "output_path": output_path, "engine": "wav2lip"}
        except Exception as exc:
            telemetry.emit("wav2lip_failed", "lipsync", {"error": str(exc)}, level="warning")

        result = await self.fallback.sync_clip(video_path, audio_path, output_path)
        result["fallback_from"] = "wav2lip"
        return result
