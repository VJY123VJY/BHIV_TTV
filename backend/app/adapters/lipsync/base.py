"""Lip-sync adapter interface (LipSyncProvider)."""
import os
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLipSyncAdapter(ABC):
    @abstractmethod
    async def sync_clip(self, video_path: str, audio_path: str, output_path: str) -> Dict[str, Any]:
        """
        Drive character mouth motion from dialogue audio.

        Returns:
            applied (bool), reason (str), output_path (str)
        """
        pass

    async def generate_lipsync(self, video_path: str, audio_path: str, output_path: str) -> Dict[str, Any]:
        """Alias for sync_clip adhering to standard naming convention."""
        return await self.sync_clip(video_path, audio_path, output_path)

    def validate_input(self, video_path: str, audio_path: str) -> bool:
        """Validate presence and readability of source video and dialogue audio tracks."""
        if not video_path or not os.path.exists(video_path):
            return False
        if not audio_path or not os.path.exists(audio_path):
            return False
        return True

    def get_status(self) -> Dict[str, Any]:
        """Return provider status and capabilities."""
        return {
            "provider": self.__class__.__name__,
            "available": True,
            "status": "ready"
        }


LipSyncProvider = BaseLipSyncAdapter
