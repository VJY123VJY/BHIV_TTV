"""Lip-sync adapter interface."""
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
