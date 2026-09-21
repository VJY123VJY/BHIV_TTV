"""Base interface for reference media retrieval adapters."""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseReferenceAdapter(ABC):
    @abstractmethod
    def can_handle(self, url: str, parsed) -> bool:
        pass

    @abstractmethod
    async def retrieve(self, url: str, dest_dir: str) -> Dict[str, Any]:
        """
        Retrieve public media into dest_dir.

        Returns a dict with:
          path, media_type ('image'|'video'), source, filename
        """
        pass
