import os
import json
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
from app.core.config import settings
from app.models.scene import Scene
from app.core.logging import telemetry

class StorageService:
    """
    Unified storage and artifact manager.
    Consolidates BucketAdapter implementations from ttv3 and ttv6.
    Generates metadata sidecars (.json) paired with each video deliverable.
    """
    def __init__(self):
        self.output_dir = settings.get_absolute_path(settings.OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def persist_video_artifact(
        self,
        execution_id: str,
        video_path: str,
        prompt: str,
        scenes: List[Scene],
        duration: float,
        style: str,
        resolution: str = "1280x720",
        fps: int = 24,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persists video, writes sidecar metadata JSON, and returns artifact payload."""
        final_filename = f"{execution_id}.mp4"
        final_dest = self.output_dir / final_filename

        if os.path.abspath(video_path) != os.path.abspath(final_dest):
            shutil.copy2(video_path, str(final_dest))

        file_size = os.path.getsize(final_dest)

        # Build comprehensive metadata sidecar
        metadata = {
            "video_id": execution_id,
            "filename": final_filename,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "fps": fps,
            "style": style,
            "file_size_bytes": file_size,
            "video_url": f"/generated/videos/{final_filename}",
            "scenes": [s.to_dict() for s in scenes],
            "created_at": Path(final_dest).stat().st_mtime
        }
        if extra:
            metadata.update(extra)

        # Save metadata JSON sidecar
        meta_file = self.output_dir / f"{execution_id}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        telemetry.emit("artifact_persisted", execution_id, {
            "video_file": str(final_dest),
            "meta_file": str(meta_file),
            "size_bytes": file_size
        })

        return metadata

    def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        meta_file = self.output_dir / f"{video_id}_metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata for video '{video_id}' not found.")
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def persist_subtitle_artifact(self, execution_id: str, source_path: str, extension: str) -> str:
        """Copy a generated subtitle sidecar beside the final MP4 and return its URL."""
        destination = self.output_dir / f"{execution_id}.{extension.lstrip('.')}"
        if os.path.abspath(source_path) != os.path.abspath(destination):
            shutil.copy2(source_path, destination)
        return f"/generated/videos/{destination.name}"

    def get_video_path(self, video_id: str) -> Path:
        video_file = self.output_dir / f"{video_id}.mp4"
        if not video_file.exists():
            raise FileNotFoundError(f"Video file for '{video_id}' not found.")
        return video_file

storage_service = StorageService()
