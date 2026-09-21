"""Public YouTube / Instagram / Facebook / X media retrieval via yt-dlp."""
from __future__ import annotations

import os
from typing import Dict, Any
from urllib.parse import urlparse

from app.adapters.reference.base import BaseReferenceAdapter
from app.adapters.reference.validator import detect_source, validate_public_url
from app.core.config import settings
from app.core.exceptions import ValidationError
from app.core.logging import telemetry


class SocialMediaAdapter(BaseReferenceAdapter):
    HANDLED = {"youtube", "instagram", "facebook", "x"}

    def __init__(self, source: str):
        self.source = source

    def can_handle(self, url: str, parsed) -> bool:
        return detect_source(parsed.hostname or "") == self.source

    async def retrieve(self, url: str, dest_dir: str) -> Dict[str, Any]:
        safe_url, source = validate_public_url(url, resolve_dns=True)
        os.makedirs(dest_dir, exist_ok=True)
        try:
            import yt_dlp
        except ImportError as exc:
            raise ValidationError(
                f"{source.title()} reference URLs require yt-dlp. Install it or paste a direct image/video URL."
            ) from exc

        outtmpl = os.path.join(dest_dir, "reference.%(ext)s")
        opts = {
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": settings.REFERENCE_DOWNLOAD_TIMEOUT,
            "retries": 1,
            "format": "mp4/best[height<=720]/best",
            "max_filesize": settings.REFERENCE_MAX_BYTES,
            "restrictfilenames": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(safe_url, download=True)
        except Exception as exc:
            telemetry.emit("reference_social_failed", "reference", {"source": source, "error": str(exc)}, level="warning")
            raise ValidationError(
                f"Could not retrieve public media from this {source.title()} URL. "
                "Confirm the post is public and that downloading is permitted."
            ) from exc

        if info is None:
            raise ValidationError(f"No public media was found at this {source.title()} URL.")

        downloaded = info.get("requested_downloads") or []
        filepath = None
        if downloaded and downloaded[0].get("filepath"):
            filepath = downloaded[0]["filepath"]
        elif info.get("filename"):
            filepath = info["filename"]
        else:
            for name in os.listdir(dest_dir):
                if name.startswith("reference."):
                    filepath = os.path.join(dest_dir, name)
                    break
        if not filepath or not os.path.exists(filepath):
            raise ValidationError(f"The {source.title()} download did not produce a usable media file.")

        ext = os.path.splitext(filepath)[1].lower()
        media_type = "image" if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"} else "video"
        return {
            "path": filepath,
            "media_type": media_type,
            "source": source,
            "filename": os.path.basename(filepath),
            "bytes": os.path.getsize(filepath),
            "title": info.get("title"),
        }
