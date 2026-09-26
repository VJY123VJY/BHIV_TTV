import asyncio
import os
from typing import Dict, Any
from urllib.parse import urlparse

from app.adapters.reference.base import BaseReferenceAdapter
from app.adapters.reference.validator import detect_source, validate_public_url
from app.core.config import settings
from app.core.exceptions import ValidationError
from app.core.logging import telemetry, logger


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
                "Social media downloader (yt-dlp) is not installed on the server. "
                "Please upload the reference file directly."
            ) from exc

        outtmpl = os.path.join(dest_dir, "reference.%(ext)s")
        # Support both standard landscape (16:9) and vertical shorts/reels (9:16)
        opts = {
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": settings.REFERENCE_DOWNLOAD_TIMEOUT,
            "retries": 2,
            "format": (
                "bestvideo[height<=1080][width<=1920]+bestaudio/"
                "bestvideo[height<=1280][width<=1080]+bestaudio/"
                "best[height<=1080]/"
                "best"
            ),
            "merge_output_format": "mp4",
            "max_filesize": settings.REFERENCE_MAX_BYTES,
            "restrictfilenames": True,
            "noprogress": True,
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
            },
        }

        def _download_task():
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(safe_url, download=True)

        try:
            info = await asyncio.to_thread(_download_task)
        except Exception as exc:
            telemetry.emit("reference_social_failed", "reference", {"source": source, "error": str(exc)}, level="warning")
            err_msg = str(exc).strip().splitlines()[-1] if str(exc) else "Media download failed"
            if "Requested format is not available" in err_msg:
                user_msg = f"Could not find a compatible media format from this {source.capitalize()} URL. Please upload the file manually."
            elif "Private video" in err_msg or "Sign in" in err_msg:
                user_msg = f"This {source.capitalize()} media is private or requires sign-in. Please use a public URL or upload manually."
            elif "Video unavailable" in err_msg:
                user_msg = f"This {source.capitalize()} media is unavailable or deleted. Please check the URL."
            else:
                user_msg = f"Could not process reference URL ({err_msg}). Please use a public direct media URL or upload the file manually."
            raise ValidationError(user_msg) from exc

        if info is None:
            raise ValidationError(
                f"Could not extract media metadata from this {source.capitalize()} URL. "
                "Please use a public direct media URL or upload the file manually."
            )

        downloaded = info.get("requested_downloads") or []
        filepath = None
        if downloaded and downloaded[0].get("filepath") and os.path.exists(downloaded[0]["filepath"]):
            filepath = downloaded[0]["filepath"]
        elif info.get("filename") and os.path.exists(info["filename"]):
            filepath = info["filename"]

        if not filepath or not os.path.exists(filepath):
            # Scan destination directory for valid non-temporary files
            candidates = [
                os.path.join(dest_dir, name)
                for name in os.listdir(dest_dir)
                if name.startswith("reference.") and not name.endswith((".part", ".ytdl", ".tmp"))
            ]
            if candidates:
                # Prefer mp4
                mp4s = [c for c in candidates if c.lower().endswith(".mp4")]
                filepath = mp4s[0] if mp4s else candidates[0]
            else:
                # Check if a .part file exists and can be rescued
                part_files = [
                    os.path.join(dest_dir, name)
                    for name in os.listdir(dest_dir)
                    if name.startswith("reference.") and name.endswith(".part")
                ]
                if part_files and os.path.getsize(part_files[0]) > 50000:
                    rescued = part_files[0][:-5]
                    if not os.path.exists(rescued):
                        os.rename(part_files[0], rescued)
                    filepath = rescued

        if not filepath or not os.path.exists(filepath):
            raise ValidationError(
                "This reference URL could not be downloaded to disk. "
                "Please use a public direct media URL or upload the file manually."
            )

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
