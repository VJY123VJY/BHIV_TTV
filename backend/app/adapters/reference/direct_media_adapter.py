"""Retrieve public direct image/video URLs with size, timeout, and type checks."""
from __future__ import annotations

import os
from typing import Dict, Any
from urllib.parse import urlparse

import httpx

from app.adapters.reference.base import BaseReferenceAdapter
from app.adapters.reference.validator import detect_source, validate_public_url
from app.core.config import settings
from app.core.exceptions import ValidationError

IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
VIDEO_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}
EXT_FALLBACK = {
    ".jpg": ("image", ".jpg"),
    ".jpeg": ("image", ".jpg"),
    ".png": ("image", ".png"),
    ".webp": ("image", ".webp"),
    ".gif": ("image", ".gif"),
    ".mp4": ("video", ".mp4"),
    ".webm": ("video", ".webm"),
    ".mov": ("video", ".mov"),
}


class DirectMediaAdapter(BaseReferenceAdapter):
    def can_handle(self, url: str, parsed) -> bool:
        return detect_source(parsed.hostname or "") == "direct"

    async def retrieve(self, url: str, dest_dir: str) -> Dict[str, Any]:
        safe_url, source = validate_public_url(url, resolve_dns=True)
        timeout = httpx.Timeout(settings.REFERENCE_DOWNLOAD_TIMEOUT)
        max_bytes = settings.REFERENCE_MAX_BYTES
        os.makedirs(dest_dir, exist_ok=True)

        headers = {"User-Agent": "BHIV-TTV-Studio/1.0 (reference-fetch)"}
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            async with client.stream("GET", safe_url) as response:
                if response.status_code >= 400:
                    raise ValidationError("The reference media could not be downloaded. Check that the URL is public.")
                final_host = urlparse(str(response.url)).hostname
                if not final_host:
                    raise ValidationError("The reference URL redirected to an invalid destination.")
                validate_public_url(str(response.url), resolve_dns=True)

                content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
                media_type, ext = self._classify(safe_url, content_type)
                length = response.headers.get("content-length")
                if length and int(length) > max_bytes:
                    raise ValidationError("The reference file is too large.")

                dest_path = os.path.join(dest_dir, f"reference{ext}")
                written = 0
                with open(dest_path, "wb") as handle:
                    async for chunk in response.aiter_bytes(64 * 1024):
                        written += len(chunk)
                        if written > max_bytes:
                            handle.close()
                            try:
                                os.remove(dest_path)
                            except OSError:
                                pass
                            raise ValidationError("The reference file exceeded the maximum allowed size.")
                        handle.write(chunk)

        if written < 32:
            raise ValidationError("The downloaded reference file is empty or invalid.")
        return {
            "path": dest_path,
            "media_type": media_type,
            "source": source,
            "filename": os.path.basename(dest_path),
            "bytes": written,
        }

    def _classify(self, url: str, content_type: str):
        if content_type in IMAGE_TYPES:
            return "image", IMAGE_TYPES[content_type]
        if content_type in VIDEO_TYPES:
            return "video", VIDEO_TYPES[content_type]
        path = urlparse(url).path.lower()
        for ext, pair in EXT_FALLBACK.items():
            if path.endswith(ext):
                return pair
        raise ValidationError(
            "Unsupported reference media type. Use a public image (JPG/PNG/WEBP) or video (MP4/WEBM) URL."
        )
