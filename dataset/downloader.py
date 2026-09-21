"""Bounded, resumable downloader used only after source-license verification."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
import httpx


class DownloadError(RuntimeError):
    pass


def extension_for_url(url: str, media_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".webm", ".mov", ".mkv", ".wav", ".mp3"}
    if suffix in allowed:
        return suffix
    return ".mp4" if media_type == "video" else ".jpg"


def download_resumable(url: str, destination: Path, max_bytes: int, timeout: int = 30) -> Path:
    """Download ``url`` to ``destination`` using a ``.part`` resume file.

    The caller is responsible for ensuring that the URL came from an approved
    source and that its license was verified.  A content-length and streamed
    byte cap prevent an ingestion command from accidentally consuming a disk.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "BHIV-TTV-DatasetEngine/2.0", "Range": f"bytes={existing}-"} if existing else {"User-Agent": "BHIV-TTV-DatasetEngine/2.0"}
    try:
        with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=timeout) as response:
            response.raise_for_status()
            # A server that ignores Range returns a full 200 response; restart
            # instead of appending a duplicate payload.
            append = existing > 0 and response.status_code == 206
            if not append:
                existing = 0
            advertised = response.headers.get("content-length")
            if advertised and existing + int(advertised) > max_bytes:
                raise DownloadError(f"asset exceeds configured limit ({max_bytes} bytes)")
            written = existing
            with part.open("ab" if append else "wb") as stream:
                for chunk in response.iter_bytes():
                    written += len(chunk)
                    if written > max_bytes:
                        raise DownloadError(f"asset exceeds configured limit ({max_bytes} bytes)")
                    stream.write(chunk)
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise DownloadError(str(exc)) from exc
    part.replace(destination)
    return destination
