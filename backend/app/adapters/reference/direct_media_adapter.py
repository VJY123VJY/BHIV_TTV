"""Retrieve public direct image/video URLs with SSRF protection, size limits, redirect checks, and media signature verification."""
from __future__ import annotations

import os
import time
import uuid
from typing import Dict, Any
from urllib.parse import urlparse, urljoin

import httpx

from app.adapters.reference.base import BaseReferenceAdapter
from app.adapters.reference.validator import (
    validate_public_url,
    validate_media_file,
    detect_source,
    safe_log_url,
    ReferenceValidationError,
)
from app.core.config import settings
from app.core.exceptions import ValidationError
from app.core.logging import telemetry, logger

MAX_REDIRECTS = 5


class DirectMediaAdapter(BaseReferenceAdapter):
    def can_handle(self, url: str, parsed) -> bool:
        return detect_source(parsed.hostname or "") == "direct"

    async def retrieve(self, url: str, dest_dir: str) -> Dict[str, Any]:
        start_time = time.time()
        safe_url, source = validate_public_url(url, resolve_dns=True)
        timeout = httpx.Timeout(settings.REFERENCE_DOWNLOAD_TIMEOUT)
        max_bytes = settings.REFERENCE_MAX_BYTES
        os.makedirs(dest_dir, exist_ok=True)

        current_url = safe_url
        headers = {
            "User-Agent": "BHIV-TTV-Studio/1.0 (reference-fetch)",
            "Accept": "*/*",
        }

        temp_filename = f"dl_{uuid.uuid4().hex[:8]}.tmp"
        temp_path = os.path.join(dest_dir, temp_filename)
        written = 0
        status_code = None

        try:
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                # Step 1: Follow redirects manually to validate every hop against SSRF rules
                for hop in range(MAX_REDIRECTS + 1):
                    safe_hop_url, _ = validate_public_url(current_url, resolve_dns=True)

                    # Stream the response to avoid loading entire body into memory
                    async with client.stream("GET", safe_hop_url, follow_redirects=False) as response:
                        status_code = response.status_code

                        # Check for redirects
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location")
                            if not location:
                                raise ValidationError("Redirect missing Location header.")
                            next_url = urljoin(safe_hop_url, location)
                            # Strictly validate the redirect target against SSRF (private IPs, localhost, etc.)
                            validate_public_url(next_url, resolve_dns=True)
                            current_url = next_url
                            continue

                        # Check HTTP status
                        if response.status_code == 401:
                            raise ValidationError("This URL requires login.")
                        elif response.status_code in {403, 404}:
                            raise ValidationError("This URL is private or inaccessible.")
                        elif response.status_code >= 400:
                            raise ValidationError("The media could not be downloaded.")

                        # Check Content-Length header if present
                        content_length = response.headers.get("content-length")
                        if content_length:
                            try:
                                if int(content_length) > max_bytes:
                                    raise ValidationError("The file is too large.")
                            except ValueError:
                                pass

                        # Stream download into temporary file with size check
                        with open(temp_path, "wb") as handle:
                            async for chunk in response.aiter_bytes(64 * 1024):
                                written += len(chunk)
                                if written > max_bytes:
                                    handle.close()
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                    raise ValidationError("The file is too large.")
                                handle.write(chunk)

                        break
                else:
                    raise ValidationError("The media could not be downloaded: too many redirects.")

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise ValidationError("The media could not be downloaded.") from exc
        except (ReferenceValidationError, ValidationError):
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        except Exception as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise ValidationError("The media could not be downloaded.") from exc

        # Step 2: Validate file signatures and actual readability via Pillow/OpenCV
        try:
            meta = validate_media_file(temp_path, max_bytes=max_bytes)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        # Step 3: Rename to safe final destination filename
        dest_filename = f"reference{meta['ext']}"
        dest_path = os.path.join(dest_dir, dest_filename)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)

        duration_sec = round(time.time() - start_time, 3)
        source_host = urlparse(safe_url).hostname or ""

        # Step 4: Structured logging without sensitive query tokens
        log_data = {
            "source_host": source_host,
            "url_type": "direct",
            "detection_result": "direct_media",
            "http_status": status_code,
            "media_type": meta["media_type"],
            "file_size": meta["size_bytes"],
            "processing_duration": duration_sec,
            "final_validation_result": "valid",
            "sanitized_url": safe_log_url(safe_url),
            "width": meta.get("width"),
            "height": meta.get("height"),
        }
        telemetry.emit("reference_direct_processed", source_host, log_data)
        logger.info(f"[ReferenceDirectMedia] Success: {source_host} -> {meta['media_type']} ({meta['size_bytes']} bytes)")

        return {
            "path": dest_path,
            "media_type": meta["media_type"],
            "source": source,
            "filename": dest_filename,
            "bytes": meta["size_bytes"],
            "width": meta.get("width"),
            "height": meta.get("height"),
            "duration": meta.get("duration", 0.0),
        }
