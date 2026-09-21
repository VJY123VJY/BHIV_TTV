"""
Single source of truth for video format, quality, and output dimensions.

Do not hardcode resolution tuples in services, adapters, or the frontend.
Clients should consume GET /api/v1/generation-options, which is built from this module.
"""
from typing import Any, Dict, Optional, Tuple

from app.core.exceptions import ValidationError

ASPECT_RATIOS = ("16:9", "9:16")
QUALITIES = ("standard", "high", "ultra")

VIDEO_SETTINGS: Dict[str, Dict[str, Any]] = {
    "16:9": {
        "label": "Landscape / YouTube",
        "standard": (1280, 720),
        "high": (1920, 1080),
        "ultra": (3840, 2160),
    },
    "9:16": {
        "label": "Vertical / Reels / Shorts",
        "standard": (720, 1280),
        "high": (1080, 1920),
        "ultra": (2160, 3840),
    },
}

QUALITY_LABELS = {
    "standard": "720p",
    "high": "1080p",
    "ultra": "4K",
}

# Backward-compatible aliases for older API clients that send a raw WxH string.
_RESOLUTION_LOOKUP = {
    "1280x720": ("16:9", "standard"),
    "1920x1080": ("16:9", "high"),
    "3840x2160": ("16:9", "ultra"),
    "720x1280": ("9:16", "standard"),
    "1080x1920": ("9:16", "high"),
    "2160x3840": ("9:16", "ultra"),
}


def normalize_aspect_ratio(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip().lower().replace(" ", "")
    aliases = {
        "16:9": "16:9",
        "16x9": "16:9",
        "landscape": "16:9",
        "youtube": "16:9",
        "widescreen": "16:9",
        "9:16": "9:16",
        "9x16": "9:16",
        "portrait": "9:16",
        "vertical": "9:16",
        "reels": "9:16",
        "shorts": "9:16",
    }
    return aliases.get(cleaned, str(value).strip())


def normalize_quality(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    aliases = {
        "standard": "standard",
        "720": "standard",
        "720p": "standard",
        "sd": "standard",
        "high": "high",
        "1080": "high",
        "1080p": "high",
        "hd": "high",
        "fhd": "high",
        "ultra": "ultra",
        "4k": "ultra",
        "2160": "ultra",
        "2160p": "ultra",
        "uhd": "ultra",
    }
    return aliases.get(cleaned, cleaned)


def get_resolution(aspect_ratio: str, quality: str) -> Tuple[int, int]:
    aspect = normalize_aspect_ratio(aspect_ratio)
    quality_key = normalize_quality(quality)
    if aspect not in VIDEO_SETTINGS:
        raise ValidationError(
            f"Unsupported aspect ratio '{aspect_ratio}'. Supported: {', '.join(ASPECT_RATIOS)}."
        )
    if quality_key not in QUALITIES:
        raise ValidationError(
            f"Unsupported quality '{quality}'. Supported: {', '.join(QUALITIES)}."
        )
    return VIDEO_SETTINGS[aspect][quality_key]


def format_resolution(width: int, height: int) -> str:
    return f"{int(width)}x{int(height)}"


def parse_resolution_string(resolution: str) -> Tuple[int, int]:
    parts = str(resolution).lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValidationError(f"Invalid resolution '{resolution}'. Expected format WIDTHxHEIGHT.")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise ValidationError(f"Invalid resolution '{resolution}'.") from exc
    if width < 16 or height < 16 or width > 7680 or height > 7680:
        raise ValidationError(f"Resolution '{resolution}' is outside the supported range.")
    return width, height


def infer_from_resolution(resolution: str) -> Tuple[str, str, int, int]:
    key = str(resolution).lower().replace(" ", "")
    if key in _RESOLUTION_LOOKUP:
        aspect, quality = _RESOLUTION_LOOKUP[key]
        width, height = VIDEO_SETTINGS[aspect][quality]
        return aspect, quality, width, height
    width, height = parse_resolution_string(resolution)
    aspect = "9:16" if height > width else "16:9"
    quality = "standard"
    return aspect, quality, width, height


def resolve_video_settings(
    aspect_ratio: Optional[str] = None,
    quality: Optional[str] = None,
    resolution: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve aspect ratio, quality, and pixel dimensions.

    Explicit aspect_ratio + quality always wins over a global default.
    A legacy `resolution` string is honored when the new fields are omitted.
    """
    aspect = normalize_aspect_ratio(aspect_ratio) if aspect_ratio else None
    quality_key = normalize_quality(quality) if quality else None

    if aspect and aspect not in VIDEO_SETTINGS:
        raise ValidationError(
            f"Unsupported aspect ratio '{aspect_ratio}'. Supported: {', '.join(ASPECT_RATIOS)}."
        )
    if quality_key and quality_key not in QUALITIES:
        raise ValidationError(
            f"Unsupported quality '{quality}'. Supported: {', '.join(QUALITIES)}."
        )

    if aspect and quality_key:
        width, height = get_resolution(aspect, quality_key)
    elif quality_key:
        aspect = aspect or "16:9"
        width, height = get_resolution(aspect, quality_key)
    elif aspect:
        quality_key = "standard"
        width, height = get_resolution(aspect, quality_key)
    elif resolution:
        aspect, quality_key, width, height = infer_from_resolution(resolution)
    else:
        aspect = "16:9"
        quality_key = "standard"
        width, height = get_resolution(aspect, quality_key)

    return {
        "aspect_ratio": aspect,
        "quality": quality_key,
        "width": width,
        "height": height,
        "resolution": format_resolution(width, height),
        "quality_label": QUALITY_LABELS[quality_key],
        "aspect_label": VIDEO_SETTINGS[aspect]["label"],
    }


def catalog() -> Dict[str, Any]:
    matrix = {}
    for aspect in ASPECT_RATIOS:
        matrix[aspect] = {
            quality: format_resolution(*VIDEO_SETTINGS[aspect][quality])
            for quality in QUALITIES
        }
    return {
        "aspect_ratios": [
            {"id": key, "label": VIDEO_SETTINGS[key]["label"]}
            for key in ASPECT_RATIOS
        ],
        "qualities": [
            {
                "id": key,
                "label": key.title(),
                "resolution_label": QUALITY_LABELS[key],
            }
            for key in QUALITIES
        ],
        "resolution_matrix": matrix,
    }
