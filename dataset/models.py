"""
Data models for dataset ingestion, provenance, licensing, and quality assurance.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class LicenseInfo:
    license: str
    license_url: Optional[str] = None
    creator: Optional[str] = None
    attribution_required: bool = True
    commercial_allowed: bool = True
    training_eligible: bool = True


@dataclass
class QualityMetrics:
    width: int = 0
    height: int = 0
    aspect_ratio: str = "16:9"
    blur_score: float = 0.0
    is_corrupt: bool = False
    is_black_frame: bool = False
    has_audio: bool = False
    speech_detected: bool = False
    snr_db: Optional[float] = None
    phash: Optional[str] = None
    passed_qc: bool = True
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class CaptionMetadata:
    caption: str
    subjects: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    environment: str = "general"
    camera: str = "medium shot"
    lighting: str = "natural daylight"
    weather: str = "clear"
    time: str = "day"
    emotion: str = "neutral"
    motion: str = "static"


@dataclass
class AssetMetadata:
    asset_id: str
    source_url: str
    source_name: str
    license: str
    license_url: Optional[str] = None
    creator: Optional[str] = None
    download_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    media_type: str = "image"  # image | video | audio
    resolution: str = "1280x720"
    duration: float = 0.0
    sha256: str = ""
    phash: Optional[str] = None
    dataset_version: str = "v1.0"
    allowed_for_training: bool = True
    category: str = "general"
    local_path: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    caption_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    total_downloaded: int = 0
    valid: int = 0
    duplicates: int = 0
    low_quality: int = 0
    license_unknown: int = 0
    corrupted: int = 0
    training_ready: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
