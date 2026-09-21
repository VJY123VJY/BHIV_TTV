from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SceneInfo(BaseModel):
    index: int
    prompt: str
    narrative: str
    duration: float
    image_url: Optional[str] = None
    video_url: Optional[str] = None


class VideoMetadata(BaseModel):
    video_id: str
    prompt: str
    duration: float
    resolution: str
    fps: int
    style: str
    voice_enabled: bool
    file_size_bytes: int
    video_url: str
    scenes: List[SceneInfo] = []
    created_at: str


class GenerationSettings(BaseModel):
    aspect_ratio: str
    quality: str
    resolution: str
    style: str
    language: str
    fps: int
    lipsync: bool = True
    character_id: Optional[str] = None
    subtitles: bool = True
    realtime_data: bool = False


class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Generation job submitted successfully"
    settings: Optional[GenerationSettings] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    stage: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    service: str = "unified-text-to-video"
    version: str = "1.0.0"
    providers: Dict[str, str]
    ffmpeg_available: bool


class ReferenceUploadResponse(BaseModel):
    reference_id: str
    media_type: str
    source: str = "upload"
    message: str = "Reference stored successfully"
