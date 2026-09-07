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

class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Generation job submitted successfully"

class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    stage: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str = "unified-text-to-video"
    version: str = "1.0.0"
    providers: Dict[str, str]
    ffmpeg_available: bool
