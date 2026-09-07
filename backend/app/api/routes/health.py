import shutil
from fastapi import APIRouter
from app.schemas.response import HealthResponse
from app.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check and provider readiness status."""
    ffmpeg_bin = shutil.which("ffmpeg") is not None
    return HealthResponse(
        status="healthy",
        service="unified-text-to-video",
        version="1.0.0",
        providers={
            "llm": settings.LLM_PROVIDER,
            "image": settings.IMAGE_PROVIDER,
            "video": settings.VIDEO_PROVIDER,
            "tts": settings.TTS_PROVIDER,
            "vision": settings.VISION_PROVIDER
        },
        ffmpeg_available=ffmpeg_bin
    )
