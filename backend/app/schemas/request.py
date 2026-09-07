from typing import Optional
from pydantic import BaseModel, Field

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Text prompt describing the video to generate")
    duration: Optional[int] = Field(default=15, ge=5, le=60, description="Desired video duration in seconds (5-60s)")
    style: Optional[str] = Field(default="cinematic", description="Visual style: cinematic, anime, realistic, cyberpunk, fantasy")
    voice: Optional[bool] = Field(default=True, description="Whether to generate speech narration")
    resolution: Optional[str] = Field(default="1280x720", description="Target video resolution (e.g. 1280x720, 1920x1080)")
    fps: Optional[int] = Field(default=24, ge=15, le=60, description="Frames per second")
    token: Optional[str] = Field(default=None, description="Optional governance authorization token")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "A small robot explores a futuristic city at sunset.",
                "duration": 15,
                "style": "cinematic",
                "voice": True
            }
        }
    }
