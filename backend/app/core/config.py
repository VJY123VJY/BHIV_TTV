import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

# Determine project base directory (workspace root)
# If backend/app/core/config.py, parents[2] is backend, parents[3] is text_to_video_final
CURRENT_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = CURRENT_FILE.parents[3]

class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    # Security & Governance
    GOVERNANCE_LOCK: str = "ON"
    WRAPPER_TOKEN: str = "TTV_SECURE_GOVERNED_TOKEN_2026"

    # Providers
    LLM_PROVIDER: str = "local"
    IMAGE_PROVIDER: str = "local"
    VIDEO_PROVIDER: str = "opencv"
    TTS_PROVIDER: str = "local"
    VISION_PROVIDER: str = "standard"

    # API Keys
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # Paths (relative to workspace root or absolute)
    OUTPUT_DIR: str = "generated/videos"
    TEMP_DIR: str = "generated/temp"
    IMAGES_DIR: str = "generated/images"
    SCENES_DIR: str = "generated/scenes"
    AUDIO_DIR: str = "generated/audio"
    FRONTEND_DIR: str = "frontend"

    # Generation Defaults
    DEFAULT_DURATION: int = 15
    MAX_DURATION: int = 60
    DEFAULT_FPS: int = 24
    DEFAULT_RESOLUTION: str = "1280x720"
    DEFAULT_STYLE: str = "cinematic"
    DEFAULT_VOICE: str = "en-US-GuyNeural"

    # Audio Defaults
    BACKGROUND_MUSIC_ENABLED: bool = True
    AUDIO_DUCKING_RATIO: float = 0.3
    AUDIO_SAMPLE_RATE: int = 44100

    model_config = {
        "env_file": [str(WORKSPACE_ROOT / ".env"), ".env"],
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    def get_absolute_path(self, relative_path: str) -> Path:
        """Resolve a path relative to the workspace root if not already absolute."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return (WORKSPACE_ROOT / p).resolve()

    def ensure_directories(self):
        """Ensure all required asset directories exist."""
        for path_attr in ["OUTPUT_DIR", "TEMP_DIR", "IMAGES_DIR", "SCENES_DIR", "AUDIO_DIR"]:
            p = self.get_absolute_path(getattr(self, path_attr))
            p.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
