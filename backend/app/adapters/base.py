from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.models.scene import Scene

class BaseLLMAdapter(ABC):
    """Abstract interface for LLM story and script generation."""
    @abstractmethod
    async def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt for entities, mood, setting, pacing, and visual theme."""
        pass

    @abstractmethod
    async def generate_story(self, prompt: str, analysis: Dict[str, Any], duration: int) -> Dict[str, Any]:
        """Generate cohesive multi-scene storyline."""
        pass

    @abstractmethod
    async def generate_scenes(self, story: Dict[str, Any], target_duration: int, style: str) -> List[Scene]:
        """Break storyline into timed, visually-specified Scene objects."""
        pass

class BaseVisionConsistencyAdapter(ABC):
    """Abstract interface for character, style, and visual consistency."""
    @abstractmethod
    def enrich_scene_prompts(
        self,
        scenes: List[Scene],
        global_style: str,
        character_refs: Optional[Dict[str, Any]] = None,
        aspect_ratio: str = "16:9",
    ) -> List[Scene]:
        """Enrich scenes with coherent style tags, lighting, and character anchors."""
        pass

class BaseImageAdapter(ABC):
    """Abstract interface for keyframe generation."""
    @abstractmethod
    async def generate_image(self, prompt: str, output_path: str, style: str = "cinematic", width: int = 1280, height: int = 720) -> str:
        """Generate a visual keyframe saved to output_path."""
        pass

class BaseVideoAdapter(ABC):
    """Abstract interface for scene video clip generation."""
    @abstractmethod
    async def generate_scene_video(self, scene: Scene, keyframe_path: str, output_path: str, width: int = 1280, height: int = 720, fps: int = 24) -> str:
        """Synthesize motion/video from keyframe and scene parameters."""
        pass

class BaseTTSAdapter(ABC):
    """Abstract interface for voice and speech generation."""
    @abstractmethod
    async def synthesize_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """Synthesize voice narration saved to output_path."""
        pass
