from typing import Dict, Any, List
from app.models.scene import Scene
from app.adapters.llm import get_llm_adapter
from app.core.config import settings

class SceneService:
    """
    Orchestrates scene breakdown, timing allocation, and cinematic composition.
    Integrates shot planning, camera sequencing, and pacing from ttv_converegence.
    """
    def __init__(self):
        self.llm_adapter = get_llm_adapter(settings.LLM_PROVIDER)

    async def generate_scenes(self, story: Dict[str, Any], duration: int, style: str) -> List[Scene]:
        """Break down story into timed, planned scenes with cinematic camera movement."""
        scenes = await self.llm_adapter.generate_scenes(story, duration, style)
        
        # Validate that cumulative duration matches target
        total_d = sum(s.duration for s in scenes)
        if total_d > 0 and abs(total_d - duration) > 1.0:
            scale = duration / total_d
            for s in scenes:
                s.duration = round(s.duration * scale, 2)

        return scenes

scene_service = SceneService()
