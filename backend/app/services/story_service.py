from typing import Dict, Any
from app.adapters.llm import get_llm_adapter
from app.core.config import settings

class StoryService:
    """Service handling narrative arc, premise development, and script generation."""
    def __init__(self):
        self.llm_adapter = get_llm_adapter(settings.LLM_PROVIDER)

    async def generate_story(self, prompt: str, analysis: Dict[str, Any], duration: int) -> Dict[str, Any]:
        """Generate structured 3-act story arc."""
        return await self.llm_adapter.generate_story(prompt, analysis, duration)

story_service = StoryService()
