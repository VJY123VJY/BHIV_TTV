import json
import os
from typing import Dict, Any, List
from app.adapters.base import BaseLLMAdapter
from app.adapters.llm.local_adapter import LocalLLMAdapter
from app.models.scene import Scene
from app.core.config import settings
from app.core.logging import telemetry

class OpenAILLMAdapter(BaseLLMAdapter):
    """OpenAI GPT Adapter with graceful fallback to LocalLLMAdapter."""
    def __init__(self):
        self.fallback = LocalLLMAdapter()
        self.client = None
        if settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            except Exception as e:
                telemetry.emit("openai_init_failed", "system", {"error": str(e)}, level="warning")

    async def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        if not self.client:
            return await self.fallback.analyze_prompt(prompt)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Analyze the video prompt. Output JSON with keys: subject, setting, lighting, mood, keywords."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            data["raw_prompt"] = prompt
            return data
        except Exception:
            return await self.fallback.analyze_prompt(prompt)

    async def generate_story(self, prompt: str, analysis: Dict[str, Any], duration: int) -> Dict[str, Any]:
        if not self.client:
            return await self.fallback.generate_story(prompt, analysis, duration)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Create a 3-act story structure for a short video. Output JSON with: title, premise, acts (array of objects with act, title, beat, narrative)."},
                    {"role": "user", "content": f"Prompt: {prompt}, Duration: {duration}s, Details: {json.dumps(analysis)}"}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return await self.fallback.generate_story(prompt, analysis, duration)

    async def generate_scenes(self, story: Dict[str, Any], target_duration: int, style: str) -> List[Scene]:
        # Scenes breakdown is structured and consistent
        return await self.fallback.generate_scenes(story, target_duration, style)
