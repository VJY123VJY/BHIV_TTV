import json
import os
from typing import Dict, Any, List
from app.adapters.base import BaseLLMAdapter
from app.adapters.llm.local_adapter import LocalLLMAdapter
from app.models.scene import Scene
from app.core.config import settings
from app.core.logging import telemetry

class GeminiLLMAdapter(BaseLLMAdapter):
    """Google Gemini LLM Adapter with fallback to LocalLLMAdapter."""
    def __init__(self):
        self.fallback = LocalLLMAdapter()
        self.model = None
        if settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception as e:
                telemetry.emit("gemini_init_failed", "system", {"error": str(e)}, level="warning")

    async def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        if not self.model:
            return await self.fallback.analyze_prompt(prompt)
        try:
            req = f"Analyze this prompt for video production: '{prompt}'. Return JSON only with: subject, setting, lighting, mood, keywords."
            resp = self.model.generate_content(req)
            text = resp.text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            data = json.loads(text)
            data["raw_prompt"] = prompt
            return data
        except Exception:
            return await self.fallback.analyze_prompt(prompt)

    async def generate_story(self, prompt: str, analysis: Dict[str, Any], duration: int) -> Dict[str, Any]:
        if not self.model:
            return await self.fallback.generate_story(prompt, analysis, duration)
        try:
            req = f"Create a short video story for: '{prompt}'. Output JSON only: title, premise, acts (array of 3 acts with act, title, beat, narrative)."
            resp = self.model.generate_content(req)
            text = resp.text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            return json.loads(text)
        except Exception:
            return await self.fallback.generate_story(prompt, analysis, duration)

    async def generate_scenes(self, story: Dict[str, Any], target_duration: int, style: str) -> List[Scene]:
        return await self.fallback.generate_scenes(story, target_duration, style)
