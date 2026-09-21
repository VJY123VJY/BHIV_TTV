import re
from typing import Dict, Any
from app.core.exceptions import ValidationError
from app.adapters.llm import get_llm_adapter
from app.core.config import settings
from app.services.prompt_constraints import prompt_constraint_service

class PromptService:
    """Service handling prompt validation, sanitization, and semantic understanding."""
    def __init__(self):
        self.llm_adapter = get_llm_adapter(settings.LLM_PROVIDER)

    def validate_prompt(self, prompt: str) -> str:
        """Enforce strict prompt boundary discipline and sanitization."""
        if not prompt or not isinstance(prompt, str):
            raise ValidationError("Prompt cannot be empty and must be a string.")

        cleaned = prompt.strip()
        if len(cleaned) < 3:
            raise ValidationError(f"Prompt '{cleaned}' is too short (minimum 3 characters).")
        if len(cleaned) > 2000:
            raise ValidationError("Prompt exceeds maximum length of 2000 characters.")

        # Sanitize control characters
        cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", cleaned)
        return cleaned

    async def understand_prompt(self, prompt: str) -> Dict[str, Any]:
        """Extract entities, visual cues, setting, and mood using LLM adapter."""
        analysis = await self.llm_adapter.analyze_prompt(prompt)
        constraints = prompt_constraint_service.build(prompt, analysis)
        analysis["prompt_constraints"] = constraints.to_dict()
        analysis["negative_prompt"] = constraints.negative_prompt()
        return analysis

prompt_service = PromptService()
