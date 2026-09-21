"""
Real-time knowledge retrieval service.
Enriches prompts with dynamic real-world facts (e.g. agricultural prices, weather, live updates)
without requiring model retraining.
"""
from typing import Dict, Any, Optional
import httpx
from app.core.config import settings
from app.core.logging import telemetry


class RealtimeDataService:
    REALTIME_KEYWORDS = [
        "today", "current", "price", "rate", "latest", "news", "weather",
        "market", "tomato", "onion", "wheat", "crop", "stock", "mandis"
    ]

    def is_realtime_query(self, prompt: str) -> bool:
        p_lower = prompt.lower()
        return any(k in p_lower for k in self.REALTIME_KEYWORDS)

    async def fetch_realtime_context(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Extracts relevant live market/environmental facts matching the prompt.
        """
        if not settings.ENABLE_REALTIME_DATA or not settings.REALTIME_DATA_URL or not self.is_realtime_query(prompt):
            return None
        try:
            async with httpx.AsyncClient(timeout=settings.REALTIME_DATA_TIMEOUT) as client:
                response = await client.get(settings.REALTIME_DATA_URL, params={"q": prompt})
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            telemetry.emit("realtime_data_unavailable", "realtime", {"error": str(exc)}, level="warning")
            return None
        facts = payload.get("facts") if isinstance(payload, dict) else None
        if not isinstance(facts, list) or not all(isinstance(fact, str) for fact in facts):
            telemetry.emit("realtime_data_rejected", "realtime", {"reason": "invalid_payload"}, level="warning")
            return None
        return {"source": settings.REALTIME_DATA_URL, "facts": facts, "summary": " ".join(facts)}

    def enrich_prompt_with_knowledge(self, prompt: str, realtime_context: Optional[Dict[str, Any]]) -> str:
        """Injects retrieved factual knowledge into the prompt context for LLM story generation."""
        if not realtime_context or not realtime_context.get("summary"):
            return prompt

        enrichment = (
            f"\n[Verified Real-Time Information ({realtime_context['date']})]: "
            f"{realtime_context['summary']} "
            f"Seamlessly weave these verified facts into the character's narrative and dialogue."
        )
        return f"{prompt.strip()}{enrichment}"


realtime_data_service = RealtimeDataService()
