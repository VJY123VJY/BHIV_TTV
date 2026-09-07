"""
LLM adapters for narrative, story, and script generation.
"""
from app.adapters.llm.local_adapter import LocalLLMAdapter
from app.adapters.llm.openai_adapter import OpenAILLMAdapter
from app.adapters.llm.gemini_adapter import GeminiLLMAdapter

def get_llm_adapter(provider: str = "local"):
    """Factory to retrieve configured LLM adapter."""
    p = (provider or "local").lower()
    if p == "openai":
        return OpenAILLMAdapter()
    elif p == "gemini":
        return GeminiLLMAdapter()
    return LocalLLMAdapter()
