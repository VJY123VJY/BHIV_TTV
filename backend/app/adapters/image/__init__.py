"""
Image generation adapters.
"""
from app.adapters.image.local_image_adapter import LocalImageAdapter
from app.adapters.image.openai_image_adapter import OpenAIImageAdapter

def get_image_adapter(provider: str = "local"):
    p = (provider or "local").lower()
    if p == "openai":
        return OpenAIImageAdapter()
    return LocalImageAdapter()
