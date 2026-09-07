import os
import urllib.request
from app.adapters.base import BaseImageAdapter
from app.adapters.image.local_image_adapter import LocalImageAdapter
from app.core.config import settings
from app.core.logging import telemetry

class OpenAIImageAdapter(BaseImageAdapter):
    """OpenAI DALL-E 3 adapter with fallback to LocalImageAdapter."""
    def __init__(self):
        self.fallback = LocalImageAdapter()
        self.client = None
        if settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            except Exception as e:
                telemetry.emit("openai_image_init_failed", "system", {"error": str(e)}, level="warning")

    async def generate_image(self, prompt: str, output_path: str, style: str = "cinematic", width: int = 1280, height: int = 720) -> str:
        if not self.client:
            return await self.fallback.generate_image(prompt, output_path, style, width, height)
        try:
            enhanced_prompt = f"{prompt}, {style} style, ultra-detailed, photorealistic, 16:9 widescreen composition."
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt[:950],
                size="1792x1024",
                quality="standard",
                n=1
            )
            image_url = response.data[0].url
            urllib.request.urlretrieve(image_url, output_path)
            return output_path
        except Exception as e:
            telemetry.emit("dalle_generation_failed", "system", {"error": str(e)}, level="warning")
            return await self.fallback.generate_image(prompt, output_path, style, width, height)
