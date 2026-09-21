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
            dalle_size = "1792x1024" if width >= height else "1024x1792"
            aspect_hint = "16:9 widescreen composition" if width >= height else "9:16 vertical composition"
            enhanced_prompt = f"{prompt}, {style} style, ultra-detailed, {aspect_hint}."
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt[:950],
                size=dalle_size,
                quality="standard",
                n=1
            )
            image_url = response.data[0].url
            urllib.request.urlretrieve(image_url, output_path)
            from PIL import Image
            with Image.open(output_path) as img:
                resized = img.convert("RGB").resize((width, height))
                resized.save(output_path, quality=92)
            return output_path
        except Exception as e:
            telemetry.emit("dalle_generation_failed", "system", {"error": str(e)}, level="warning")
            return await self.fallback.generate_image(prompt, output_path, style, width, height)
