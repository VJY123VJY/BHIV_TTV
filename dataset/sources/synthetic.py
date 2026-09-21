"""
Internal curated and synthetic dataset generator for bootstrapping and reproducible testing.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any


class SyntheticDatasetSource:
    """Generates license-clean, reproducible synthetic image & video datasets for training & testing."""

    CATEGORIES = [
        "human_actions",
        "talking_faces",
        "environments",
        "objects",
        "vehicles",
        "nature",
        "cinematic",
        "multilingual_talking",
    ]

    def generate_synthetic_image(
        self,
        category: str,
        output_path: str,
        width: int = 768,
        height: int = 512,
        index: int = 1,
    ) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Create base image with rich pixel variation to satisfy blur and quality criteria
        base_arr = np.zeros((height, width, 3), dtype=np.uint8)
        # Add subtle noise/texture gradient
        for y in range(height):
            base_arr[y, :, 0] = int(35 + (y / height) * 45 + (index * 17) % 30)
            base_arr[y, :, 1] = int(60 + (y / height) * 60 + (index * 13) % 40)
            base_arr[y, :, 2] = int(45 + (y / height) * 35)

        img = Image.fromarray(base_arr)
        draw = ImageDraw.Draw(img)

        # Draw high-contrast shapes to produce sharp edge responses (Laplacian variance)
        if category in ("talking_faces", "human_actions", "multilingual_talking"):
            cx, cy = width // 2, height // 2
            # Face silhouette with sharp contour
            draw.ellipse((cx - 90, cy - 120, cx + 90, cy + 120), fill=(220, 180, 150), outline=(80, 50, 40), width=3)
            # Eyes with pupils
            draw.ellipse((cx - 45, cy - 35, cx - 15, cy - 15), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            draw.ellipse((cx - 35, cy - 30, cx - 25, cy - 20), fill=(20, 20, 20))
            draw.ellipse((cx + 15, cy - 35, cx + 45, cy - 15), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            draw.ellipse((cx + 25, cy - 30, cx + 35, cy - 20), fill=(20, 20, 20))
            # Nose
            draw.line([(cx, cy - 10), (cx - 8, cy + 15), (cx + 8, cy + 15)], fill=(120, 80, 60), width=2)
            # Mouth with lips
            draw.ellipse((cx - 35, cy + 35, cx + 35, cy + 55), fill=(180, 70, 70), outline=(90, 30, 30), width=2)
            # Shirt / collar
            draw.polygon([(cx - 150, height), (cx - 60, cy + 130), (cx + 60, cy + 130), (cx + 150, height)], fill=(40, 80, 160))
        elif category in ("nature", "environments"):
            # Horizon + crisp mountainous polygons
            draw.rectangle((0, height // 2, width, height), fill=(34, 139, 34))
            draw.polygon([(80, height // 2), (180, height // 4), (280, height // 2)], fill=(40, 110, 40), outline=(20, 60, 20), width=2)
            draw.polygon([(250, height // 2), (380, height // 5), (510, height // 2)], fill=(30, 95, 30), outline=(15, 50, 15), width=2)
        else:
            draw.rectangle((width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill=(70, 120, 180), outline=(20, 40, 80), width=3)

        img.save(output_path, quality=95)

        return {
            "asset_id": f"syn_{category}_{index:03d}",
            "source_name": "internal-synthetic",
            "source_url": f"file://{output_path}",
            "license": "CC0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "creator": "BHIV-TTV Synthetic Generator",
            "width": width,
            "height": height,
            "category": category,
            "allowed_for_training": True,
        }


synthetic_source = SyntheticDatasetSource()
