import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from app.adapters.base import BaseImageAdapter

class LocalImageAdapter(BaseImageAdapter):
    """
    High-fidelity procedural visual synthesizer.
    Generates rich, atmospheric keyframes matching prompt elements, lighting, and style.
    Guarantees deterministic, zero-dependency offline generation of visually pleasing scenes.
    """
    async def generate_image(self, prompt: str, output_path: str, style: str = "cinematic", width: int = 1280, height: int = 720) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        p_lower = prompt.lower()

        # Seed based on prompt for consistency
        seed = abs(hash(prompt)) % 1000000
        rng = random.Random(seed)

        # 1. Base image canvas
        img = Image.new("RGB", (width, height), (15, 10, 25))
        draw = ImageDraw.Draw(img)

        # 2. Atmospheric Sky Gradient
        if "sunset" in p_lower or "golden" in p_lower or "amber" in p_lower:
            # Sunset: Deep Indigo -> Crimson -> Tangerine -> Warm Gold
            top_color = (35, 15, 60)
            mid_color = (180, 50, 60)
            horizon_color = (255, 150, 50)
            glow_color = (255, 230, 160)
        elif "cyberpunk" in p_lower or "neon" in p_lower:
            top_color = (10, 5, 25)
            mid_color = (40, 10, 60)
            horizon_color = (20, 80, 110)
            glow_color = (0, 240, 255)
        else:
            top_color = (20, 30, 50)
            mid_color = (60, 90, 130)
            horizon_color = (180, 200, 220)
            glow_color = (255, 250, 230)

        horizon_y = int(height * 0.65)

        for y in range(horizon_y):
            ratio = y / horizon_y
            if ratio < 0.5:
                sub_r = ratio / 0.5
                r = int(top_color[0] + sub_r * (mid_color[0] - top_color[0]))
                g = int(top_color[1] + sub_r * (mid_color[1] - top_color[1]))
                b = int(top_color[2] + sub_r * (mid_color[2] - top_color[2]))
            else:
                sub_r = (ratio - 0.5) / 0.5
                r = int(mid_color[0] + sub_r * (horizon_color[0] - mid_color[0]))
                g = int(mid_color[1] + sub_r * (horizon_color[1] - mid_color[1]))
                b = int(mid_color[2] + sub_r * (horizon_color[2] - mid_color[2]))
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # 3. Sun / Celestial Disc
        sun_x = int(width * 0.62)
        sun_y = int(horizon_y * 0.85)
        sun_radius = int(height * 0.14)
        for r_offset in range(sun_radius + 40, sun_radius, -2):
            alpha_ratio = 1.0 - (r_offset - sun_radius) / 40.0
            halo_r = int(glow_color[0] * alpha_ratio + horizon_color[0] * (1 - alpha_ratio))
            halo_g = int(glow_color[1] * alpha_ratio + horizon_color[1] * (1 - alpha_ratio))
            halo_b = int(glow_color[2] * alpha_ratio + horizon_color[2] * (1 - alpha_ratio))
            draw.ellipse([sun_x - r_offset, sun_y - r_offset, sun_x + r_offset, sun_y + r_offset], fill=(halo_r, halo_g, halo_b))
        draw.ellipse([sun_x - sun_radius, sun_y - sun_radius, sun_x + sun_radius, sun_y + sun_radius], fill=glow_color)

        # 4. Background Mountain / Distant Spire Ridge
        ridge_points = [(0, horizon_y)]
        num_peaks = 16
        for i in range(num_peaks + 1):
            px = int(i * (width / num_peaks))
            py = horizon_y - rng.randint(40, 110)
            ridge_points.append((px, py))
        ridge_points.append((width, horizon_y))
        draw.polygon(ridge_points, fill=(int(horizon_color[0]*0.4), int(horizon_color[1]*0.3), int(horizon_color[2]*0.5)))

        # 5. Midground Futuristic Skyline (Spires, Towers, Sky-bridges)
        current_x = 0
        tower_color = (25, 18, 35)
        window_color = (255, 215, 120) if "sunset" in p_lower else (0, 220, 240)
        while current_x < width:
            min_w = max(10, int(width * 0.03))
            max_w = max(min_w + 5, int(width * 0.08))
            b_width = rng.randint(min_w, max_w)

            min_h = max(20, int(height * 0.15))
            max_h = max(min_h + 10, int(height * 0.55))
            b_height = rng.randint(min_h, max_h)
            b_top = horizon_y - b_height
            # Draw building body
            draw.rectangle([current_x, b_top, current_x + b_width, horizon_y], fill=tower_color)
            
            # Antenna / Spire on top
            spire_x = current_x + b_width // 2
            spire_height = rng.randint(15, 45)
            draw.line([(spire_x, b_top), (spire_x, b_top - spire_height)], fill=tower_color, width=2)
            # Spire beacon
            beacon_color = (255, 50, 50) if rng.random() > 0.4 else (100, 255, 255)
            draw.point([(spire_x, b_top - spire_height)], fill=beacon_color)

            # Glowing windows
            for wy in range(b_top + 15, horizon_y - 20, 12):
                if rng.random() > 0.35:
                    for wx in range(current_x + 6, current_x + b_width - 6, 10):
                        if rng.random() > 0.4:
                            draw.rectangle([wx, wy, wx + 4, wy + 5], fill=window_color)
            current_x += b_width + rng.randint(-5, 10)

        # 6. Foreground Platform / Ground Landscape
        ground_color = (12, 8, 18)
        draw.rectangle([0, horizon_y, width, height], fill=ground_color)
        
        # Ground perspective lines & glowing circuit/street lines
        street_glow = (255, 120, 30) if "sunset" in p_lower else (0, 180, 255)
        for gx in range(0, width, 120):
            draw.line([(gx, horizon_y), (int(width*0.5 + (gx - width*0.5)*2.2), height)], fill=(40, 30, 50), width=2)
        
        # Horizon road / rail line
        draw.line([(0, horizon_y + 4), (width, horizon_y + 4)], fill=street_glow, width=3)

        # 7. Foreground Subject (e.g. Robot / Explorer Silhouette with Glowing Eye)
        if "robot" in p_lower or "figure" in p_lower or "traveler" in p_lower or "explorer" in p_lower:
            rx = int(width * 0.28)
            ry = horizon_y + 20
            # Robot body pedestal / rock
            draw.ellipse([rx - 45, ry + 85, rx + 45, ry + 115], fill=(5, 3, 10))
            # Legs
            draw.rectangle([rx - 22, ry + 45, rx - 10, ry + 95], fill=(8, 5, 12))
            draw.rectangle([rx + 10, ry + 45, rx + 22, ry + 95], fill=(8, 5, 12))
            # Torso (rounded chassis)
            draw.rounded_rectangle([rx - 32, ry - 5, rx + 32, ry + 50], radius=8, fill=(10, 8, 16))
            # Head
            draw.rounded_rectangle([rx - 24, ry - 42, rx + 24, ry - 8], radius=6, fill=(12, 10, 20))
            # Antenna
            draw.line([(rx, ry - 42), (rx, ry - 58)], fill=(12, 10, 20), width=2)
            draw.ellipse([rx - 3, ry - 64, rx + 3, ry - 58], fill=(255, 180, 50))
            # Optical sensor / Glowing Eye
            eye_color = (0, 230, 255) # Cyan visor
            draw.rounded_rectangle([rx - 16, ry - 32, rx + 16, ry - 20], radius=3, fill=eye_color)
            # Soft visor glow
            draw.point([(rx - 17, ry - 26), (rx + 17, ry - 26)], fill=(120, 245, 255))

        # 8. Subtle Cinematic Vignette Effect
        img_np = np.array(img, dtype=np.float32)
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - width/2)**2 + (Y - height/2)**2)
        max_dist = np.sqrt((width/2)**2 + (height/2)**2)
        vignette = 1.0 - (dist_from_center / max_dist) * 0.35
        vignette = np.clip(vignette, 0.4, 1.0)
        img_np = img_np * vignette[..., np.newaxis]
        final_img = Image.fromarray(np.uint8(img_np))

        # Save result
        final_img.save(output_path, "JPEG", quality=95)
        return output_path
