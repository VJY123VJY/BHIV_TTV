import os
import math
import random
import cv2
import numpy as np
from app.adapters.base import BaseVideoAdapter
from app.models.scene import Scene
from app.core.exceptions import VisualGenerationError

class OpenCVVideoAdapter(BaseVideoAdapter):
    """
    Cinematic Video Synthesis Engine.
    Extends and unifies the keyframe motion and rendering concepts from text_to_video2
    and ttv_converegence into a robust, high-resolution kinematic video generator.
    """
    async def generate_scene_video(self, scene: Scene, keyframe_path: str, output_path: str, width: int = 1280, height: int = 720, fps: int = 24) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not os.path.exists(keyframe_path):
            raise VisualGenerationError(f"Keyframe image missing: {keyframe_path}")

        img = cv2.imread(keyframe_path)
        if img is None:
            raise VisualGenerationError(f"Could not load keyframe image: {keyframe_path}")

        # Scale keyframe to 1.2x size to allow headroom for panning and zooming
        pad_factor = 1.20
        large_w = int(width * pad_factor)
        large_h = int(height * pad_factor)
        large_img = cv2.resize(img, (large_w, large_h), interpolation=cv2.INTER_LANCZOS4)

        total_frames = max(12, int(scene.duration * fps))
        motion = scene.camera_motion or "pan_right"

        # Try mp4v fourcc
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

        if not out.isOpened():
            # Fallback fourcc
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
            if not out.isOpened():
                raise VisualGenerationError(f"Failed to initialize VideoWriter for {output_path}")

        max_x_offset = large_w - width
        max_y_offset = large_h - height

        # Dynamic particle seed for subtle atmospheric particles based on scene seed or random
        scene_seed = scene.metadata.get("seed") if scene.metadata else None
        if scene_seed is None:
            scene_seed = random.randint(1, 2**31 - 1)
        rng = random.Random(scene_seed + scene.index * 73)

        desc_lower = (f"{scene.visual_description} {scene.narrative}").lower()
        is_rain = "rain" in desc_lower
        if is_rain:
            particle_color = (240, 210, 150) # Rain streaks (BGR)
            num_particles = 70
        elif any(k in desc_lower for k in ["mars", "desert", "dust", "sand"]):
            particle_color = (60, 110, 210) # Rust-red / golden dust
            num_particles = 35
        elif any(k in desc_lower for k in ["beach", "ocean", "sea", "water", "shore"]):
            particle_color = (255, 245, 230) # White sea foam mist
            num_particles = 25
        elif any(k in desc_lower for k in ["space", "cosmic", "star", "galaxy"]):
            particle_color = (255, 255, 255) # Brilliant stars
            num_particles = 40
        else:
            particle_color = (200, 225, 240) # Natural atmospheric motes
            num_particles = 30

        particles = [
            {
                "x": rng.randint(0, width),
                "y": rng.randint(0, height),
                "speed": rng.uniform(2.5, 6.0) if is_rain else rng.uniform(0.5, 2.0),
                "size": rng.randint(1, 2)
            }
            for _ in range(num_particles)
        ]

        for t in range(total_frames):
            # Smooth cosine ease-in-out progress
            progress = t / float(max(1, total_frames - 1))
            ease = 0.5 * (1.0 - math.cos(progress * math.pi))

            if motion == "pan_right":
                x_off = int(ease * max_x_offset)
                y_off = int(max_y_offset * 0.5)
            elif motion == "pan_left":
                x_off = int((1.0 - ease) * max_x_offset)
                y_off = int(max_y_offset * 0.5)
            elif motion == "zoom_in":
                # Crop area shrinks as zoom increases
                curr_w = int(large_w - ease * (large_w - width))
                curr_h = int(large_h - ease * (large_h - height))
                cx = large_w // 2
                cy = large_h // 2
                x_off = cx - curr_w // 2
                y_off = cy - curr_h // 2
                cropped = large_img[y_off:y_off + curr_h, x_off:x_off + curr_w]
                frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            elif motion == "zoom_out":
                curr_w = int(width + ease * (large_w - width))
                curr_h = int(height + ease * (large_h - height))
                cx = large_w // 2
                cy = large_h // 2
                x_off = cx - curr_w // 2
                y_off = cy - curr_h // 2
                cropped = large_img[y_off:y_off + curr_h, x_off:x_off + curr_w]
                frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            elif motion == "tilt_up":
                x_off = int(max_x_offset * 0.5)
                y_off = int((1.0 - ease) * max_y_offset)
            else: # tilt_down or default
                x_off = int(max_x_offset * 0.5)
                y_off = int(ease * max_y_offset)

            if motion not in ["zoom_in", "zoom_out"]:
                frame = large_img[y_off:y_off + height, x_off:x_off + width].copy()

            # Add dynamic atmospheric particles / rain drifting
            for p in particles:
                if is_rain:
                    p["y"] += p["speed"]
                    p["x"] -= p["speed"] * 0.25
                    if p["y"] > height:
                        p["y"] = 0
                        p["x"] = rng.randint(0, width)
                    px = int(p["x"])
                    py = int(p["y"])
                    if 0 <= px < width and 0 <= py < height:
                        cv2.line(frame, (px, py), (px - 2, min(height - 1, py + 8)), particle_color, 1)
                else:
                    p["y"] -= p["speed"]
                    if p["y"] < 0:
                        p["y"] = height
                        p["x"] = rng.randint(0, width)
                    px = int(p["x"])
                    py = int(p["y"])
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(frame, (px, py), p["size"], particle_color, -1)

            out.write(frame)

        out.release()
        return output_path
