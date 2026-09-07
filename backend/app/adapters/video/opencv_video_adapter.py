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

        # Particle seed for subtle atmospheric particles
        rng = random.Random(abs(hash(scene.title)) % 5000)
        particles = [{"x": rng.randint(0, width), "y": rng.randint(0, height), "speed": rng.uniform(0.5, 2.0), "size": rng.randint(1, 2)} for _ in range(30)]

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

            # Add subtle atmospheric dust particles drifting
            for p in particles:
                p["y"] -= p["speed"]
                if p["y"] < 0:
                    p["y"] = height
                    p["x"] = rng.randint(0, width)
                px = int(p["x"])
                py = int(p["y"])
                if 0 <= px < width and 0 <= py < height:
                    cv2.circle(frame, (px, py), p["size"], (255, 220, 180), -1)

            out.write(frame)

        out.release()
        return output_path
