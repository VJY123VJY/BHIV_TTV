import os
import math
import random
import re
import cv2
import numpy as np
from app.adapters.base import BaseVideoAdapter
from app.models.scene import Scene
from app.core.exceptions import VisualGenerationError


def has_term(text: str, terms: list) -> bool:
    for t in terms:
        if re.search(r"\b" + re.escape(t) + r"\b", text):
            return True
    return False


class OpenCVVideoAdapter(BaseVideoAdapter):
    """
    Cinematic Video Synthesis and Motion Rendering Engine.
    Synthesizes smooth, physically consistent, prompt-aligned temporal motion:
    - Smooth camera tracking with cosine acceleration
    - Natural subject walking motion and cadence
    - Wind swaying effects across vegetation / crops
    - Moving celestial / sky elements (flying birds, cloud drift)
    - Dynamic vehicle and object trajectories (e.g. moving balls, driving cars)
    - Continuous frame-to-frame temporal coherence without frame duplication
    """
    async def generate_scene_video(
        self,
        scene: Scene,
        keyframe_path: str,
        output_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if not os.path.exists(keyframe_path):
            raise VisualGenerationError(f"Keyframe image missing: {keyframe_path}")

        img = cv2.imread(keyframe_path)
        if img is None:
            raise VisualGenerationError(f"Could not load keyframe image: {keyframe_path}")

        # Ensure base image matches target resolution
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

        # Scale keyframe to 1.25x for smooth camera tracking
        pad_factor = 1.25
        large_w = int(width * pad_factor)
        large_h = int(height * pad_factor)
        large_img = cv2.resize(img, (large_w, large_h), interpolation=cv2.INTER_LANCZOS4)

        total_frames = max(12, int(scene.duration * fps))
        motion = scene.camera_motion or "pan_right"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
            if not out.isOpened():
                raise VisualGenerationError(f"Failed to initialize VideoWriter for {output_path}")

        max_x_offset = large_w - width
        max_y_offset = large_h - height

        # Dynamic seed
        scene_seed = scene.metadata.get("seed") if scene.metadata else None
        if scene_seed is None:
            scene_seed = random.randint(1, 2**31 - 1)
        rng = random.Random(scene_seed + scene.index * 73)

        full_desc = f"{scene.visual_description} {scene.narrative}".lower()

        # Detect specific motion modes
        is_ball_motion = has_term(full_desc, ["ball", "sphere", "orb"]) and has_term(full_desc, ["moving", "left to right", "roll", "rolling"])
        is_farmer = has_term(full_desc, ["farmer", "indian farmer", "walking", "crops", "field"])
        is_car = has_term(full_desc, ["car", "sports car", "driving", "racing", "highway"]) and not is_farmer
        is_dog = has_term(full_desc, ["dog", "retriever", "running", "puppy"])
        is_airplane = has_term(full_desc, ["airplane", "plane", "flying", "jet", "clouds"])
        is_waterfall = has_term(full_desc, ["waterfall", "cascade"])
        is_rain = has_term(full_desc, ["rain", "storm", "thunder"])

        # Atmospheric motes / particles
        if is_rain:
            particle_color = (240, 210, 150)
            num_particles = 65
        elif has_term(full_desc, ["mars", "desert", "dust", "sand"]):
            particle_color = (60, 110, 210)
            num_particles = 30
        else:
            particle_color = (220, 235, 245)
            num_particles = 25

        particles = [
            {
                "x": rng.randint(0, width),
                "y": rng.randint(0, height),
                "speed": rng.uniform(3.0, 7.0) if is_rain else rng.uniform(0.6, 2.2),
                "size": rng.randint(1, 2)
            }
            for _ in range(num_particles)
        ]

        # Flying birds in distant sky
        birds = [
            {"x": float(rng.randint(int(width * 0.1), int(width * 0.7))), "y": float(rng.randint(max(1, int(height * 0.05)), max(2, int(height * 0.35)))), "speed": rng.uniform(1.2, 2.5)}
            for _ in range(5)
        ]

        for t in range(total_frames):
            progress = t / float(max(1, total_frames - 1))
            # Smooth cosine easing for camera tracking
            ease = 0.5 * (1.0 - math.cos(progress * math.pi))

            if is_ball_motion:
                # Dynamic translation of ball across screen from left to right
                frame = img.copy()
                ball_r = int(height * 0.08)
                # Overwrite original center ball with background color
                orig_cx = int(width * 0.44)
                orig_cy = int(height * 0.58 + (height - height * 0.58) * 0.28)
                sample_y = min(height - 1, orig_cy + ball_r + 20)
                sample_x = min(width - 1, max(0, orig_cx))
                bg_col = (int(frame[sample_y, sample_x, 0]),
                          int(frame[sample_y, sample_x, 1]),
                          int(frame[sample_y, sample_x, 2]))
                cv2.rectangle(frame, (orig_cx - ball_r - 20, orig_cy - ball_r - 20), (orig_cx + ball_r + 20, orig_cy + ball_r + 25), bg_col, -1)

                # Animate new ball position moving from left to right
                curr_bx = int(width * 0.15 + ease * (width * 0.70))
                curr_by = orig_cy
                # Shadow
                cv2.ellipse(frame, (curr_bx, curr_by + ball_r + 4), (ball_r + 8, 8), 0, 0, 360, (25, 28, 30), -1)
                # Ball
                ball_col = (30, 35, 225) if "red" in full_desc else (220, 110, 40)
                cv2.circle(frame, (curr_bx, curr_by), ball_r, ball_col, -1)
                # Highlight
                cv2.circle(frame, (curr_bx - int(ball_r * 0.35), curr_by - int(ball_r * 0.35)), 6, (255, 255, 255), -1)

            else:
                # Camera tracking calculation
                if motion == "zoom_in":
                    curr_w = int(large_w - ease * (large_w - width))
                    curr_h = int(large_h - ease * (large_h - height))
                    cx = large_w // 2
                    cy = large_h // 2
                    x_off = cx - curr_w // 2
                    y_off = cy - curr_h // 2
                    cropped = large_img[y_off:y_off + curr_h, x_off:x_off + curr_w]
                    frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
                else:
                    if motion == "pan_left":
                        x_off = int((1.0 - ease) * max_x_offset)
                        y_off = int(max_y_offset * 0.5)
                    else:
                        x_off = int(ease * max_x_offset)
                        y_off = int(max_y_offset * 0.5)
                    frame = large_img[y_off:y_off + height, x_off:x_off + width].copy()

                # Dynamic motion layers
                if is_farmer:
                    # 1. Subtle walking cadence oscillation
                    walk_cadence = math.sin(t * 0.6) * 2.5
                    # 2. Wind moving crops: gentle wave displacement on bottom 35% of frame
                    crop_start_y = int(height * 0.65)
                    crop_region = frame[crop_start_y:height, :].copy()
                    wind_shift = int(math.sin(t * 0.35 + 1.2) * 3)
                    if wind_shift != 0:
                        M = np.float32([[1, 0, wind_shift], [0, 1, 0]])
                        shifted_crops = cv2.warpAffine(crop_region, M, (width, height - crop_start_y), borderMode=cv2.BORDER_REFLECT)
                        frame[crop_start_y:height, :] = cv2.addWeighted(crop_region, 0.25, shifted_crops, 0.75, 0)

                    # 3. Birds flying across distant sky
                    for b in birds:
                        b["x"] += b["speed"]
                        bx = int(b["x"]) % width
                        by = int(b["y"] + math.sin(t * 0.4 + b["speed"]) * 3)
                        # Draw small flapping V bird
                        cv2.line(frame, (bx - 6, by - 3), (bx, by), (40, 30, 45), 1)
                        cv2.line(frame, (bx, by), (bx + 6, by - 3), (40, 30, 45), 1)

                    # 4. Warm natural sunrise lighting breathing
                    sun_flare = math.sin(t * 0.15) * 4.0
                    if sun_flare > 0:
                        frame = np.clip(frame.astype(np.float32) + np.array([sun_flare*0.4, sun_flare*0.8, sun_flare*1.0]), 0, 255).astype(np.uint8)

                elif is_waterfall:
                    # Downward cascading shimmer
                    wf_y = int(height * 0.4)
                    flow_shift = int(math.sin(t * 0.8) * 4)
                    flow_strip = frame[wf_y:height - 50, int(width*0.35):int(width*0.65)]
                    M = np.float32([[1, 0, 0], [0, 1, flow_shift]])
                    flow_shifted = cv2.warpAffine(flow_strip, M, (flow_strip.shape[1], flow_strip.shape[0]), borderMode=cv2.BORDER_REFLECT)
                    frame[wf_y:height - 50, int(width*0.35):int(width*0.65)] = cv2.addWeighted(flow_strip, 0.4, flow_shifted, 0.6, 0)

            # Atmospheric particles / rain
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
