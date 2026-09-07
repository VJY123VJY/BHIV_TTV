import math
import re
from typing import Dict, Any, List
from app.adapters.base import BaseLLMAdapter
from app.models.scene import Scene
from app.core.logging import telemetry

class LocalLLMAdapter(BaseLLMAdapter):
    """
    High-quality offline deterministic story and scene generator.
    Implements rule-based cinematic scene composition derived from ttv_converegence.
    """
    async def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        p_lower = prompt.lower()
        
        # Detect primary subject / character
        subjects = ["robot", "scientist", "astronaut", "dragon", "warrior", "child", "traveler", "cat", "car", "bird", "wanderer"]
        subject = "traveler"
        for s in subjects:
            if s in p_lower:
                subject = s
                break

        # Detect setting / environment
        settings = ["futuristic city", "city", "mountains", "forest", "desert", "planet", "ocean", "cyberpunk alley", "space station", "ancient ruins"]
        setting = "futuristic metropolis"
        for s in settings:
            if s in p_lower:
                setting = s
                break

        # Detect time of day / lighting
        lighting = "sunset, golden hour, warm orange and violet glow"
        if "sunset" in p_lower:
            lighting = "sunset, radiant warm amber and purple sky"
        elif "night" in p_lower or "midnight" in p_lower:
            lighting = "neon-lit night, dark indigo shadows and glowing highlights"
        elif "morning" in p_lower or "dawn" in p_lower:
            lighting = "misty morning dawn, soft pastel sunlight"

        # Detect mood
        mood = "wonder and exploration"
        if "mysterious" in p_lower or "danger" in p_lower:
            mood = "suspense and mystery"
        elif "peaceful" in p_lower or "calm" in p_lower:
            mood = "serenity and peaceful contemplation"

        return {
            "raw_prompt": prompt,
            "subject": subject,
            "setting": setting,
            "lighting": lighting,
            "mood": mood,
            "keywords": [w for w in re.findall(r"\w+", p_lower) if len(w) > 3][:8]
        }

    async def generate_story(self, prompt: str, analysis: Dict[str, Any], duration: int) -> Dict[str, Any]:
        subject = analysis["subject"]
        setting = analysis["setting"]
        mood = analysis["mood"]
        lighting = analysis["lighting"]

        story = {
            "title": f"Chronicles of {setting.title()}",
            "premise": f"In a breathtaking {setting}, a lone {subject} embarks on an unexpected journey under {lighting}.",
            "acts": [
                {
                    "act": 1,
                    "title": "The Awakening Horizon",
                    "beat": f"The {subject} stands quietly against the vast backdrop of {setting}, watching light dance across the towers.",
                    "narrative": f"As daylight wanes over the {setting}, a lone {subject} pauses to witness the glowing horizon."
                },
                {
                    "act": 2,
                    "title": "Through the Shining Pathways",
                    "beat": f"Venturing deeper, the {subject} explores towering structures bathed in the warm amber twilight.",
                    "narrative": f"Every step unveils ancient secrets intertwined with soaring spires, alive with quiet energy."
                },
                {
                    "act": 3,
                    "title": "The Grand Discovery",
                    "beat": f"Reaching the summit, the {subject} gazes out over the endless neon expanse at twilight.",
                    "narrative": f"Underneath the fading sun, the true wonder of this world reveals itself in all its majesty."
                }
            ]
        }
        return story

    async def generate_scenes(self, story: Dict[str, Any], target_duration: int, style: str) -> List[Scene]:
        acts = story.get("acts", [])
        # Determine number of scenes (3 to 5 based on duration)
        num_scenes = max(3, min(5, math.ceil(target_duration / 6)))
        scene_duration = round(target_duration / num_scenes, 2)

        camera_motions = ["pan_right", "zoom_in", "pan_left", "tilt_up", "zoom_out"]
        scenes: List[Scene] = []

        for i in range(num_scenes):
            act_idx = min(i, len(acts) - 1)
            act = acts[act_idx]
            cam_motion = camera_motions[i % len(camera_motions)]
            
            # Format visual description
            visual_desc = (
                f"{style.capitalize()} style establishing shot. {act['beat']}. "
                f"Camera movement: {cam_motion.replace('_', ' ')}. Volumetric lighting, 8k resolution, photorealistic cinematic framing."
            )

            scene = Scene(
                index=i + 1,
                title=f"Scene {i+1}: {act['title']}",
                narrative=act["narrative"],
                visual_description=visual_desc,
                duration=scene_duration,
                camera_motion=cam_motion,
                metadata={"style": style, "act": act_idx + 1}
            )
            scenes.append(scene)

        return scenes
