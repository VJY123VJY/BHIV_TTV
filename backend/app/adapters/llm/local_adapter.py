import math
import re
import random
from typing import Dict, Any, List, Optional
from app.adapters.base import BaseLLMAdapter
from app.models.scene import Scene
from app.core.logging import telemetry, logger

class LocalLLMAdapter(BaseLLMAdapter):
    """
    Intelligent dynamic story, script, and scene generator.
    Dynamically extracts subjects, settings, actions, and atmospheric elements from the exact user prompt.
    """
    async def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        p_clean = prompt.strip()
        p_lower = p_clean.lower()
        
        # 1. Subject extraction
        subject_patterns = [
            # Animals
            ("dog", ["dog", "puppy", "hound", "retriever", "canine", "shepherd", "husky", "bulldog"]),
            ("cat", ["cat", "kitten", "feline"]),
            ("horse", ["horse", "stallion", "mare", "steed"]),
            ("lion", ["lion", "lioness"]),
            ("tiger", ["tiger"]),
            ("wolf", ["wolf", "wolves"]),
            ("bear", ["bear"]),
            ("bird", ["bird", "eagle", "hawk", "falcon", "seagull", "crow"]),
            ("whale", ["whale", "dolphin"]),
            ("shark", ["shark"]),
            # Sci-fi / Tech
            ("astronaut", ["astronaut", "cosmonaut", "spacewalker"]),
            ("robot", ["robot", "android", "cyborg", "mech", "drone", "automaton", "rover"]),
            ("alien", ["alien", "extraterrestrial"]),
            ("scientist", ["scientist", "researcher", "inventor"]),
            # Humans / Roles
            ("child", ["child", "kid", "boy", "girl"]),
            ("warrior", ["warrior", "knight", "soldier", "samurai", "ninja"]),
            ("wizard", ["wizard", "mage", "sorcerer"]),
            ("dancer", ["dancer", "ballerina"]),
            ("explorer", ["explorer", "wanderer", "traveler", "hiker", "traveler", "person", "man", "woman"]),
            # Vehicles
            ("car", ["car", "sports car", "racecar", "vehicle", "automobile", "ferrari", "supercar", "truck"]),
            ("motorcycle", ["motorcycle", "bike", "motorbike"]),
            ("ship", ["ship", "boat", "sailboat", "yacht", "vessel", "pirate ship", "submarine"]),
            ("airplane", ["airplane", "plane", "jet", "aircraft"]),
            ("spaceship", ["spaceship", "starship", "rocket", "shuttle"]),
            # Mythical
            ("dragon", ["dragon", "wyvern", "phoenix", "monster"])
        ]
        
        detected_subject = None
        for category, synonyms in subject_patterns:
            for syn in synonyms:
                if re.search(r"\b" + re.escape(syn) + r"\b", p_lower):
                    detected_subject = syn
                    break
            if detected_subject:
                break

        if not detected_subject:
            # Extract first noun phrase after 'a', 'an', or 'the'
            match = re.search(r"\b(?:a|an|the)\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:running|walking|flying|sailing|swimming|driving|exploring|standing|in|on|at|through|near)", p_lower)
            if match:
                detected_subject = match.group(1).strip()
            else:
                words = [w for w in re.findall(r"\b[a-z]{3,}\b", p_lower) if w not in ["the", "and", "with", "from", "over", "into", "under", "about", "scene", "video", "shot", "like"]]
                detected_subject = words[0] if words else "subject"

        # 2. Action extraction
        action_patterns = [
            ("running", ["running", "runs", "sprinting", "jogging", "bounding", "chasing", "dashing"]),
            ("walking", ["walking", "walks", "strolling", "striding", "wandering", "marching"]),
            ("flying", ["flying", "flies", "soaring", "gliding", "hovering"]),
            ("sailing", ["sailing", "sails", "cruising", "floating", "drifting", "navigating"]),
            ("driving", ["driving", "drives", "racing", "speeding", "cruising"]),
            ("swimming", ["swimming", "swims", "diving"]),
            ("exploring", ["exploring", "explores", "investigating", "searching", "surveying"]),
            ("standing", ["standing", "stands", "resting", "observing", "gazing", "looking"]),
            ("dancing", ["dancing", "dances", "leaping"]),
        ]
        detected_action = None
        for act, synonyms in action_patterns:
            for syn in synonyms:
                if re.search(r"\b" + re.escape(syn) + r"\b", p_lower):
                    detected_action = act
                    break
            if detected_action:
                break
        if not detected_action:
            detected_action = "exploring"

        # 3. Setting / Environment extraction
        setting_patterns = [
            ("beach", ["beach", "shore", "coast", "shoreline", "sand", "seaside", "ocean shore"]),
            ("mars", ["mars", "martian", "red planet"]),
            ("moon", ["moon", "lunar"]),
            ("ocean", ["ocean", "sea", "underwater", "reef", "waves", "water"]),
            ("space", ["space", "cosmos", "galaxy", "orbit", "nebula", "deep space"]),
            ("desert", ["desert", "dunes", "sahara", "wasteland"]),
            ("forest", ["forest", "woods", "jungle", "rainforest", "grove", "trees"]),
            ("mountains", ["mountains", "mountain", "peak", "alps", "cliff", "canyon", "valley"]),
            ("city", ["city", "metropolis", "downtown", "streets", "avenue", "urban"]),
            ("cyberpunk", ["cyberpunk", "futuristic city", "neon city", "sci-fi city"]),
            ("laboratory", ["laboratory", "lab", "facility", "station"]),
            ("snow", ["snow", "winter", "ice", "arctic", "glacier", "blizzard"]),
            ("meadow", ["meadow", "field", "grassland", "plains", "prairie"]),
            ("racetrack", ["racetrack", "track", "highway", "road", "speedway"])
        ]
        detected_setting = None
        for env, synonyms in setting_patterns:
            for syn in synonyms:
                if re.search(r"\b" + re.escape(syn) + r"\b", p_lower):
                    detected_setting = env
                    break
            if detected_setting:
                break

        if not detected_setting:
            # Look for location prepositions: in/on/at/through <location>
            match = re.search(r"\b(?:in|on|at|through|across|over|near|along)\s+(?:a|an|the)?\s*([a-z]+(?:\s+[a-z]+)?)\b", p_lower)
            if match:
                detected_setting = match.group(1).strip()
            else:
                detected_setting = "open world"

        # 4. Lighting & Time of Day
        if "sunset" in p_lower or "golden hour" in p_lower:
            lighting = "radiant golden hour sunset with rich amber, orange, and purple hues"
            time_of_day = "sunset"
        elif "sunrise" in p_lower or "dawn" in p_lower:
            lighting = "early morning dawn with soft golden sunlight and gentle pastel mist"
            time_of_day = "sunrise"
        elif "night" in p_lower or "midnight" in p_lower:
            lighting = "deep night with luminous moonlight, shadows, and glowing accent lights"
            time_of_day = "night"
        elif "storm" in p_lower or "rain" in p_lower:
            lighting = "dramatic stormy overcast with lightning flares, wet reflections, and dark thunderclouds"
            time_of_day = "stormy"
        elif detected_setting == "mars":
            lighting = "atmospheric Martian reddish-amber daylight with subtle pink haze"
            time_of_day = "martian_day"
        elif detected_setting == "beach":
            lighting = "brilliant tropical sunlight shimmering across white water and golden sand"
            time_of_day = "daylight"
        elif detected_setting == "cyberpunk":
            lighting = "neon illumination glowing with cyan and magenta light reflections"
            time_of_day = "neon_night"
        elif detected_setting == "space":
            lighting = "high-contrast stellar illumination with glowing celestial nebulae"
            time_of_day = "cosmic"
        else:
            lighting = "cinematic natural daylight with sharp depth of field and soft ambient shadows"
            time_of_day = "daylight"

        # 5. Mood extraction
        if "fast" in p_lower or "speed" in p_lower or "racing" in p_lower:
            mood = "high-speed exhilaration and dynamic adrenaline"
        elif "peaceful" in p_lower or "calm" in p_lower or "quiet" in p_lower:
            mood = "serene tranquility and peaceful beauty"
        elif "mysterious" in p_lower or "dark" in p_lower:
            mood = "suspenseful intrigue and mysterious discovery"
        elif "joy" in p_lower or "happy" in p_lower or detected_subject == "dog":
            mood = "vibrant freedom and exhilarating joy"
        else:
            mood = "cinematic wonder and awe"

        analysis_result = {
            "raw_prompt": prompt,
            "subject": detected_subject,
            "action": detected_action,
            "setting": detected_setting,
            "time_of_day": time_of_day,
            "lighting": lighting,
            "mood": mood,
            "keywords": [w for w in re.findall(r"\b[a-z]{4,}\b", p_lower) if w not in ["this", "that", "with", "from", "video", "scene"]]
        }
        logger.info(f"Prompt Analyzed: subject='{detected_subject}', action='{detected_action}', setting='{detected_setting}', lighting='{lighting}'")
        return analysis_result

    async def generate_story(self, prompt: str, analysis: Dict[str, Any], duration: int) -> Dict[str, Any]:
        subject = analysis["subject"]
        action = analysis["action"]
        setting = analysis["setting"]
        lighting = analysis["lighting"]
        mood = analysis["mood"]

        # Generate unique tailored 3-act narrative strictly for this subject & setting
        if setting in ["beach", "ocean", "coast"]:
            act1_beat = f"A {subject} {action} along the sunlit shore, leaving tracks across the glistening wet sand."
            act1_narr = f"Under the warm sky, a {subject} {action} along the wide open coastline as gentle waves wash ashore."
            act2_beat = f"Splashing through sparkling sea foam, the {subject} gathers momentum along the water's edge."
            act2_narr = f"Every leap sends sprays of crystal seawater catching the brilliant light, full of energy and movement."
            act3_beat = f"The {subject} pauses on a scenic sand ridge, gazing out across the vast ocean horizon."
            act3_narr = f"Looking out across the endless sea, the journey captures a moment of pure freedom and boundless beauty."
        elif setting in ["mars", "red planet", "moon", "space"]:
            act1_beat = f"A solitary {subject} {action} across the rugged, crimson Martian terrain under a dusky sky."
            act1_narr = f"On a distant, quiet world, a {subject} takes historic steps across the rust-red soil of Mars."
            act2_beat = f"Navigating ancient impact ridges, the {subject} surveys the desolate beauty of towering canyons."
            act2_narr = f"Dust particles swirl in the thin atmosphere as the {subject} presses forward into uncharted territory."
            act3_beat = f"Standing upon a towering plateau, the {subject} gazes out over the endless Martian horizon."
            act3_narr = f"Beneath the distant sun, the vast planetary expanse stretches into eternity, breathtaking and profound."
        elif setting in ["cyberpunk", "city", "metropolis", "racetrack"]:
            act1_beat = f"A sleek {subject} {action} through the heart of the bustling metropolis under {lighting}."
            act1_narr = f"Amidst towering architecture, the {subject} bursts into motion as lights reflect across the scene."
            act2_beat = f"Carving through dynamic avenues, the {subject} navigates shifting currents of motion and energy."
            act2_narr = f"Power and precision intertwine as every turn reveals new angles of the breathtaking urban landscape."
            act3_beat = f"Surging towards the illuminated horizon, the {subject} claims the expanse of the open thoroughfare."
            act3_narr = f"Light and speed converge into a stunning panorama of modern cinematic grandeur."
        elif setting in ["forest", "jungle", "meadow", "nature"]:
            act1_beat = f"A {subject} {action} through the lush greenery, surrounded by towering ancient trees."
            act1_narr = f"In the heart of the vibrant wilderness, a {subject} moves gracefully through sun-dappled paths."
            act2_beat = f"Crossing clear stream waters, the {subject} weaves between moss-covered boulders and vibrant flora."
            act2_narr = f"The living forest hums with vitality as every stride brings deeper harmony with nature."
            act3_beat = f"Emerging into a sun-drenched clearing, the {subject} pauses in majestic serenity."
            act3_narr = f"Bathed in warm golden sunlight, the natural world reveals its peaceful perfection."
        else:
            # Fully customized dynamic storyline
            act1_beat = f"The {subject} begins {action} across the vibrant landscape of {setting}."
            act1_narr = f"The journey unfolds as a {subject} moves across {setting}, illuminated by {lighting}."
            act2_beat = f"Continuing forward, the {subject} dynamically traverses the terrain with growing momentum."
            act2_narr = f"Surrounded by the unique atmosphere of {setting}, every moment reveals rich visual detail."
            act3_beat = f"Reaching a scenic vantage point, the {subject} overlooks the breathtaking expanse."
            act3_narr = f"Against the scenic vista of {setting}, the sequence concludes with cinematic majesty."

        story = {
            "title": f"The Journey of the {subject.title()} on {setting.title()}",
            "premise": f"In {setting}, a {subject} {action} under {lighting}, capturing {mood}.",
            "acts": [
                {
                    "act": 1,
                    "title": f"Beginning the Traverse",
                    "beat": act1_beat,
                    "narrative": act1_narr
                },
                {
                    "act": 2,
                    "title": f"Into the Flow",
                    "beat": act2_beat,
                    "narrative": act2_narr
                },
                {
                    "act": 3,
                    "title": f"The Scenic Horizon",
                    "beat": act3_beat,
                    "narrative": act3_narr
                }
            ]
        }
        return story

    async def generate_scenes(self, story: Dict[str, Any], target_duration: int, style: str) -> List[Scene]:
        acts = story.get("acts", [])
        num_scenes = max(3, min(5, math.ceil(target_duration / 6)))
        scene_duration = round(target_duration / num_scenes, 2)

        camera_motions = ["pan_right", "zoom_in", "pan_left", "tilt_up", "zoom_out"]
        scenes: List[Scene] = []

        for i in range(num_scenes):
            act_idx = min(i, len(acts) - 1)
            act = acts[act_idx]
            cam_motion = camera_motions[i % len(camera_motions)]
            
            visual_desc = (
                f"{style.capitalize()} photography. {act['beat']} "
                f"Camera movement: {cam_motion.replace('_', ' ')}. Highly detailed, authentic environment, high dynamic range."
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
