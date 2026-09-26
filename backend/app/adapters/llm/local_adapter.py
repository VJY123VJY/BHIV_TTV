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
            # Farmers & Humans
            ("farmer", ["indian farmer", "farmer", "peasant", "cultivator", "harvester", "agriculture worker", "villager"]),
            ("chef", ["chef", "cook", "baker"]),
            ("child", ["child", "kid", "boy", "girl"]),
            ("warrior", ["warrior", "knight", "soldier", "samurai", "ninja"]),
            ("wizard", ["wizard", "mage", "sorcerer"]),
            ("dancer", ["dancer", "ballerina"]),
            ("explorer", ["explorer", "wanderer", "traveler", "hiker", "person", "man", "woman"]),
            ("scientist", ["scientist", "researcher", "inventor"]),
            ("astronaut", ["astronaut", "cosmonaut", "spacewalker"]),
            # Animals
            ("dog", ["golden retriever", "retriever", "dog", "puppy", "hound", "canine", "shepherd", "husky", "wolf"]),
            ("cat", ["cat", "kitten", "feline"]),
            ("elephant", ["elephant", "elephants"]),
            ("horse", ["horse", "stallion", "mare", "steed"]),
            ("lion", ["lion", "lioness"]),
            ("tiger", ["tiger"]),
            ("bear", ["bear"]),
            ("bird", ["bird", "eagle", "hawk", "falcon", "seagull", "crow"]),
            ("whale", ["whale", "dolphin"]),
            # Vehicles & Objects
            ("airplane", ["airplane", "aeroplane", "plane", "jet", "aircraft", "airliner"]),
            ("car", ["sports car", "racecar", "car", "vehicle", "automobile", "ferrari", "supercar", "truck"]),
            ("ball", ["red ball", "ball", "sphere", "orb"]),
            ("waterfall", ["waterfall", "cascade", "falls"]),
            ("motorcycle", ["motorcycle", "bike", "motorbike"]),
            ("ship", ["ship", "boat", "sailboat", "yacht", "vessel"]),
            ("spaceship", ["spaceship", "starship", "rocket", "shuttle"]),
            ("robot", ["robot", "android", "cyborg", "mech", "drone", "automaton", "rover"]),
            ("alien", ["alien", "extraterrestrial"]),
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
            ("walking", ["walking slowly", "walking", "walks", "strolling", "striding", "wandering", "marching"]),
            ("flying", ["flying through", "flying", "flies", "soaring", "gliding", "hovering"]),
            ("cooking", ["cooking food", "cooking", "cooks", "baking", "preparing food"]),
            ("driving", ["driving on", "driving", "drives", "racing", "speeding", "cruising"]),
            ("moving", ["moving from left to right", "moving", "moves", "rolling", "traversing"]),
            ("sailing", ["sailing", "sails", "cruising", "floating", "drifting", "navigating"]),
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
            detected_action = "moving"

        # 3. Setting / Environment extraction
        setting_patterns = [
            ("field", ["agricultural field", "agriculture", "crop field", "paddy field", "farmland", "farm", "crops", "wheat field", "rice field", "meadow", "pasture", "field"]),
            ("savanna", ["african savanna", "savanna", "savannah", "safari", "grassland", "plains"]),
            ("kitchen", ["kitchen", "restaurant", "bakery", "cooking area"]),
            ("clouds", ["clouds", "cloud", "sky", "overcast", "aviation"]),
            ("highway", ["highway", "freeway", "road", "expressway", "asphalt", "speedway"]),
            ("snow", ["snowy forest", "snow", "winter", "ice", "arctic", "glacier", "blizzard"]),
            ("waterfall", ["waterfall", "dense forest", "cascade"]),
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
            match = re.search(r"\b(?:in|on|at|through|across|over|near|along)\s+(?:a|an|the)?\s*([a-z]+(?:\s+[a-z]+)?)\b", p_lower)
            if match:
                detected_setting = match.group(1).strip()
            else:
                detected_setting = "open world"

        # 4. Lighting & Time of Day
        if re.search(r"\b(?:sunrise|dawn|early morning|daybreak)\b", p_lower):
            lighting = "warm natural sunrise with radiant golden sunlight and soft morning shadows"
            time_of_day = "sunrise"
        elif re.search(r"\b(?:sunset|golden hour|dusk|twilight)\b", p_lower):
            lighting = "radiant golden hour sunset with rich amber, orange, and purple hues"
            time_of_day = "sunset"
        elif re.search(r"\b(?:night|midnight|dark sky)\b", p_lower):
            lighting = "deep night with luminous moonlight, shadows, and glowing accent lights"
            time_of_day = "night"
        elif re.search(r"\b(?:storm|rain|thunder)\b", p_lower):
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
        if re.search(r"\b(?:fast|speed|racing)\b", p_lower):
            mood = "high-speed exhilaration and dynamic adrenaline"
        elif re.search(r"\b(?:peaceful|calm|quiet|slowly|serene)\b", p_lower):
            mood = "serene tranquility and peaceful authentic life"
        elif re.search(r"\b(?:mysterious|dark)\b", p_lower):
            mood = "suspenseful intrigue and mysterious discovery"
        elif re.search(r"\b(?:joy|happy|playful)\b", p_lower) or detected_subject == "dog":
            mood = "vibrant freedom and exhilarating joy"
        else:
            mood = "cinematic wonder and authentic realism"

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
        if setting in ["field", "farmland", "farm", "crops", "agriculture"]:
            act1_beat = f"An Indian farmer in traditional attire walking slowly through the lush green agricultural field at sunrise."
            act1_narr = f"Under the warm sunrise sky, an Indian farmer walks slowly through the lush green field, morning light casting gentle shadows."
            act2_beat = f"Carrying a small farming tool, the farmer advances through rows of crops as gentle wind creates ripples across the field."
            act2_narr = f"Every step reflects authentic rural rhythm as wind stirs the green crops and distant birds fly across the pastel sky."
            act3_beat = f"The farmer pauses amidst the vibrant crop rows under the golden morning sun, peaceful and grounded in nature."
            act3_narr = f"Bathed in the warm natural sunlight of dawn, the morning work begins with timeless grace and shallow depth of field framing."

        elif setting in ["beach", "ocean", "coast"]:
            act1_beat = f"A {subject} {action} along the sunlit shore, leaving tracks across the glistening wet sand."
            act1_narr = f"Under the warm sky, a {subject} {action} along the wide open coastline as gentle waves wash ashore."
            act2_beat = f"Splashing through sparkling sea foam, the {subject} gathers momentum along the water's edge."
            act2_narr = f"Every leap sends sprays of crystal seawater catching the brilliant light, full of energy and movement."
            act3_beat = f"The {subject} pauses on a scenic sand ridge, gazing out across the vast ocean horizon."
            act3_narr = f"Looking out across the endless sea, the journey captures a moment of pure freedom and boundless beauty."

        elif setting in ["kitchen", "restaurant"]:
            act1_beat = f"A chef preparing fresh ingredients at the culinary station under warm focused lighting."
            act1_narr = f"In the kitchen, a master chef begins the culinary journey with precision and focus."
            act2_beat = f"Cooking food over the hot stove, aromatic steam and sizzling flavors fill the air."
            act2_narr = f"With experienced hands, the chef works the pan as flavors come alive in a vibrant culinary display."
            act3_beat = f"Plating the finished dish with artistry, the culinary masterpiece is complete."
            act3_narr = f"The art of cuisine is captured in every glistening detail, full of warmth and dedication."

        elif setting in ["clouds", "sky"]:
            act1_beat = f"A sleek {subject} soaring gracefully through expansive white cumulus clouds under clear blue skies."
            act1_narr = f"High above the earth, a {subject} cuts smoothly through towering cloud formations."
            act2_beat = f"Gliding effortlessly between sunlit vapor banks, the aircraft maintains steady flight."
            act2_narr = f"Sunlight reflects across polished wings as the journey through the upper atmosphere unfolds."
            act3_beat = f"Banking gently into the vast open horizon, the {subject} claims the sky."
            act3_narr = f"Against the endless blue expanse, the flight concludes with majestic cinematic grace."

        elif setting in ["savanna", "safari"]:
            act1_beat = f"A majestic elephant walking steadily across the golden African savanna under the open sun."
            act1_narr = f"Across the sweeping savanna grasslands, an elephant strides gracefully past acacia trees."
            act2_beat = f"Moving with quiet power along the ancient trail, dust gently rises under every step."
            act2_narr = f"The living wilderness provides a timeless backdrop as the elephant navigates the open plains."
            act3_beat = f"Pausing beneath the wide golden horizon, the elephant surveys the vast African landscape."
            act3_narr = f"Under the golden light, the serene majesty of nature is captured in all its grandeur."

        elif setting in ["mars", "red planet", "moon", "space"]:
            act1_beat = f"A solitary {subject} {action} across the rugged, crimson Martian terrain under a dusky sky."
            act1_narr = f"On a distant, quiet world, a {subject} takes historic steps across the rust-red soil of Mars."
            act2_beat = f"Navigating ancient impact ridges, the {subject} surveys the desolate beauty of towering canyons."
            act2_narr = f"Dust particles swirl in the thin atmosphere as the {subject} presses forward into uncharted territory."
            act3_beat = f"Standing upon a towering plateau, the {subject} gazes out over the endless Martian horizon."
            act3_narr = f"Beneath the distant sun, the vast planetary expanse stretches into eternity, breathtaking and profound."

        elif setting in ["highway", "road", "city", "cyberpunk"]:
            act1_beat = f"A sleek {subject} {action} along the open thoroughfare under {lighting}."
            act1_narr = f"Amidst the open road, the {subject} bursts into smooth motion as light catches its streamlined curves."
            act2_beat = f"Carving smoothly along the route, the {subject} maintains dynamic speed and precision."
            act2_narr = f"Power and momentum intertwine as every frame captures the fluidity of motion."
            act3_beat = f"Surging towards the distant horizon, the sequence captures the spirit of the open road."
            act3_narr = f"Light and speed converge into a stunning panorama of modern cinematic motion."

        elif setting in ["snow", "winter"]:
            act1_beat = f"A {subject} {action} through the pristine snow-covered forest path under crisp winter light."
            act1_narr = f"Through the quiet snowy pine trees, a {subject} moves joyfully across fresh powdery snow."
            act2_beat = f"Bounding past frost-tipped branches, energetic strides send powder kicking into the air."
            act2_narr = f"The winter landscape sparkles in the cool daylight as movement brings warmth to the forest."
            act3_beat = f"Pausing in a sun-dappled snow clearing, the {subject} catches its breath in the quiet wilderness."
            act3_narr = f"Surrounded by silent snowy beauty, the sequence concludes with crisp winter serenity."

        else:
            # Fully customized dynamic storyline
            act1_beat = f"The {subject} begins {action} across the landscape of {setting} under {lighting}."
            act1_narr = f"The journey unfolds as a {subject} moves across {setting}, illuminated by {lighting}."
            act2_beat = f"Continuing forward, the {subject} dynamically traverses the environment with growing momentum."
            act2_narr = f"Surrounded by the unique atmosphere of {setting}, every moment reveals rich visual detail."
            act3_beat = f"Reaching a scenic vantage point, the {subject} overlooks the breathtaking vista."
            act3_narr = f"Against the scenic expanse of {setting}, the sequence concludes with cinematic majesty."

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
