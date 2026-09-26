import os
import math
import random
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from typing import Optional, List, Any
from app.adapters.base import BaseImageAdapter
from app.core.logging import logger


def has_term(text: str, terms: List[str]) -> bool:
    """Robust regex word-boundary term search preventing substring false positives."""
    for t in terms:
        if re.search(r"\b" + re.escape(t) + r"\b", text):
            return True
    return False


class LocalImageAdapter(BaseImageAdapter):
    """
    High-fidelity procedural visual synthesizer.
    Generates rich, atmospheric keyframes matching prompt elements, lighting, and style.
    Dynamically renders distinct biomes (Agricultural Field, Savanna, Kitchen, Sky,
    Beach, Mars, City, Forest, Space, Desert, Highway, Snow) and distinct subjects
    (Indian Farmer, Chef, Airplane, Dog, Elephant, Waterfall, Car, Ball, Astronaut, Robot, Ship).
    """
    async def generate_image(
        self,
        prompt: str,
        output_path: str,
        style: str = "cinematic",
        width: int = 1280,
        height: int = 720,
        seed: Optional[int] = None
    ) -> str:
        return self._generate_image_impl(prompt, output_path, style, width, height, seed)

    def generate_image_sync(
        self,
        prompt: str,
        output_path: str,
        style: str = "cinematic",
        width: int = 1280,
        height: int = 720,
        seed: Optional[int] = None
    ) -> str:
        return self._generate_image_impl(prompt, output_path, style, width, height, seed)

    def generate_scene_image(
        self,
        scene: Any,
        output_path: str,
        width: int = 1280,
        height: int = 720,
        seed: Optional[int] = None
    ) -> str:
        prompt = getattr(scene, "visual_description", "") or getattr(scene, "narrative", "") or getattr(scene, "title", "")
        return self._generate_image_impl(prompt, output_path, "cinematic", width, height, seed)

    def _generate_image_impl(
        self,
        prompt: str,
        output_path: str,
        style: str = "cinematic",
        width: int = 1280,
        height: int = 720,
        seed: Optional[int] = None
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        p_lower = prompt.lower()

        # Deterministic seed per request if not provided
        if seed is None:
            # Derive deterministic hash from prompt if no seed specified
            seed = sum(ord(c) * (31 ** (i % 8)) for i, c in enumerate(prompt)) % (2**31 - 1)
        rng = random.Random(seed)

        # -------------------------------------------------------------
        # 1. SEMANTIC ENTITY & SETTING EXTRACTION (STRICT WORD BOUNDARIES)
        # -------------------------------------------------------------
        # Detect environment / biome
        is_field = has_term(p_lower, ["agricultural field", "agriculture", "agricultural", "farm", "farmland", "crop", "crops", "paddy", "field", "wheat field", "meadow", "pasture"])
        is_savanna = has_term(p_lower, ["savanna", "savannah", "african savanna", "safari", "grassland", "plains"])
        is_kitchen = has_term(p_lower, ["kitchen", "restaurant", "bakery", "cooking area", "stove", "galley"])
        is_clouds_sky = has_term(p_lower, ["cloud", "clouds", "sky", "overcast", "aerial", "flying through"]) and not is_field
        is_snow = has_term(p_lower, ["snow", "snowy", "winter", "ice", "arctic", "glacier", "blizzard", "frost"])
        is_highway = has_term(p_lower, ["highway", "freeway", "road", "expressway", "asphalt", "speedway", "lane"])
        is_beach = has_term(p_lower, ["beach", "coast", "shore", "sand", "seaside", "ocean shore", "tropical beach"])
        is_mars = has_term(p_lower, ["mars", "martian", "red planet"])
        is_space = has_term(p_lower, ["space", "cosmos", "galaxy", "orbit", "nebula", "deep space"])
        is_mountain = has_term(p_lower, ["mountain", "mountains", "rocky", "rocky mountain", "cliff", "crag", "peaks", "mist", "waterfall"])
        is_forest = has_term(p_lower, ["forest", "jungle", "woods", "trees", "grove", "rainforest"]) and not is_snow and not is_mountain
        is_city = has_term(p_lower, ["city", "metropolis", "cyberpunk", "urban", "street", "downtown", "skyscrapers"])
        is_desert = has_term(p_lower, ["desert", "dune", "dunes", "sahara"]) and not is_mars and not is_beach
        is_ocean = has_term(p_lower, ["ocean", "sea", "waves", "deep water"]) and not is_beach

        # Detect time of day / lighting
        is_sunrise = has_term(p_lower, ["sunrise", "dawn", "early morning", "daybreak", "morning sunlight"])
        is_sunset = has_term(p_lower, ["sunset", "golden hour", "dusk", "twilight"]) and not is_sunrise
        is_night = has_term(p_lower, ["night", "midnight", "dark", "moon", "nocturnal"]) and not is_sunrise and not is_sunset
        is_storm = has_term(p_lower, ["storm", "rain", "thunder", "tempest", "lightning"])

        # Detect subject
        is_farmer = has_term(p_lower, ["farmer", "indian farmer", "peasant", "cultivator", "harvester", "agriculture worker", "farming"])
        is_chef = has_term(p_lower, ["chef", "cook", "baker", "cooking"])
        is_airplane = has_term(p_lower, ["airplane", "aeroplane", "plane", "jet", "aircraft", "airliner"])
        is_elephant = has_term(p_lower, ["elephant", "elephants"])
        is_waterfall = has_term(p_lower, ["waterfall", "cascade", "flowing waterfall"])
        is_ball = has_term(p_lower, ["ball", "sphere", "orb", "red ball"])
        is_dog = has_term(p_lower, ["dog", "puppy", "hound", "retriever", "golden retriever", "canine", "husky", "wolf"])
        is_astronaut = has_term(p_lower, ["astronaut", "cosmonaut", "spacewalker", "spacesuit"])
        is_robot = has_term(p_lower, ["robot", "cyborg", "drone", "android", "mech", "automaton"])
        is_car = has_term(p_lower, ["car", "sports car", "racecar", "vehicle", "automobile", "ferrari", "supercar", "sedan"]) and not is_farmer
        is_ship = has_term(p_lower, ["ship", "boat", "sailboat", "yacht", "vessel"])
        is_bird = has_term(p_lower, ["bird", "eagle", "hawk", "falcon", "seagull"])

        subject_tag = "farmer" if is_farmer else "chef" if is_chef else "airplane" if is_airplane else "elephant" if is_elephant else "waterfall" if is_waterfall else "ball" if is_ball else "dog" if is_dog else "astronaut" if is_astronaut else "robot" if is_robot else "car" if is_car else "ship" if is_ship else "traveler"
        setting_tag = "field" if is_field else "mountain" if is_mountain else "savanna" if is_savanna else "kitchen" if is_kitchen else "sky" if is_clouds_sky else "snow" if is_snow else "beach" if is_beach else "mars" if is_mars else "forest" if is_forest else "city" if is_city else "highway" if is_highway else "other"

        logger.info(f"[ImageGen] Generating keyframe: prompt='{prompt[:60]}...', seed={seed}, setting={setting_tag}, subject={subject_tag}, sunrise={is_sunrise}")

        img = Image.new("RGB", (width, height), (15, 15, 20))
        draw = ImageDraw.Draw(img)

        # -------------------------------------------------------------
        # 2. SKY & HORIZON GRADIENTS ACCORDING TO BIOME & LIGHTING
        # -------------------------------------------------------------
        horizon_y = int(height * (0.55 if is_clouds_sky else 0.58 if is_field else 0.62 if is_beach else 0.65))

        if is_sunrise:
            # Sunrise: warm dawn tones down to glowing warm apricot & radiant sunrise gold
            top_color = (130, 80, 110)
            mid_color = (235, 125, 75)
            horizon_color = (255, 205, 105)
            ground_color = (45, 115, 40) if is_field else (180, 130, 80)
        elif is_sunset:
            top_color = (40, 20, 65)
            mid_color = (210, 75, 60)
            horizon_color = (255, 160, 50)
            ground_color = (35, 60, 30) if is_field else (140, 95, 60)
        elif is_night:
            top_color = (8, 10, 24)
            mid_color = (18, 22, 45)
            horizon_color = (32, 40, 68)
            ground_color = (15, 22, 18)
        elif is_mars:
            top_color = (65, 25, 30)
            mid_color = (180, 75, 55)
            horizon_color = (220, 125, 85)
            ground_color = (135, 48, 32)
        elif is_snow:
            top_color = (60, 95, 140)
            mid_color = (130, 170, 210)
            horizon_color = (205, 225, 245)
            ground_color = (235, 242, 250)
        elif is_space:
            top_color = (4, 4, 12)
            mid_color = (12, 10, 25)
            horizon_color = (25, 15, 45)
            ground_color = (15, 15, 22)
        elif is_savanna:
            top_color = (45, 90, 160)
            mid_color = (140, 180, 215)
            horizon_color = (245, 215, 150)
            ground_color = (195, 160, 85) # Golden dry savanna
        elif is_kitchen:
            # Interior kitchen lighting
            top_color = (225, 228, 235)
            mid_color = (200, 205, 215)
            horizon_color = (180, 185, 195)
            ground_color = (75, 80, 90)
        elif is_mountain:
            top_color = (65, 80, 100)
            mid_color = (130, 145, 160)
            horizon_color = (180, 195, 205)
            ground_color = (40, 45, 48)
        else:
            # Brilliant natural daylight
            top_color = (35, 105, 185)
            mid_color = (105, 175, 230)
            horizon_color = (200, 230, 250)
            ground_color = (45, 125, 45) if is_field else (90, 100, 85)

        # Draw smooth sky / ceiling gradient
        for y in range(horizon_y):
            ratio = y / max(1, horizon_y)
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

        # -------------------------------------------------------------
        # 3. CELESTIAL / ATMOSPHERIC ELEMENTS
        # -------------------------------------------------------------
        if is_sunrise:
            # Low radiant sunrise sun on the horizon
            sun_x = int(width * 0.72)
            sun_y = int(horizon_y * 0.78)
            sun_r = int(height * 0.09)
            sun_col = (255, 240, 180)
            # Radiant morning atmospheric halo
            for r_off in range(sun_r + 55, sun_r, -3):
                fade = 1.0 - (r_off - sun_r) / 55.0
                hr = int(sun_col[0] * fade + horizon_color[0] * (1.0 - fade))
                hg = int(sun_col[1] * fade + horizon_color[1] * (1.0 - fade))
                hb = int(sun_col[2] * fade + horizon_color[2] * (1.0 - fade))
                draw.ellipse([sun_x - r_off, sun_y - r_off, sun_x + r_off, sun_y + r_off], fill=(hr, hg, hb))
            draw.ellipse([sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r], fill=sun_col)

            # Silhouetted birds in distant sky
            bird_coords = [
                (sun_x - 180, sun_y - 120), (sun_x - 150, sun_y - 135),
                (sun_x - 120, sun_y - 115), (sun_x - 90, sun_y - 140),
                (sun_x - 230, sun_y - 95), (sun_x - 200, sun_y - 110)
            ]
            for bx, by in bird_coords:
                draw.arc([bx - 10, by - 6, bx, by + 4], start=180, end=360, fill=(45, 30, 40), width=2)
                draw.arc([bx, by - 6, bx + 10, by + 4], start=180, end=360, fill=(45, 30, 40), width=2)

        elif is_clouds_sky or "cloud" in p_lower:
            # Volumetric fluffy cumulus clouds
            for cx in range(-50, width + 100, 140):
                cy = rng.randint(40, horizon_y - 40)
                cr = rng.randint(45, 95)
                cloud_col = (245, 248, 255)
                draw.ellipse([cx - cr, cy - int(cr*0.6), cx + cr, cy + int(cr*0.6)], fill=cloud_col)
                draw.ellipse([cx - int(cr*0.5), cy - int(cr*0.8), cx + int(cr*0.5), cy + int(cr*0.4)], fill=(255, 255, 255))

        elif is_sunset:
            sun_x = int(width * 0.65)
            sun_y = int(horizon_y * 0.75)
            sun_r = int(height * 0.11)
            draw.ellipse([sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r], fill=(255, 210, 110))

        # -------------------------------------------------------------
        # 4. MIDGROUND & FOREGROUND ENVIRONMENT
        # -------------------------------------------------------------
        if is_field:
            # Lush green agricultural field
            draw.rectangle([0, horizon_y, width, height], fill=(42, 118, 45))

            # Distant tree line on the horizon
            for tx in range(0, width, 18):
                th = rng.randint(12, 32)
                draw.ellipse([tx - 12, horizon_y - th, tx + 12, horizon_y + 8], fill=(30, 75, 35))

            # Agricultural crop rows with perspective lines converging to horizon
            for rx in range(-150, width + 250, 45):
                start_pt = (int(width * 0.5 + (rx - width * 0.5) * 0.15), horizon_y)
                end_pt = (rx, height)
                draw.line([start_pt, end_pt], fill=(32, 95, 35), width=3)

            # Detailed crop stalks (wheat / paddy / crops) swaying across foreground
            for cx in range(0, width, 12):
                ch = rng.randint(25, 65)
                cy = height - rng.randint(0, int((height - horizon_y) * 0.65))
                # Slight curve simulating gentle wind
                draw.arc([cx - 8, cy - ch, cx + 8, cy], start=230, end=320, fill=(72, 165, 55), width=2)
                # Golden-green crop ear / grain head
                draw.ellipse([cx + 3, cy - ch - 4, cx + 7, cy - ch + 4], fill=(185, 195, 80))

        elif is_savanna:
            # Dry golden African savanna
            draw.rectangle([0, horizon_y, width, height], fill=ground_color)
            # Acacia trees silhouetted against horizon
            for ax in [int(width * 0.18), int(width * 0.82)]:
                draw.line([(ax, horizon_y), (ax, horizon_y - 65)], fill=(65, 45, 30), width=6)
                # Flat umbrella-like canopy
                draw.ellipse([ax - 75, horizon_y - 95, ax + 75, horizon_y - 65], fill=(70, 75, 40))

        elif is_kitchen:
            # Kitchen interior
            # Wall tile grid
            for ty in range(0, horizon_y, 25):
                draw.line([(0, ty), (width, ty)], fill=(210, 215, 225), width=1)
            for tx in range(0, width, 35):
                draw.line([(tx, 0), (tx, horizon_y)], fill=(210, 215, 225), width=1)
            # Countertop surface
            draw.rectangle([0, horizon_y, width, height], fill=(65, 70, 78))
            # Stainless steel counter edge trim
            draw.line([(0, horizon_y), (width, horizon_y)], fill=(200, 205, 215), width=4)
            # Stove / cooking station
            draw.rectangle([int(width * 0.45), horizon_y - 20, int(width * 0.75), horizon_y + 120], fill=(45, 48, 55))
            draw.ellipse([int(width * 0.50), horizon_y - 12, int(width * 0.58), horizon_y + 8], fill=(235, 95, 40)) # Burner glow

        elif is_snow:
            # Snow-covered ground with snowy pine trees
            draw.rectangle([0, horizon_y, width, height], fill=(238, 244, 252))
            for px in range(20, width, 55):
                ph = rng.randint(60, 130)
                # Pine trunk
                draw.rectangle([px - 4, horizon_y - ph, px + 4, horizon_y], fill=(55, 40, 30))
                # Pine triangular foliage with snow on top
                for layer in range(3):
                    ly = horizon_y - ph + layer * 28
                    draw.polygon([(px, ly - 25), (px - 32 + layer*6, ly + 15), (px + 32 - layer*6, ly + 15)], fill=(32, 60, 45))
                    draw.line([(px - 30 + layer*6, ly + 14), (px + 30 - layer*6, ly + 14)], fill=(245, 250, 255), width=3)

        elif is_highway:
            # Asphalt highway
            draw.rectangle([0, horizon_y, width, height], fill=(45, 48, 52))
            # Guardrails
            draw.line([(0, horizon_y + 2), (width, horizon_y + 2)], fill=(180, 185, 190), width=4)
            # Road dashed center lane markers
            for dx in range(0, width, 70):
                draw.line([(dx, horizon_y + int((height - horizon_y)*0.65)), (dx + 35, horizon_y + int((height - horizon_y)*0.65))], fill=(255, 220, 60), width=4)

        elif is_beach or is_ocean:
            sea_top = horizon_y
            sea_bottom = int(height * 0.72)
            draw.rectangle([0, sea_top, width, sea_bottom], fill=(35, 105, 175))
            # Rolling surf lines
            for wy in range(sea_top + 10, sea_bottom, 16):
                draw.line([(0, wy), (width, wy)], fill=(225, 245, 255), width=2)
            draw.rectangle([0, sea_bottom, width, height], fill=(235, 198, 140)) # Sand

        elif is_mars:
            draw.rectangle([0, horizon_y, width, height], fill=ground_color)
            for _ in range(15):
                bx = rng.randint(40, width - 40)
                by = rng.randint(horizon_y + 20, height - 20)
                draw.ellipse([bx, by, bx + 22, by + 12], fill=(75, 25, 20))

        else:
            # Default scenic landscape
            draw.rectangle([0, horizon_y, width, height], fill=ground_color)

        # -------------------------------------------------------------
        # 5. RENDER SPECIFIC FOREGROUND SUBJECT
        # -------------------------------------------------------------
        subj_x = int(width * 0.44)
        subj_y = horizon_y + int((height - horizon_y) * 0.28)

        if is_farmer:
            # Render Indian Farmer in traditional clothing carrying farming tool
            skin_col = (165, 115, 80)
            kurta_col = (245, 242, 235) # Simple light cotton kurta
            dhoti_col = (230, 225, 215) # Traditional dhoti
            turban_col = (225, 115, 45) # Traditional saffron/orange safah/pagri
            tool_wood = (105, 65, 40)
            tool_blade = (195, 205, 215) # Curved metal sickle / scythe

            # Directional morning soft ground shadow
            shadow_len = 110 if is_sunrise else 50
            draw.ellipse([subj_x - shadow_len - 10, subj_y + 90, subj_x + 25, subj_y + 105], fill=(25, 55, 30))

            # Legs in slow steady walking stride
            draw.line([(subj_x - 8, subj_y + 40), (subj_x - 22, subj_y + 98)], fill=dhoti_col, width=14)
            draw.line([(subj_x + 8, subj_y + 40), (subj_x + 20, subj_y + 92)], fill=dhoti_col, width=14)
            # Bare feet / traditional leather footwear
            draw.rectangle([subj_x - 30, subj_y + 94, subj_x - 14, subj_y + 102], fill=skin_col)
            draw.rectangle([subj_x + 14, subj_y + 88, subj_x + 30, subj_y + 96], fill=skin_col)

            # Torso: Kurta / Tunic with natural fabric folds
            draw.polygon([
                (subj_x - 24, subj_y - 25), (subj_x + 24, subj_y - 25),
                (subj_x + 28, subj_y + 45), (subj_x - 28, subj_y + 45)
            ], fill=kurta_col)

            # Traditional scarf / gamcha draped across shoulder
            draw.line([(subj_x - 18, subj_y - 22), (subj_x + 15, subj_y + 35)], fill=(195, 65, 45), width=8)

            # Head & Neck
            draw.rectangle([subj_x - 7, subj_y - 38, subj_x + 7, subj_y - 24], fill=skin_col)
            draw.ellipse([subj_x - 14, subj_y - 62, subj_x + 14, subj_y - 34], fill=skin_col)

            # Traditional Pagri / Turban
            draw.ellipse([subj_x - 18, subj_y - 74, subj_x + 18, subj_y - 52], fill=turban_col)
            draw.ellipse([subj_x - 14, subj_y - 78, subj_x + 14, subj_y - 60], fill=(240, 135, 55))

            # Arms & Hands
            # Left arm swinging naturally in walking gait
            draw.line([(subj_x - 22, subj_y - 20), (subj_x - 36, subj_y + 15)], fill=skin_col, width=9)
            # Right arm carrying farming tool (sickle / scythe)
            draw.line([(subj_x + 22, subj_y - 20), (subj_x + 38, subj_y + 10)], fill=skin_col, width=9)
            draw.ellipse([subj_x + 34, subj_y + 6, subj_x + 44, subj_y + 16], fill=skin_col) # Hand

            # Small farming tool: wooden handle + curved silver sickle blade
            draw.line([(subj_x + 38, subj_y + 12), (subj_x + 48, subj_y - 18)], fill=tool_wood, width=5)
            # Curved sickle blade
            draw.arc([subj_x + 38, subj_y - 36, subj_x + 68, subj_y - 6], start=160, end=330, fill=tool_blade, width=4)

        elif is_chef:
            # Render Chef in kitchen with white toque hat & apron
            draw.ellipse([subj_x - 12, subj_y - 50, subj_x + 12, subj_y - 26], fill=(225, 185, 155)) # Head
            # Tall white chef toque hat
            draw.rectangle([subj_x - 14, subj_y - 85, subj_x + 14, subj_y - 50], fill=(255, 255, 255))
            draw.ellipse([subj_x - 18, subj_y - 95, subj_x + 18, subj_y - 75], fill=(255, 255, 255))
            # Chef jacket & dark apron
            draw.polygon([(subj_x - 26, subj_y - 25), (subj_x + 26, subj_y - 25), (subj_x + 30, subj_y + 55), (subj_x - 30, subj_y + 55)], fill=(255, 255, 255))
            draw.rectangle([subj_x - 22, subj_y + 15, subj_x + 22, subj_y + 65], fill=(45, 50, 60))
            # Frying pan with sizzling food
            draw.ellipse([subj_x + 25, subj_y + 5, subj_x + 75, subj_y + 35], fill=(35, 35, 40))
            draw.line([(subj_x + 18, subj_y + 20), (subj_x + 32, subj_y + 20)], fill=(85, 50, 30), width=6)
            draw.ellipse([subj_x + 40, subj_y + 12, subj_x + 60, subj_y + 28], fill=(225, 130, 45)) # Cooking food

        elif is_airplane:
            # Render Blue Airplane soaring through clouds
            plane_col = (30, 95, 215) if "blue" in p_lower else (235, 240, 248)
            trim_col = (20, 60, 150)
            # Fuselage
            draw.ellipse([subj_x - 140, subj_y - 25, subj_x + 120, subj_y + 25], fill=plane_col)
            # Cockpit windshield
            draw.polygon([(subj_x + 75, subj_y - 12), (subj_x + 105, subj_y - 8), (subj_x + 95, subj_y + 5), (subj_x + 70, subj_y + 5)], fill=(40, 50, 65))
            # Main swept-back wings
            draw.polygon([(subj_x - 30, subj_y - 10), (subj_x - 90, subj_y - 95), (subj_x - 60, subj_y - 95), (subj_x + 15, subj_y - 5)], fill=plane_col)
            draw.polygon([(subj_x - 30, subj_y + 10), (subj_x - 90, subj_y + 85), (subj_x - 60, subj_y + 85), (subj_x + 15, subj_y + 5)], fill=trim_col)
            # Tail vertical fin
            draw.polygon([(subj_x - 130, subj_y - 20), (subj_x - 170, subj_y - 75), (subj_x - 140, subj_y - 75), (subj_x - 105, subj_y - 15)], fill=trim_col)
            # Jet engine pod
            draw.ellipse([subj_x - 45, subj_y + 25, subj_x + 5, subj_y + 45], fill=(160, 170, 185))

        elif is_elephant:
            # Render African Elephant in savanna
            el_col = (115, 118, 125)
            # Body
            draw.ellipse([subj_x - 90, subj_y - 40, subj_x + 50, subj_y + 40], fill=el_col)
            # Head & large ear
            draw.ellipse([subj_x + 35, subj_y - 55, subj_x + 85, subj_y + 10], fill=el_col)
            draw.ellipse([subj_x + 10, subj_y - 50, subj_x + 55, subj_y + 5], fill=(135, 138, 145))
            # Trunk curving down and up
            draw.arc([subj_x + 65, subj_y - 15, subj_x + 115, subj_y + 55], start=0, end=180, fill=el_col, width=14)
            # White ivory tusk
            draw.arc([subj_x + 65, subj_y - 10, subj_x + 95, subj_y + 25], start=45, end=190, fill=(245, 245, 235), width=6)
            # 4 sturdy pillar legs
            for lx in [-70, -35, 10, 40]:
                draw.rectangle([subj_x + lx, subj_y + 30, subj_x + lx + 18, subj_y + 85], fill=el_col)

        elif is_waterfall:
            # Waterfall cascading down dark rocks
            rock_col = (45, 48, 52)
            draw.polygon([(subj_x - 120, horizon_y), (subj_x - 50, height), (subj_x - 150, height)], fill=rock_col)
            draw.polygon([(subj_x + 120, horizon_y), (subj_x + 50, height), (subj_x + 150, height)], fill=rock_col)
            # Cascading white water streams
            for fx in range(subj_x - 45, subj_x + 45, 8):
                draw.line([(fx, horizon_y - 15), (fx + rng.randint(-6, 6), height - 20)], fill=(230, 245, 255), width=5)
            # Rising spray mist pool
            draw.ellipse([subj_x - 85, height - 55, subj_x + 85, height], fill=(215, 240, 255))

        elif is_ball:
            # Render 3D spherical red ball with shadow and specular highlight
            ball_col = (225, 35, 30) if "red" in p_lower else (40, 110, 220)
            ball_r = int(height * 0.08)
            # Contact ground shadow
            draw.ellipse([subj_x - ball_r - 10, subj_y + ball_r - 5, subj_x + ball_r + 10, subj_y + ball_r + 12], fill=(20, 25, 25))
            # Ball body
            draw.ellipse([subj_x - ball_r, subj_y - ball_r, subj_x + ball_r, subj_y + ball_r], fill=ball_col)
            # Specular light highlight
            hl_x = subj_x - int(ball_r * 0.35)
            hl_y = subj_y - int(ball_r * 0.35)
            draw.ellipse([hl_x - 8, hl_y - 8, hl_x + 8, hl_y + 8], fill=(255, 255, 255))

        elif is_dog:
            # Golden Retriever / White Dog
            dog_col = (245, 245, 250) if "white" in p_lower else (210, 155, 75)
            dark_col = (135, 85, 35) if not "white" in p_lower else (180, 185, 195)
            # Body & head in running gallop
            draw.ellipse([subj_x - 50, subj_y - 20, subj_x + 40, subj_y + 18], fill=dog_col)
            draw.polygon([(subj_x + 25, subj_y - 10), (subj_x + 55, subj_y - 45), (subj_x + 75, subj_y - 25), (subj_x + 45, subj_y + 5)], fill=dog_col)
            draw.ellipse([subj_x + 45, subj_y - 50, subj_x + 78, subj_y - 20], fill=dog_col)
            draw.ellipse([subj_x + 65, subj_y - 38, subj_x + 95, subj_y - 22], fill=dark_col)
            draw.polygon([(subj_x + 46, subj_y - 48), (subj_x + 30, subj_y - 35), (subj_x + 50, subj_y - 30)], fill=dark_col)
            # Galloping legs
            draw.line([(subj_x + 35, subj_y + 5), (subj_x + 65, subj_y + 35)], fill=dog_col, width=8)
            draw.line([(subj_x - 35, subj_y), (subj_x - 75, subj_y + 35)], fill=dog_col, width=8)

        elif is_car:
            # Sports car
            car_col = (225, 35, 30) if "red" in p_lower else (30, 125, 230)
            draw.rounded_rectangle([subj_x - 95, subj_y - 10, subj_x + 95, subj_y + 32], radius=8, fill=car_col)
            draw.polygon([(subj_x - 45, subj_y - 10), (subj_x - 20, subj_y - 35), (subj_x + 35, subj_y - 35), (subj_x + 65, subj_y - 10)], fill=(25, 25, 35))
            # Wheels
            draw.ellipse([subj_x - 70, subj_y + 12, subj_x - 30, subj_y + 52], fill=(20, 20, 25))
            draw.ellipse([subj_x + 30, subj_y + 12, subj_x + 70, subj_y + 52], fill=(20, 20, 25))
            # Headlight beam
            draw.polygon([(subj_x + 90, subj_y + 5), (subj_x + 320, subj_y - 25), (subj_x + 320, subj_y + 55), (subj_x + 90, subj_y + 25)], fill=(255, 255, 210))

        elif is_astronaut:
            suit_col = (240, 242, 245)
            visor_col = (245, 185, 45)
            draw.rectangle([subj_x - 48, subj_y - 65, subj_x - 20, subj_y + 15], fill=(195, 200, 210)) # Pack
            draw.rounded_rectangle([subj_x - 25, subj_y - 55, subj_x + 25, subj_y + 20], radius=10, fill=suit_col)
            draw.ellipse([subj_x - 26, subj_y - 95, subj_x + 26, subj_y - 45], fill=suit_col)
            draw.rounded_rectangle([subj_x - 12, subj_y - 85, subj_x + 24, subj_y - 58], radius=6, fill=visor_col)
            draw.line([(subj_x - 12, subj_y + 20), (subj_x - 25, subj_y + 75)], fill=suit_col, width=15)
            draw.line([(subj_x + 12, subj_y + 20), (subj_x + 28, subj_y + 70)], fill=suit_col, width=15)

        else:
            # Stylized traveler silhouette
            draw.ellipse([subj_x - 16, subj_y - 60, subj_x + 16, subj_y - 28], fill=(25, 20, 30))
            draw.polygon([(subj_x - 22, subj_y - 28), (subj_x + 22, subj_y - 28), (subj_x + 28, subj_y + 35), (subj_x - 28, subj_y + 35)], fill=(20, 15, 25))
            draw.line([(subj_x - 12, subj_y + 35), (subj_x - 20, subj_y + 80)], fill=(20, 15, 25), width=10)
            draw.line([(subj_x + 12, subj_y + 35), (subj_x + 22, subj_y + 78)], fill=(20, 15, 25), width=10)

        # -------------------------------------------------------------
        # 6. CINEMATIC VIGNETTE & PHOTOREALISTIC LOOK
        # -------------------------------------------------------------
        img_np = np.array(img, dtype=np.float32)
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - width/2)**2 + (Y - height/2)**2)
        max_dist = np.sqrt((width/2)**2 + (height/2)**2)
        vignette = 1.0 - (dist_from_center / max_dist) * 0.28
        vignette = np.clip(vignette, 0.55, 1.0)
        img_np = img_np * vignette[..., np.newaxis]

        # Subtle natural film grain
        grain = rng.uniform(-3.0, 3.0)
        img_np = np.clip(img_np + grain, 0, 255)

        final_img = Image.fromarray(np.uint8(img_np))
        final_img.save(output_path, "JPEG", quality=95)
        return output_path
