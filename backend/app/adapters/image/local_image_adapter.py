import os
import math
import random
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from typing import Optional
from app.adapters.base import BaseImageAdapter
from app.core.logging import logger

class LocalImageAdapter(BaseImageAdapter):
    """
    High-fidelity procedural visual synthesizer.
    Generates rich, atmospheric keyframes matching prompt elements, lighting, and style.
    Dynamically renders distinct biomes (Beach, Mars, City, Forest, Space, Desert, etc.)
    and distinct subjects (Dog, Astronaut, Robot, Car, Ship, Figure, etc.).
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        p_lower = prompt.lower()

        # Generate unique seed per request if not provided
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        rng = random.Random(seed)

        # Detect setting / environment
        is_beach = any(k in p_lower for k in ["beach", "coast", "shore", "sand", "seaside", "ocean shore"])
        is_mars = any(k in p_lower for k in ["mars", "martian", "red planet"])
        is_space = any(k in p_lower for k in ["space", "cosmos", "galaxy", "orbit", "nebula"])
        is_forest = any(k in p_lower for k in ["forest", "jungle", "woods", "trees", "grove"])
        is_city = any(k in p_lower for k in ["city", "metropolis", "cyberpunk", "urban", "street", "highway", "tower", "spire"])
        is_desert = any(k in p_lower for k in ["desert", "dune", "sahara"]) and not is_mars and not is_beach
        is_ocean = any(k in p_lower for k in ["ocean", "sea", "ship", "boat", "water", "waves"]) and not is_beach

        # Detect time of day / lighting
        is_sunset = any(k in p_lower for k in ["sunset", "golden hour", "dusk", "twilight"])
        is_night = any(k in p_lower for k in ["night", "midnight", "dark", "moon"])
        is_storm = any(k in p_lower for k in ["storm", "rain", "thunder", "tempest", "lightning"])

        # Detect subject
        is_dog = any(k in p_lower for k in ["dog", "puppy", "hound", "retriever", "canine", "husky", "wolf"])
        is_astronaut = any(k in p_lower for k in ["astronaut", "cosmonaut", "spacewalker", "spacesuit"])
        is_robot = any(k in p_lower for k in ["robot", "cyborg", "drone", "android", "mech", "automaton"])
        is_car = any(k in p_lower for k in ["car", "sports car", "racecar", "vehicle", "automobile", "ferrari", "supercar"])
        is_ship = any(k in p_lower for k in ["ship", "boat", "sailboat", "yacht", "vessel"])
        is_bird = any(k in p_lower for k in ["bird", "eagle", "hawk", "falcon", "dragon", "phoenix"])

        logger.info(f"[ImageGen] Generating keyframe: prompt='{prompt[:60]}...', seed={seed}, setting={('beach' if is_beach else 'mars' if is_mars else 'city' if is_city else 'other')}, subject={('dog' if is_dog else 'astronaut' if is_astronaut else 'robot' if is_robot else 'car' if is_car else 'figure')}")

        img = Image.new("RGB", (width, height), (10, 10, 15))
        draw = ImageDraw.Draw(img)

        # -------------------------------------------------------------
        # 1. RENDER SKY & HORIZON GRADIENT ACCORDING TO BIOME
        # -------------------------------------------------------------
        horizon_y = int(height * (0.60 if (is_beach or is_ocean) else 0.65))

        if is_mars:
            # Mars: Butterscotch / Salmon / Deep Crimson
            top_color = (60, 20, 25)
            mid_color = (180, 70, 50)
            horizon_color = (220, 120, 80)
            ground_color = (130, 45, 30)
        elif is_beach:
            if is_sunset:
                top_color = (40, 20, 70)
                mid_color = (220, 80, 70)
                horizon_color = (255, 175, 75)
            else:
                # Brilliant Tropical Azure Sky
                top_color = (30, 110, 210)
                mid_color = (90, 175, 240)
                horizon_color = (190, 235, 255)
            ground_color = (235, 195, 135) # Golden sand
        elif is_space:
            top_color = (5, 5, 15)
            mid_color = (15, 10, 30)
            horizon_color = (30, 15, 50)
            ground_color = (15, 15, 25)
        elif is_forest:
            top_color = (25, 60, 85)
            mid_color = (70, 130, 140)
            horizon_color = (160, 205, 180)
            ground_color = (25, 55, 25)
        elif is_city or "cyberpunk" in p_lower:
            if is_night or "cyberpunk" in p_lower:
                top_color = (10, 5, 25)
                mid_color = (45, 15, 65)
                horizon_color = (20, 75, 110)
                ground_color = (15, 12, 22)
            else:
                top_color = (35, 25, 60)
                mid_color = (180, 75, 65)
                horizon_color = (255, 160, 60)
                ground_color = (20, 15, 25)
        elif is_sunset:
            top_color = (35, 15, 60)
            mid_color = (190, 60, 60)
            horizon_color = (255, 160, 50)
            ground_color = (30, 20, 25)
        else:
            top_color = (35, 90, 160)
            mid_color = (100, 160, 215)
            horizon_color = (200, 225, 245)
            ground_color = (45, 65, 40)

        # Draw smooth sky gradient
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
        # 2. CELESTIAL / ATMOSPHERIC ELEMENTS
        # -------------------------------------------------------------
        if is_space:
            # Draw thousands of star points
            for _ in range(350):
                sx = rng.randint(0, width)
                sy = rng.randint(0, horizon_y)
                sc = rng.randint(180, 255)
                draw.point((sx, sy), fill=(sc, sc, sc))
            # Glowing nebula clouds
            for _ in range(8):
                nx = rng.randint(int(width*0.2), int(width*0.8))
                ny = rng.randint(40, horizon_y - 60)
                nr = rng.randint(60, 150)
                draw.ellipse([nx-nr, ny-nr, nx+nr, ny+nr], fill=(rng.randint(60, 140), rng.randint(10, 60), rng.randint(100, 200)))
        elif is_mars:
            # Two small Martian moons: Phobos and Deimos
            draw.ellipse([int(width*0.75), int(horizon_y*0.35), int(width*0.75)+22, int(horizon_y*0.35)+22], fill=(210, 190, 180))
            draw.ellipse([int(width*0.82), int(horizon_y*0.22), int(width*0.82)+10, int(horizon_y*0.22)+10], fill=(170, 150, 145))
            # Pale distant sun
            sun_x = int(width * 0.35)
            sun_y = int(horizon_y * 0.5)
            draw.ellipse([sun_x-25, sun_y-25, sun_x+25, sun_y+25], fill=(255, 230, 210))
        else:
            # Sun or Moon
            sun_x = int(width * 0.65)
            sun_y = int(horizon_y * (0.75 if is_sunset else 0.45))
            sun_radius = int(height * (0.12 if is_sunset else 0.08))
            sun_col = (255, 210, 120) if is_sunset else (255, 255, 230)
            # Atmospheric halo
            for r_off in range(sun_radius + 35, sun_radius, -3):
                fade = 1.0 - (r_off - sun_radius) / 35.0
                hr = int(sun_col[0] * fade + horizon_color[0] * (1.0 - fade))
                hg = int(sun_col[1] * fade + horizon_color[1] * (1.0 - fade))
                hb = int(sun_col[2] * fade + horizon_color[2] * (1.0 - fade))
                draw.ellipse([sun_x - r_off, sun_y - r_off, sun_x + r_off, sun_y + r_off], fill=(hr, hg, hb))
            draw.ellipse([sun_x - sun_radius, sun_y - sun_radius, sun_x + sun_radius, sun_y + sun_radius], fill=sun_col)

        # -------------------------------------------------------------
        # 3. MIDGROUND TERRAIN SPECIFIC TO BIOME
        # -------------------------------------------------------------
        if is_beach or is_ocean:
            # Distant sea horizon line
            sea_top = horizon_y
            sea_bottom = int(height * 0.72)
            # Ocean water gradient
            for y in range(sea_top, sea_bottom):
                frac = (y - sea_top) / max(1, sea_bottom - sea_top)
                wr = int(25 + frac * 20)
                wg = int(90 + frac * 60)
                wb = int(160 + frac * 40)
                draw.line([(0, y), (width, y)], fill=(wr, wg, wb))

            # Rolling ocean waves and surf foam lines
            for wy in range(sea_top + 15, sea_bottom, 18):
                wave_offset = rng.randint(0, 50)
                points = []
                for wx in range(0, width + 40, 30):
                    wave_amp = math.sin((wx + wave_offset) * 0.04) * 4
                    points.append((wx, wy + wave_amp))
                for idx in range(len(points) - 1):
                    draw.line([points[idx], points[idx + 1]], fill=(220, 245, 255), width=2)

            # Sandy beach shoreline
            sand_poly = [(0, sea_bottom), (width, sea_bottom - 20), (width, height), (0, height)]
            draw.polygon(sand_poly, fill=ground_color)
            # Wet sand reflection strip
            draw.rectangle([0, sea_bottom - 8, width, sea_bottom + 12], fill=(int(ground_color[0]*0.85), int(ground_color[1]*0.85), int(ground_color[2]*0.75)))
            # Foam wash along the wet sand line
            for fx in range(0, width, 25):
                draw.ellipse([fx, sea_bottom - 3, fx + 35, sea_bottom + 6], fill=(245, 250, 255))

        elif is_mars:
            # Jagged volcanic crater ridges in background
            ridge_pts = [(0, horizon_y)]
            for i in range(25):
                px = int(i * (width / 24))
                py = horizon_y - rng.randint(30, 95)
                ridge_pts.append((px, py))
            ridge_pts.append((width, horizon_y))
            draw.polygon(ridge_pts, fill=(100, 35, 25))

            # Mars reddish soil foreground
            draw.rectangle([0, horizon_y, width, height], fill=ground_color)

            # Mars undulating dunes & craters
            for dy in range(horizon_y + 20, height, 40):
                dune_color = (int(ground_color[0] + rng.randint(-15, 25)), int(ground_color[1] + rng.randint(-10, 15)), int(ground_color[2] + rng.randint(-8, 10)))
                draw.chord([rng.randint(-100, 0), dy, width + rng.randint(0, 100), dy + 80], 0, 180, fill=dune_color)

            # Scattered volcanic boulders
            for _ in range(12):
                bx = rng.randint(40, width - 40)
                by = rng.randint(horizon_y + 30, height - 30)
                bsize = rng.randint(8, 26)
                draw.ellipse([bx, by, bx + bsize, by + int(bsize*0.65)], fill=(75, 25, 20))

        elif is_city or "cyberpunk" in p_lower:
            # City skyline
            current_x = 0
            tower_col = (20, 15, 28) if is_night else (45, 35, 50)
            win_col = (0, 230, 255) if "cyberpunk" in p_lower else (255, 215, 130)

            while current_x < width:
                b_w = rng.randint(int(width*0.035), int(width*0.09))
                b_h = rng.randint(int(height*0.25), int(height*0.55))
                b_t = horizon_y - b_h
                draw.rectangle([current_x, b_t, current_x + b_w, horizon_y], fill=tower_col)

                # Antennas & beacons
                spire_x = current_x + b_w // 2
                draw.line([(spire_x, b_t), (spire_x, b_t - 25)], fill=tower_col, width=2)
                draw.point((spire_x, b_t - 25), fill=(255, 60, 60))

                # Window lights
                for wy in range(b_t + 15, horizon_y - 10, 14):
                    if rng.random() > 0.35:
                        for wx in range(current_x + 6, current_x + b_w - 6, 12):
                            if rng.random() > 0.4:
                                draw.rectangle([wx, wy, wx + 5, wy + 6], fill=win_col)
                current_x += b_w + rng.randint(2, 10)

            # Ground platform / road
            draw.rectangle([0, horizon_y, width, height], fill=ground_color)
            lane_color = (0, 200, 255) if "cyberpunk" in p_lower else (255, 200, 50)
            for gx in range(0, width, 140):
                draw.line([(gx, horizon_y), (int(width*0.5 + (gx - width*0.5)*2.5), height)], fill=(45, 40, 60), width=2)
            draw.line([(0, horizon_y + 5), (width, horizon_y + 5)], fill=lane_color, width=3)

        elif is_forest:
            # Forest background hills
            draw.chord([-100, horizon_y - 80, width + 100, horizon_y + 120], 0, 180, fill=(35, 75, 45))
            draw.rectangle([0, horizon_y, width, height], fill=ground_color)

            # Silhouetted tree trunks and foliage
            for tx in range(20, width, rng.randint(50, 95)):
                th = rng.randint(int(height*0.35), int(height*0.55))
                ttop = horizon_y - th
                draw.rectangle([tx, ttop, tx + 14, horizon_y + 40], fill=(20, 40, 20))
                # Tree canopy foliage
                for cy in range(ttop, ttop + int(th*0.6), 25):
                    cw = rng.randint(35, 65)
                    draw.ellipse([tx + 7 - cw, cy - 20, tx + 7 + cw, cy + 20], fill=(28, 65, 30))

        else:
            # Generic scenic rolling hills and mountain ridges
            mountain_pts = [(0, horizon_y)]
            for i in range(18):
                px = int(i * (width / 17))
                py = horizon_y - rng.randint(35, 110)
                mountain_pts.append((px, py))
            mountain_pts.append((width, horizon_y))
            draw.polygon(mountain_pts, fill=(int(horizon_color[0]*0.4), int(horizon_color[1]*0.35), int(horizon_color[2]*0.5)))
            draw.rectangle([0, horizon_y, width, height], fill=ground_color)

        # -------------------------------------------------------------
        # 4. RENDER DISTINCT FOREGROUND SUBJECT
        # -------------------------------------------------------------
        subj_x = int(width * 0.42)
        subj_y = horizon_y + int(height * 0.12)

        if is_dog:
            # Render a dynamic running dog
            dog_col = (180, 110, 45) # Golden retriever / tan canine
            dark_col = (120, 65, 25)
            
            # Torso (arched in running gallop)
            draw.ellipse([subj_x - 50, subj_y - 25, subj_x + 40, subj_y + 15], fill=dog_col)
            # Neck & Head
            draw.polygon([(subj_x + 25, subj_y - 10), (subj_x + 55, subj_y - 45), (subj_x + 75, subj_y - 25), (subj_x + 45, subj_y + 5)], fill=dog_col)
            draw.ellipse([subj_x + 45, subj_y - 50, subj_x + 78, subj_y - 20], fill=dog_col)
            # Snout
            draw.ellipse([subj_x + 65, subj_y - 38, subj_x + 95, subj_y - 22], fill=dark_col)
            # Floppy / Pointed Ear streaming back in wind
            draw.polygon([(subj_x + 46, subj_y - 48), (subj_x + 30, subj_y - 35), (subj_x + 50, subj_y - 30)], fill=dark_col)
            # Front Legs (extended forward in gallop)
            draw.line([(subj_x + 35, subj_y + 5), (subj_x + 65, subj_y + 35)], fill=dog_col, width=8)
            draw.line([(subj_x + 25, subj_y + 5), (subj_x + 50, subj_y + 38)], fill=dark_col, width=7)
            # Back Legs (extended backward in stride)
            draw.line([(subj_x - 35, subj_y), (subj_x - 65, subj_y + 25), (subj_x - 85, subj_y + 35)], fill=dog_col, width=8)
            draw.line([(subj_x - 25, subj_y), (subj_x - 55, subj_y + 28), (subj_x - 75, subj_y + 38)], fill=dark_col, width=7)
            # Tail (raised and streaming backward)
            draw.line([(subj_x - 45, subj_y - 15), (subj_x - 85, subj_y - 35)], fill=dog_col, width=7)

            # Sand / Water spray kicked up under paws
            spray_col = (255, 255, 255) if (is_beach or is_ocean) else (220, 180, 120)
            for _ in range(18):
                sp_x = subj_x + rng.randint(-95, -45)
                sp_y = subj_y + rng.randint(20, 42)
                draw.ellipse([sp_x, sp_y, sp_x + rng.randint(3, 7), sp_y + rng.randint(3, 7)], fill=spray_col)

        elif is_astronaut:
            # Render a full astronaut in spacesuit
            suit_col = (240, 242, 245)
            pack_col = (200, 205, 215)
            visor_col = (245, 185, 45) # Gold reflective visor

            # Life support backpack
            draw.rectangle([subj_x - 48, subj_y - 65, subj_x - 20, subj_y + 15], fill=pack_col)
            # Torso
            draw.rounded_rectangle([subj_x - 25, subj_y - 55, subj_x + 25, subj_y + 20], radius=10, fill=suit_col)
            # Helmet
            draw.ellipse([subj_x - 26, subj_y - 95, subj_x + 26, subj_y - 45], fill=suit_col)
            # Gold reflective visor
            draw.rounded_rectangle([subj_x - 12, subj_y - 85, subj_x + 24, subj_y - 58], radius=6, fill=visor_col)
            # Visor light reflection highlight
            draw.line([(subj_x - 5, subj_y - 80), (subj_x + 12, subj_y - 80)], fill=(255, 255, 255), width=2)
            # Legs in walking stride
            draw.line([(subj_x - 12, subj_y + 20), (subj_x - 25, subj_y + 75)], fill=suit_col, width=15)
            draw.line([(subj_x + 12, subj_y + 20), (subj_x + 28, subj_y + 70)], fill=suit_col, width=15)
            # Boots
            draw.rectangle([subj_x - 38, subj_y + 70, subj_x - 18, subj_y + 82], fill=(90, 95, 105))
            draw.rectangle([subj_x + 20, subj_y + 65, subj_x + 40, subj_y + 77], fill=(90, 95, 105))
            # Arms
            draw.line([(subj_x - 20, subj_y - 40), (subj_x - 40, subj_y - 5)], fill=suit_col, width=12)
            draw.line([(subj_x + 20, subj_y - 40), (subj_x + 38, subj_y - 10)], fill=suit_col, width=12)

            # Footprints in soil
            fp_col = (90, 25, 15) if is_mars else (30, 30, 40)
            for f_off in range(1, 5):
                fx = subj_x - f_off * 38
                fy = subj_y + 75 + f_off * 2
                draw.ellipse([fx, fy, fx + 16, fy + 7], fill=fp_col)

        elif is_car:
            # Render sleek sports car
            car_col = (235, 45, 40) if "red" in p_lower else (0, 190, 245)
            # Chassis lower body
            draw.rounded_rectangle([subj_x - 95, subj_y - 10, subj_x + 95, subj_y + 32], radius=8, fill=car_col)
            # Cabin / Windshield
            draw.polygon([(subj_x - 45, subj_y - 10), (subj_x - 20, subj_y - 35), (subj_x + 35, subj_y - 35), (subj_x + 65, subj_y - 10)], fill=(25, 25, 35))
            draw.polygon([(subj_x - 16, subj_y - 32), (subj_x + 30, subj_y - 32), (subj_x + 55, subj_y - 12), (subj_x - 10, subj_y - 12)], fill=(120, 210, 255))
            # Wheels
            draw.ellipse([subj_x - 70, subj_y + 12, subj_x - 30, subj_y + 52], fill=(20, 20, 25))
            draw.ellipse([subj_x + 30, subj_y + 12, subj_x + 70, subj_y + 52], fill=(20, 20, 25))
            draw.ellipse([subj_x - 56, subj_y + 26, subj_x - 44, subj_y + 38], fill=(200, 200, 200))
            draw.ellipse([subj_x + 44, subj_y + 26, subj_x + 56, subj_y + 38], fill=(200, 200, 200))
            # Headlights light beam casting forward
            draw.polygon([(subj_x + 90, subj_y + 5), (subj_x + 350, subj_y - 40), (subj_x + 350, subj_y + 65), (subj_x + 90, subj_y + 25)], fill=(255, 255, 200))
            # Taillight red glow trail behind
            draw.line([(subj_x - 95, subj_y + 8), (subj_x - 160, subj_y + 8)], fill=(255, 30, 30), width=4)

        elif is_robot:
            # Render a robot
            draw.rounded_rectangle([subj_x - 32, subj_y - 15, subj_x + 32, subj_y + 40], radius=8, fill=(35, 40, 50))
            draw.rounded_rectangle([subj_x - 24, subj_y - 50, subj_x + 24, subj_y - 20], radius=6, fill=(45, 52, 65))
            draw.rounded_rectangle([subj_x - 16, subj_y - 40, rx := subj_x + 16, subj_y - 30], radius=3, fill=(0, 240, 255))
            draw.rectangle([subj_x - 20, subj_y + 40, subj_x - 8, subj_y + 80], fill=(25, 30, 38))
            draw.rectangle([subj_x + 8, subj_y + 40, subj_x + 20, subj_y + 80], fill=(25, 30, 38))

        elif is_ship:
            # Render sailing ship
            hull_col = (75, 45, 25)
            draw.polygon([(subj_x - 90, subj_y + 5), (subj_x + 90, subj_y + 5), (subj_x + 60, subj_y + 35), (subj_x - 70, subj_y + 35)], fill=hull_col)
            draw.line([(subj_x - 20, subj_y + 5), (subj_x - 20, subj_y - 80)], fill=(45, 25, 15), width=4)
            draw.line([(subj_x + 30, subj_y + 5), (subj_x + 30, subj_y - 95)], fill=(45, 25, 15), width=4)
            # Sails
            draw.polygon([(subj_x - 18, subj_y - 75), (subj_x + 15, subj_y - 45), (subj_x - 18, subj_y - 15)], fill=(240, 235, 225))
            draw.polygon([(subj_x + 32, subj_y - 90), (subj_x + 75, subj_y - 55), (subj_x + 32, subj_y - 20)], fill=(240, 235, 225))

        else:
            # Generic stylized cinematic traveler silhouette
            draw.ellipse([subj_x - 16, subj_y - 60, subj_x + 16, subj_y - 28], fill=(20, 15, 25))
            draw.polygon([(subj_x - 22, subj_y - 28), (subj_x + 22, subj_y - 28), (subj_x + 30, subj_y + 35), (subj_x - 30, subj_y + 35)], fill=(15, 10, 20))
            draw.line([(subj_x - 12, subj_y + 35), (subj_x - 20, subj_y + 80)], fill=(15, 10, 20), width=10)
            draw.line([(subj_x + 12, subj_y + 35), (subj_x + 22, subj_y + 78)], fill=(15, 10, 20), width=10)

        # -------------------------------------------------------------
        # 5. CINEMATIC VIGNETTE & FILM LOOK
        # -------------------------------------------------------------
        img_np = np.array(img, dtype=np.float32)
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - width/2)**2 + (Y - height/2)**2)
        max_dist = np.sqrt((width/2)**2 + (height/2)**2)
        vignette = 1.0 - (dist_from_center / max_dist) * 0.32
        vignette = np.clip(vignette, 0.45, 1.0)
        img_np = img_np * vignette[..., np.newaxis]

        # Subtle natural film grain
        grain = rng.uniform(-4.0, 4.0)
        img_np = np.clip(img_np + grain, 0, 255)

        final_img = Image.fromarray(np.uint8(img_np))
        final_img.save(output_path, "JPEG", quality=95)
        return output_path
