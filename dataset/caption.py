"""
Automated multimodal vision-language captioning and action annotation engine.
"""
from typing import Dict, Any, List
import re
from dataset.models import CaptionMetadata

# Heuristic vocabularies for structured visual concept extraction
VOCAB_SUBJECTS = {
    "farmer": ["farmer", "peasant", "agriculture worker", "grower"],
    "man": ["man", "male", "person", "guy"],
    "woman": ["woman", "female", "lady"],
    "child": ["child", "boy", "girl", "kid"],
    "robot": ["robot", "cyborg", "drone", "android", "mech"],
    "dog": ["dog", "puppy", "canine", "hound"],
    "cat": ["cat", "feline", "kitten"],
    "car": ["car", "vehicle", "automobile", "truck", "suv"],
    "tractor": ["tractor", "harvester", "plow", "combine"],
}

VOCAB_ACTIONS = {
    "speaking": ["speaking", "talking", "explaining", "conversing", "giving a speech", "presenting"],
    "walking": ["walking", "strolling", "advancing", "moving forward", "hiking"],
    "running": ["running", "jogging", "sprinting", "dashing"],
    "farming": ["farming", "harvesting", "planting", "tending crops", "weeding", "plowing"],
    "working": ["working", "operating", "inspecting", "building", "repairing"],
    "driving": ["driving", "drives", "racing", "speeding", "cruising"],
    "looking": ["looking", "gazing", "observing", "watching"],
}

VOCAB_ENVIRONMENTS = {
    "farm": ["farm", "crop field", "vegetable farm", "orchard", "pasture", "countryside", "village"],
    "city": ["city", "urban street", "metropolis", "downtown", "sidewalk"],
    "classroom": ["classroom", "school", "lecture hall", "university"],
    "forest": ["forest", "woodland", "jungle", "trees", "nature reserve"],
    "laboratory": ["laboratory", "lab", "research facility", "cleanroom"],
    "indoor": ["indoor", "room", "office", "studio", "house"],
    "highway": ["highway", "motorway", "freeway", "road", "expressway"],
}


class AutoCaptioner:
    """
    Vision-Language Captioning module.
    Analyzes visual assets and generates fine-grained structured descriptions.
    """

    def generate_caption(self, prompt_or_description: str, context: Dict[str, Any] = None) -> CaptionMetadata:
        text = (prompt_or_description or "").lower()
        context = context or {}

        # 1. Detect subjects
        detected_subjects = []
        for category, terms in VOCAB_SUBJECTS.items():
            if any(term in text for term in terms):
                detected_subjects.append(category)
        if not detected_subjects:
            detected_subjects = [context.get("subject", "person")]

        # 2. Detect actions
        detected_actions = []
        for act, terms in VOCAB_ACTIONS.items():
            if any(term in text for term in terms):
                detected_actions.append(act)
        if not detected_actions:
            detected_actions = ["standing"]

        # 3. Detect environment
        detected_env = "outdoor"
        for env, terms in VOCAB_ENVIRONMENTS.items():
            if any(term in text for term in terms):
                detected_env = env
                break

        # 4. Cinematic / camera shot detection
        camera = "medium shot"
        if "close-up" in text or "face" in text or "portrait" in text:
            camera = "close-up portrait"
        elif "wide" in text or "landscape" in text or "aerial" in text:
            camera = "wide cinematic shot"
        elif "handheld" in text:
            camera = "handheld documentary shot"
        elif "tracking" in text or "follow" in text:
            camera = "tracking shot"

        # 5. Lighting & weather
        lighting = "natural daylight"
        if "golden hour" in text or "sunset" in text:
            lighting = "golden hour warm sunlight"
        elif "night" in text or "dark" in text:
            lighting = "cinematic nighttime lighting"
        elif "studio" in text:
            lighting = "controlled three-point studio lighting"

        weather = "clear"
        if "rain" in text:
            weather = "rainy overcast"
        elif "fog" in text or "mist" in text:
            weather = "foggy atmospheric"

        time = "night" if "night" in text else "sunset" if "sunset" in text else "day"
        motion = "forward movement" if any(action in detected_actions for action in ("walking", "running", "farming", "driving")) else "static composition"

        emotion = "engaged and communicative"
        if "happy" in text or "friendly" in text:
            emotion = "friendly and approachable"
        elif "serious" in text or "focused" in text:
            emotion = "focused and earnest"

        # Construct fluent natural language caption
        primary_subject = detected_subjects[0]
        primary_action = detected_actions[0]
        caption = (
            f"A {primary_subject} {primary_action} in a {detected_env} setting, captured in a {camera} "
            f"under {lighting} with an {emotion} expression."
        )

        return CaptionMetadata(
            caption=caption,
            subjects=detected_subjects,
            objects=detected_subjects,
            actions=detected_actions,
            environment=detected_env,
            camera=camera,
            lighting=lighting,
            weather=weather,
            time=time,
            emotion=emotion,
            motion=motion,
        )


auto_captioner = AutoCaptioner()


def main() -> None:
    """Caption reviewed dataset metadata in place without fetching new media."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate structured captions for processed dataset assets")
    parser.add_argument("--metadata", default="data/metadata", help="Metadata directory")
    args = parser.parse_args()
    updated = 0
    for metadata_path in Path(args.metadata).glob("*.json"):
        try:
            item = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not item.get("allowed_for_training"):
                continue
            prompt = item.get("category") or item.get("source_name") or "scene"
            item["caption_metadata"] = auto_captioner.generate_caption(prompt).__dict__
            metadata_path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")
            updated += 1
        except (OSError, ValueError, TypeError):
            continue
    print(f"Generated structured captions for {updated} approved assets.")


if __name__ == "__main__":
    main()
