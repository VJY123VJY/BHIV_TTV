"""
Character Consistency Service.
Maintains identity anchors, reference embeddings, wardrobe locks, and voice profiles across multi-scene video generations.
"""
from typing import Dict, Any, Optional, List
from app.models.character import CharacterProfile, VoiceProfile, LanguageProfile
from app.models.scene import Scene


class CharacterService:
    def __init__(self):
        self._profiles: Dict[str, CharacterProfile] = {}
        self._seed_builtin_profiles()

    def _seed_builtin_profiles(self):
        # 1. Rahul (Farmer) - specifically requested for acceptance test
        self.register_profile(
            CharacterProfile(
                character_id="rahul_farmer",
                name="Rahul",
                age=28,
                gender="male",
                ethnicity="Indian",
                appearance_prompt="Indian male, 28 years old, short black hair, medium skin tone, blue cotton shirt, friendly approachable demeanor",
                wardrobe="traditional durable blue cotton collared shirt and neutral earthy trousers",
                distinguishing_features="short neat black hair, medium Indian skin tone, warm honest smile, expressive communicative eyes",
                default_language="mr",
                voice_id="mr-IN-ManoharNeural",
                voice_profile=VoiceProfile(
                    voice_id="mr-IN-ManoharNeural",
                    language="mr-IN",
                    gender="male",
                    style="friendly",
                    provider="edge",
                ),
                language_profile=LanguageProfile(
                    default_language="mr",
                    allowed_languages=["mr", "hi", "en", "gu"],
                    accent_locale="mr-IN",
                ),
            )
        )

        # 2. Priya (Teacher / Educator)
        self.register_profile(
            CharacterProfile(
                character_id="priya_teacher",
                name="Priya",
                age=32,
                gender="female",
                ethnicity="Indian",
                appearance_prompt="Indian female, 32 years old, long black hair tied neatly, glasses, elegant teal cotton kurta",
                wardrobe="teal cotton kurta with subtle embroidery and dark trousers",
                distinguishing_features="gentle intelligent expression, dark eyes, polite professional posture",
                default_language="hi",
                voice_id="hi-IN-SwaraNeural",
                voice_profile=VoiceProfile(
                    voice_id="hi-IN-SwaraNeural",
                    language="hi-IN",
                    gender="female",
                    style="warm_professional",
                    provider="edge",
                ),
            )
        )

        # 3. Alex (Scientist / Explorer)
        self.register_profile(
            CharacterProfile(
                character_id="alex_explorer",
                name="Alex",
                age=30,
                gender="male",
                ethnicity="ambiguous",
                appearance_prompt="30 year old explorer in a high-tech field jacket, focused and observant",
                wardrobe="dark navy technical expedition jacket with brass zippers",
                distinguishing_features="keen observant gaze, athletic build, short styled hair",
                default_language="en",
                voice_id="en-US-GuyNeural",
            )
        )

    def register_profile(self, profile: CharacterProfile) -> None:
        self._profiles[profile.character_id.lower()] = profile

    def get_profile(self, character_id_or_name: Optional[str]) -> Optional[CharacterProfile]:
        if not character_id_or_name:
            return None
        key = character_id_or_name.strip().lower()
        if key in self._profiles:
            return self._profiles[key]
        for pid, p in self._profiles.items():
            if key in pid or key in p.name.lower():
                return p
        return None

    def list_profiles(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._profiles.values()]

    def apply_character_consistency(
        self,
        scenes: List[Scene],
        character_id: Optional[str],
        language: str = "en",
    ) -> List[Scene]:
        """
        Enriches each scene's visual prompt with the persistent character anchor
        and guarantees identical character identity throughout every keyframe.
        """
        char = self.get_profile(character_id)
        if not char:
            return scenes

        anchor = char.get_visual_anchor()

        for scene in scenes:
            scene.metadata["character_id"] = char.character_id
            scene.metadata["character_name"] = char.name

            # Inject character visual description into visual prompt
            curr_desc = scene.visual_description or ""
            if char.name not in curr_desc:
                scene.visual_description = f"{anchor}. Scene action: {curr_desc}"

            # Preserve character identity in enriched prompt
            if "enriched_prompt" in scene.metadata:
                scene.metadata["enriched_prompt"] = f"{anchor}. {scene.metadata['enriched_prompt']}"

        return scenes


character_service = CharacterService()
