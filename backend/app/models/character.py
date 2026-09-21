"""
Character consistency data models: profile, reference images, embeddings, voice, and language profiles.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class VoiceProfile:
    voice_id: str
    language: str = "mr-IN"
    gender: str = "male"  # male | female
    style: str = "friendly"
    provider: str = "edge"
    consent_status: str = "confirmed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LanguageProfile:
    default_language: str = "mr"
    allowed_languages: List[str] = field(default_factory=lambda: ["mr", "hi", "en"])
    accent_locale: str = "mr-IN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CharacterReference:
    reference_id: str
    image_path: str
    embedding_vector: Optional[List[float]] = None
    face_bbox: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CharacterProfile:
    character_id: str
    name: str
    appearance_prompt: str
    age: int = 30
    gender: str = "male"
    ethnicity: str = "Indian"
    wardrobe: str = "blue cotton shirt, brown trousers"
    distinguishing_features: str = "short black hair, warm engaging smile, expressive eyes"
    reference_images: List[str] = field(default_factory=list)
    voice_id: str = "mr-IN-ManoharNeural"
    default_language: str = "mr"
    voice_profile: Optional[VoiceProfile] = None
    language_profile: Optional[LanguageProfile] = None

    def get_visual_anchor(self) -> str:
        """Returns visual prompt anchor string guaranteeing cross-scene character identity lock."""
        return (
            f"{self.name}, a {self.age}-year-old {self.ethnicity} {self.gender}, "
            f"{self.distinguishing_features}, wearing {self.wardrobe}. "
            f"Consistent facial structure, skin tone, hairstyle, and outfit across every scene."
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["visual_anchor"] = self.get_visual_anchor()
        return d
