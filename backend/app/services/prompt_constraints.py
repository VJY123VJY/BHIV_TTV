"""Deterministic prompt-adherence planning shared by image and video providers."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

from app.models.scene import Scene

_COLORS = ("red", "blue", "green", "yellow", "black", "white", "silver", "orange", "purple", "brown", "golden")
_WEATHER = ("heavy rain", "rain", "snow", "fog", "storm", "sunny", "overcast", "gentle wind", "wind")
_ENVIRONMENTS = (
    "lush green agricultural field",
    "agricultural field",
    "vegetable field",
    "snowy forest",
    "highway",
    "city street",
    "city",
    "farm",
    "field",
    "classroom",
    "kitchen",
    "park",
    "forest",
    "beach",
    "road",
    "savanna",
    "sky",
    "clouds",
    "ocean",
    "waterfall",
    "mountain",
)
_ACTIONS = (
    "driving",
    "running",
    "walking slowly",
    "walking",
    "flying",
    "speaking",
    "talking",
    "cooking",
    "riding",
    "working",
    "teaching",
    "grazing",
    "flowing",
    "rolling",
    "moving",
)


def _first_present(text: str, values: Iterable[str]) -> Optional[str]:
    return next((value for value in values if re.search(rf"\b{re.escape(value)}\b", text)), None)


@dataclass(frozen=True)
class PromptConstraints:
    subject: str
    action: str
    environment: str
    weather: Optional[str]
    time: Optional[str]
    camera: str
    style: str
    color: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def positive_prompt(self) -> str:
        conditions = ", ".join(item for item in (self.weather, self.time) if item)
        subject = f"{self.color} {self.subject}" if self.color and self.color not in self.subject else self.subject
        return (
            "Prompt adherence requirements — preserve every item in every frame: "
            f"primary subject: {subject}; action: {self.action}; environment: {self.environment}; "
            f"conditions: {conditions or 'natural conditions'}; camera: {self.camera}; style: {self.style}. "
            f"Keep the {subject} clearly visible and visually consistent."
        )

    def negative_prompt(self) -> str:
        prohibited = ["wrong subject", "wrong color", "wrong action", "wrong environment", "duplicate subject", "deformed object", "static scene", "unwanted camera cuts", "blurry", "text artifacts", "watermark"]
        if self.weather and "rain" in self.weather:
            prohibited.extend(["dry road", "sunny weather", "no visible rain"])
        if self.time == "night":
            prohibited.extend(["daylight", "bright midday sun"])
        if self.action == "driving":
            prohibited.extend(["parked vehicle", "stationary vehicle"])
        return ", ".join(prohibited)


class PromptConstraintService:
    def build(self, prompt: str, analysis: Optional[Dict[str, Any]] = None, style: str = "cinematic") -> PromptConstraints:
        text = prompt.lower()
        analysis = analysis or {}
        color = _first_present(text, _COLORS)
        action = _first_present(text, _ACTIONS) or str(analysis.get("action") or "moving")
        environment = _first_present(text, _ENVIRONMENTS) or str(analysis.get("setting") or "environment")
        weather = _first_present(text, _WEATHER)
        time = _first_present(text, ("night", "sunset", "sunrise", "morning", "evening", "day"))
        camera = _first_present(text, ("tracking shot", "close-up", "wide shot", "aerial shot", "handheld", "side tracking shot"))
        if not camera:
            camera = "smooth tracking shot" if action in {"driving", "running", "riding"} else "cinematic medium shot"
        subject_match = re.search(
            r"\b((?:(?:red|blue|green|yellow|black|white|silver|orange|golden)\s+)?(?:(?:sports|race|delivery|indian)\s+)?(?:sedan|suv|truck|car|motorcycle|bike|farmer|chef|student|woman|man|person|animal|dog|cat|retriever|airplane|plane|jet|elephant|waterfall|ball|astronaut|robot|bird))\b",
            text,
        )
        subject = subject_match.group(1) if subject_match else str(analysis.get("subject") or "subject")
        return PromptConstraints(subject, action, environment, weather, time, camera, style, color)

    def apply_to_scenes(self, scenes: List[Scene], constraints: PromptConstraints) -> List[Scene]:
        positive, negative = constraints.positive_prompt(), constraints.negative_prompt()
        for scene in scenes:
            scene.visual_description = f"{scene.visual_description}\n{positive}\nAvoid: {negative}."
            scene.metadata["prompt_constraints"] = constraints.to_dict()
            scene.metadata["negative_prompt"] = negative
            if scene.metadata.get("enriched_prompt"):
                scene.metadata["enriched_prompt"] = f"{scene.metadata['enriched_prompt']}\n{positive}\nNegative prompt: {negative}."
        return scenes


prompt_constraint_service = PromptConstraintService()
