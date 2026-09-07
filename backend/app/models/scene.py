from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class Scene:
    """Represents a single visual and auditory scene in the storyboard."""
    index: int
    title: str
    narrative: str
    visual_description: str
    duration: float
    camera_motion: str = "pan_right" # pan_left, pan_right, zoom_in, zoom_out, tilt_up, tilt_down
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "narrative": self.narrative,
            "visual_description": self.visual_description,
            "duration": self.duration,
            "camera_motion": self.camera_motion,
            "image_path": self.image_path,
            "video_path": self.video_path,
            "audio_path": self.audio_path,
            "metadata": self.metadata
        }
