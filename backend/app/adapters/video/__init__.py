"""
Video generation adapters.
"""
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.adapters.video.external_video_adapter import ExternalVideoAdapter
from app.core.config import settings

def get_video_adapter(provider: str = "opencv"):
    if getattr(settings, "MODEL_MODE", "base").lower() == "finetuned":
        from app.adapters.video.neural_video_adapter import NeuralVideoAdapter
        return NeuralVideoAdapter()
    p = (provider or "opencv").lower()
    if p == "external":
        return ExternalVideoAdapter()
    if p == "wan":
        # Imported lazily: diffusers/torch are NOT required unless VIDEO_PROVIDER=wan
        from app.adapters.video.wan_video_adapter import WanVideoAdapter
        return WanVideoAdapter()
    return OpenCVVideoAdapter()

