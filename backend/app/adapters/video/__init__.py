"""
Video generation adapters.
"""
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.adapters.video.external_video_adapter import ExternalVideoAdapter

def get_video_adapter(provider: str = "opencv"):
    p = (provider or "opencv").lower()
    if p == "external":
        return ExternalVideoAdapter()
    return OpenCVVideoAdapter()
