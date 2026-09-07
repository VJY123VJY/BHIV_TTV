import os
import cv2
from typing import Dict, Any, Tuple

def validate_video_file(file_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Rigorously validate that a video file exists, is readable, and is a valid MP4 container.
    """
    if not os.path.exists(file_path):
        return False, {"error": f"File does not exist: {file_path}"}

    file_size = os.path.getsize(file_path)
    if file_size < 1000:
        return False, {"error": f"File too small ({file_size} bytes), likely corrupt or dummy mock"}

    # Check MP4 container magic header bytes
    with open(file_path, "rb") as f:
        header = f.read(16)
        if b"ftyp" not in header:
            return False, {"error": "Invalid MP4 header (missing 'ftyp' atom)"}

    # Open with OpenCV to verify actual stream decoding
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False, {"error": "Failed to decode video stream via OpenCV"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps and fps > 0 else 0

    # Read the first frame to confirm decodability
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, {"error": "OpenCV failed to read initial frame from video stream"}

    return True, {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration": round(duration, 2),
        "file_size_bytes": file_size
    }
