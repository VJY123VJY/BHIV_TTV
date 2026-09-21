"""
Video dataset ingestion: scene detection, shot splitting, keyframe extraction, and clip chunking.
"""
import os
import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataset.models import QualityMetrics


class VideoProcessor:
    """
    Processes raw video assets into standardized training chunks with keyframe and motion metadata.
    """

    SUPPORTED_CLIP_DURATIONS = [2, 4, 6, 8, 12]

    def inspect_video(self, video_path: str) -> Dict[str, Any]:
        """Extracts technical metadata from a video file."""
        if not os.path.exists(video_path):
            return {"valid": False, "error": "File does not exist"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"valid": False, "error": "Could not open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = round(frame_count / fps, 2) if fps > 0 else 0.0

        cap.release()
        has_audio = False
        # OpenCV does not expose audio streams. ffprobe is used when available;
        # unavailable probing is represented as false/unknown rather than true.
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
                capture_output=True, text=True, check=False, timeout=15,
            )
            has_audio = "audio" in result.stdout
        except (OSError, subprocess.SubprocessError):
            pass

        return {
            "valid": True,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "aspect_ratio": "16:9" if width >= height else "9:16",
            "has_audio": has_audio,
            "speech_detected": False,  # requires an explicitly configured ASR/VAD provider
            "faces_detected": False,   # filled by optional downstream face annotation
        }

    def detect_scenes(self, video_path: str, threshold: float = 30.0) -> List[Tuple[int, int]]:
        """
        Detects scene shot changes via frame luminance difference.
        Returns list of (start_frame, end_frame) intervals.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        scenes = []
        prev_gray = None
        start_frame = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = np.mean(cv2.absdiff(gray, prev_gray))
                if diff > threshold and (frame_idx - start_frame) > 24:
                    scenes.append((start_frame, frame_idx - 1))
                    start_frame = frame_idx

            prev_gray = gray
            frame_idx += 1

        if frame_idx > start_frame:
            scenes.append((start_frame, frame_idx - 1))

        cap.release()
        return scenes

    def extract_keyframes(self, video_path: str, max_keyframes: int = 5) -> List[np.ndarray]:
        """Extracts evenly spaced keyframe images across the video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            cap.release()
            return []

        step = max(1, frame_count // (max_keyframes + 1))
        keyframes = []

        for i in range(1, max_keyframes + 1):
            target = i * step
            if target >= frame_count:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if ret and frame is not None:
                keyframes.append(frame)

        cap.release()
        return keyframes

    def chunk_video(
        self,
        video_path: str,
        output_dir: str,
        target_duration: int = 4,
    ) -> List[str]:
        """
        Splits a video into standardized training clips (e.g. 2s, 4s, 6s, 8s, 12s).
        Returns paths to created clip files.
        """
        os.makedirs(output_dir, exist_ok=True)
        info = self.inspect_video(video_path)
        if not info.get("valid") or info.get("duration", 0) < target_duration:
            return []

        fps = info["fps"]
        width = info["width"]
        height = info["height"]
        frames_per_clip = int(fps * target_duration)

        cap = cv2.VideoCapture(video_path)
        clip_paths = []
        clip_idx = 1
        current_frames = []

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_frames.append(frame)
            if len(current_frames) >= frames_per_clip:
                out_path = os.path.join(output_dir, f"clip_{clip_idx:03d}_{target_duration}s.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))
                for f in current_frames:
                    writer.write(f)
                writer.release()
                clip_paths.append(out_path)
                clip_idx += 1
                current_frames = []

        cap.release()
        return clip_paths

    def process_video(self, video_path: str, output_dir: str, target_duration: int = 4) -> Dict[str, Any]:
        """Create scene/keyframe/clip artifacts and return traceable metadata.

        Speech and speaker identity require configured ASR/diarization models and
        remain explicitly false until such a provider annotates the asset.
        """
        if target_duration not in self.SUPPORTED_CLIP_DURATIONS:
            raise ValueError(f"target_duration must be one of {self.SUPPORTED_CLIP_DURATIONS}")
        output = Path(output_dir)
        keyframe_dir = output / "keyframes"
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        info = self.inspect_video(video_path)
        if not info.get("valid"):
            return info
        scenes = self.detect_scenes(video_path)
        keyframes = self.extract_keyframes(video_path)
        keyframe_paths = []
        for index, frame in enumerate(keyframes, 1):
            path = keyframe_dir / f"keyframe_{index:03d}.jpg"
            cv2.imwrite(str(path), frame)
            keyframe_paths.append(str(path))
        clips = self.chunk_video(video_path, str(output / "clips"), target_duration)
        return {
            **info,
            "scene_count": len(scenes),
            "scene_ranges": scenes,
            "keyframes": keyframe_paths,
            "clips": clips,
            "clip_duration": target_duration,
        }


video_processor = VideoProcessor()
