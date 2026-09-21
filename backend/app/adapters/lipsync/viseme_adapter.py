"""
Audio-driven viseme lip-sync.

Detects a visible face, measures per-frame speech energy from the dialogue track,
and deforms the mouth region so mouth opening tracks the spoken audio.

This is real mouth-motion lip-sync (not A/V muxing). It is the offline default
when a neural Wav2Lip checkpoint is not configured.
"""
from __future__ import annotations

import math
import os
import struct
import subprocess
import wave
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

from app.adapters.lipsync.base import BaseLipSyncAdapter
from app.core.logging import telemetry


class VisemeLipSyncAdapter(BaseLipSyncAdapter):
    def __init__(self):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None

    async def sync_clip(self, video_path: str, audio_path: str, output_path: str) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(video_path):
            return {"applied": False, "reason": "missing_video", "output_path": video_path}
        if not audio_path or not os.path.exists(audio_path):
            return {"applied": False, "reason": "missing_audio", "output_path": video_path}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"applied": False, "reason": "unreadable_video", "output_path": video_path}

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        envelope = self._audio_envelope(audio_path, fps, max(frame_count, 1))
        first_face = None
        frames: List[np.ndarray] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            face = self._detect_face(frame)
            if face is not None:
                first_face = face
                frame = self._animate_mouth(frame, face, envelope[idx] if idx < len(envelope) else 0.0)
            frames.append(frame)
            idx += 1
        cap.release()

        if not frames:
            return {"applied": False, "reason": "empty_video", "output_path": video_path}
        if first_face is None:
            telemetry.emit("lipsync_skipped_no_face", "lipsync", {"video": os.path.basename(video_path)})
            return {"applied": False, "reason": "no_visible_face", "output_path": video_path}

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not writer.isOpened():
            return {"applied": False, "reason": "writer_failed", "output_path": video_path}
        for frame in frames:
            writer.write(frame)
        writer.release()
        return {
            "applied": True,
            "reason": "viseme_mouth_animation",
            "output_path": output_path,
            "engine": "viseme",
        }

    def _detect_face(self, frame) -> Optional[Tuple[int, int, int, int]]:
        if self.face_cascade is None or self.face_cascade.empty():
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
        if len(faces) == 0:
            return None
        return tuple(max(faces, key=lambda f: f[2] * f[3]))

    def _animate_mouth(self, frame, face, open_amount: float):
        x, y, w, h = face
        mouth_y = y + int(h * 0.58)
        mouth_h = max(8, int(h * 0.32))
        mouth_x = x + int(w * 0.22)
        mouth_w = max(8, int(w * 0.56))
        y2 = min(frame.shape[0], mouth_y + mouth_h)
        x2 = min(frame.shape[1], mouth_x + mouth_w)
        roi = frame[mouth_y:y2, mouth_x:x2]
        if roi.size == 0:
            return frame
        stretch = 1.0 + (0.55 * max(0.0, min(1.0, open_amount)))
        new_h = max(1, int(roi.shape[0] * stretch))
        stretched = cv2.resize(roi, (roi.shape[1], new_h), interpolation=cv2.INTER_LINEAR)
        crop = stretched[: roi.shape[0], : roi.shape[1]]
        if crop.shape[0] < roi.shape[0]:
            pad = np.repeat(crop[-1:, :, :], roi.shape[0] - crop.shape[0], axis=0)
            crop = np.concatenate([crop, pad], axis=0)
        blend = 0.35 + 0.55 * max(0.0, min(1.0, open_amount))
        frame[mouth_y:y2, mouth_x:x2] = cv2.addWeighted(crop[: roi.shape[0], : roi.shape[1]], blend, roi, 1.0 - blend, 0)
        return frame

    def _audio_envelope(self, audio_path: str, fps: float, n_frames: int) -> List[float]:
        samples, sr = self._load_pcm(audio_path)
        if not samples:
            return [0.0] * n_frames
        window = max(1, int(sr / max(fps, 1.0)))
        envelope = []
        for i in range(n_frames):
            start = i * window
            chunk = samples[start:start + window]
            if not chunk:
                envelope.append(0.0)
                continue
            rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
            envelope.append(min(1.0, rms * 4.0))
        peak = max(envelope) or 1.0
        return [v / peak for v in envelope]

    def _load_pcm(self, audio_path: str) -> Tuple[List[float], int]:
        wav_path = audio_path
        tmp = None
        if not audio_path.lower().endswith(".wav"):
            tmp = audio_path + ".lipsync.wav"
            cmd = ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", tmp]
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                wav_path = tmp
            except Exception:
                return [], 16000
        try:
            with wave.open(wav_path, "rb") as wf:
                sr = wf.getframerate()
                n = wf.getnframes()
                raw = wf.readframes(n)
                width = wf.getsampwidth()
                if width == 2:
                    samples = [s / 32768.0 for s in struct.unpack("<" + "h" * (len(raw) // 2), raw)]
                else:
                    samples = [((b - 128) / 128.0) for b in raw]
                return samples, sr
        except Exception:
            return [], 16000
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
