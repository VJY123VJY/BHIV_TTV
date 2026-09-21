import os
import wave
import struct
import math

import cv2
import numpy as np
import pytest

from app.adapters.lipsync.viseme_adapter import VisemeLipSyncAdapter


def _write_sine_wav(path, seconds=1.0, sr=16000):
    n = int(sr * seconds)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = []
        for i in range(n):
            val = int(8000 * math.sin(2 * math.pi * 180 * (i / sr)))
            frames.append(struct.pack("<h", val))
        wf.writeframes(b"".join(frames))


def _write_clip(path, with_face: bool, size=(160, 160), frames=12):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12, size)
    for _ in range(frames):
        frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        if with_face:
            cv2.ellipse(frame, (80, 70), (40, 50), 0, 0, 360, (190, 170, 150), -1)
            cv2.circle(frame, (65, 60), 6, (20, 20, 20), -1)
            cv2.circle(frame, (95, 60), 6, (20, 20, 20), -1)
            cv2.ellipse(frame, (80, 90), (16, 8), 0, 0, 360, (40, 20, 20), -1)
        writer.write(frame)
    writer.release()


@pytest.mark.asyncio
async def test_lipsync_applies_when_face_present(tmp_path):
    video = str(tmp_path / "face.mp4")
    audio = str(tmp_path / "voice.wav")
    out = str(tmp_path / "synced.mp4")
    _write_clip(video, with_face=True)
    _write_sine_wav(audio)
    adapter = VisemeLipSyncAdapter()
    adapter._detect_face = lambda frame: (40, 30, 80, 100)
    result = await adapter.sync_clip(video, audio, out)
    assert result["applied"] is True
    assert os.path.exists(result["output_path"])


@pytest.mark.asyncio
async def test_lipsync_skips_when_no_face(tmp_path):
    video = str(tmp_path / "landscape.mp4")
    audio = str(tmp_path / "voice.wav")
    out = str(tmp_path / "synced.mp4")
    _write_clip(video, with_face=False)
    _write_sine_wav(audio)
    result = await VisemeLipSyncAdapter().sync_clip(video, audio, out)
    assert result["applied"] is False
    assert result["reason"] == "no_visible_face"
