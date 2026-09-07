import pytest
import os
from pathlib import Path
from app.adapters.llm.local_adapter import LocalLLMAdapter
from app.adapters.vision.consistency_adapter import VisualConsistencyAdapter
from app.adapters.image.local_image_adapter import LocalImageAdapter
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.adapters.tts.local_tts_adapter import LocalTTSAdapter
from app.models.scene import Scene
from app.utils.media_check import validate_video_file

@pytest.mark.asyncio
async def test_llm_local_adapter():
    adapter = LocalLLMAdapter()
    analysis = await adapter.analyze_prompt("A lone astronaut walks on Mars during red sunset.")
    assert "astronaut" in analysis["subject"]
    assert "planet" in analysis["setting"] or "mars" in analysis.get("raw_prompt", "").lower()

    story = await adapter.generate_story("A lone astronaut walks on Mars.", analysis, 15)
    assert "acts" in story
    assert len(story["acts"]) == 3

    scenes = await adapter.generate_scenes(story, 15, "cinematic")
    assert len(scenes) >= 3
    assert scenes[0].duration > 0

def test_visual_consistency_adapter():
    adapter = VisualConsistencyAdapter()
    scenes = [
        Scene(index=1, title="Scene 1", narrative="Walking", visual_description="A robot walks", duration=5.0),
        Scene(index=2, title="Scene 2", narrative="Stopping", visual_description="A robot stops", duration=5.0)
    ]
    enriched = adapter.enrich_scene_prompts(scenes, "cinematic")
    assert len(enriched) == 2
    assert "enriched_prompt" in enriched[0].metadata
    assert "cinematic" in enriched[0].metadata["enriched_prompt"].lower()

@pytest.mark.asyncio
async def test_image_local_adapter(tmp_path):
    adapter = LocalImageAdapter()
    out_img = str(tmp_path / "test_image.jpg")
    result = await adapter.generate_image("A futuristic city at sunset", out_img, "cinematic", width=640, height=360)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 5000

@pytest.mark.asyncio
async def test_video_local_adapter(tmp_path):
    img_adapter = LocalImageAdapter()
    vid_adapter = OpenCVVideoAdapter()

    img_path = str(tmp_path / "test_keyframe.jpg")
    await img_adapter.generate_image("A small robot in a city", img_path, width=320, height=240)

    scene = Scene(index=1, title="Test Scene", narrative="Exploring", visual_description="Desc", duration=2.0, camera_motion="pan_right")
    vid_path = str(tmp_path / "test_clip.mp4")

    await vid_adapter.generate_scene_video(scene, img_path, vid_path, width=320, height=240, fps=15)
    assert os.path.exists(vid_path)
    assert os.path.getsize(vid_path) > 1000

    is_valid, info = validate_video_file(vid_path)
    assert is_valid
    assert info["frame_count"] > 0

@pytest.mark.asyncio
async def test_tts_local_adapter(tmp_path):
    tts = LocalTTSAdapter()
    out_audio = str(tmp_path / "test_voice.wav")
    result = await tts.synthesize_speech("Hello world, this is a test narration.", out_audio)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 500
