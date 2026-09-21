import pytest
import os
import shutil
from app.services.ffmpeg_service import ffmpeg_service
from app.services.audio_service import audio_service
from app.models.scene import Scene
from app.adapters.image.local_image_adapter import LocalImageAdapter
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.utils.media_check import validate_video_file

def test_ffmpeg_installed():
    assert shutil.which("ffmpeg") is not None, "FFmpeg binary must be installed and accessible in PATH"

@pytest.mark.asyncio
async def test_ffmpeg_assembly_pipeline(tmp_path):
    # 1. Create 2 test scene video clips
    img_adapter = LocalImageAdapter()
    vid_adapter = OpenCVVideoAdapter()

    scenes = []
    for i in [1, 2]:
        img_p = str(tmp_path / f"scene_{i}.jpg")
        await img_adapter.generate_image(f"Scene {i} in city", img_p, width=320, height=240)
        
        scene = Scene(index=i, title=f"Scene {i}", narrative=f"Narrative for scene {i}", visual_description="Desc", duration=2.0)
        vid_p = str(tmp_path / f"scene_{i}.mp4")
        await vid_adapter.generate_scene_video(scene, img_p, vid_p, width=320, height=240, fps=15)
        scene.video_path = vid_p
        scenes.append(scene)

    # 2. Mix audio
    audio_p = str(tmp_path / "mixed_audio.wav")
    final_audio = audio_service.mix_complete_audio(scenes, 4.0, audio_p)
    assert os.path.exists(final_audio)

    # 3. Assemble final video via FFmpeg
    final_mp4 = str(tmp_path / "final_assembled.mp4")
    assembled = ffmpeg_service.assemble_final_video(
        scenes, final_audio, final_mp4, "test_exec_001", width=320, height=240, fps=15
    )
    assert os.path.exists(assembled)

    # 4. Media validation
    is_valid, info = validate_video_file(assembled)
    assert is_valid
    assert info["frame_count"] > 0
    assert info["duration"] > 0
    assert info["width"] == 320
    assert info["height"] == 240


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "width,height",
    [
        (1280, 720),
        (1920, 1080),
        (3840, 2160),
        (720, 1280),
        (1080, 1920),
        (2160, 3840),
    ],
)
async def test_ffmpeg_respects_aspect_and_quality(tmp_path, width, height):
    img_adapter = LocalImageAdapter()
    vid_adapter = OpenCVVideoAdapter()
    img_p = str(tmp_path / "scene.jpg")
    await img_adapter.generate_image("A farm path", img_p, width=160, height=160)
    scene = Scene(index=1, title="Scene", narrative="Walking", visual_description="Farm", duration=1.0)
    vid_p = str(tmp_path / "scene.mp4")
    await vid_adapter.generate_scene_video(scene, img_p, vid_p, width=160, height=160, fps=8)
    scene.video_path = vid_p
    audio_p = str(tmp_path / "audio.wav")
    audio_service.mix_complete_audio([scene], 1.0, audio_p)
    final_mp4 = str(tmp_path / f"final_{width}x{height}.mp4")
    assembled = ffmpeg_service.assemble_final_video(
        [scene], audio_p, final_mp4, f"dim_{width}_{height}", width=width, height=height, fps=8
    )
    is_valid, info = validate_video_file(assembled)
    assert is_valid
    assert info["width"] == width
    assert info["height"] == height
