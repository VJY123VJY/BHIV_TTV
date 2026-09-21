import pytest
import os
from app.pipelines.text_to_video import pipeline
from app.core.exceptions import ValidationError

@pytest.mark.asyncio
async def test_pipeline_validation_error():
    with pytest.raises(ValidationError):
        await pipeline.execute(prompt="")

@pytest.mark.asyncio
async def test_pipeline_short_prompt_error():
    with pytest.raises(ValidationError):
        await pipeline.execute(prompt="hi")

@pytest.mark.asyncio
async def test_pipeline_execution_success(tmp_path):
    # Test short duration for fast test cycle
    result = await pipeline.execute(
        prompt="A scientist in a clean laboratory observing glowing crystals.",
        duration=6,
        style="cinematic",
        voice=True
    )
    assert result["status"] == "success"
    assert "metadata" in result
    meta = result["metadata"]
    assert "video_url" in meta
    assert len(meta["scenes"]) >= 2
    assert meta["resolution"] == "1280x720"


@pytest.mark.asyncio
async def test_pipeline_vertical_high_settings():
    result = await pipeline.execute(
        prompt="A farmer walking through a green vegetable farm",
        duration=6,
        style="realistic",
        voice=True,
        aspect_ratio="9:16",
        quality="high",
        language="en",
        lipsync=True,
    )
    meta = result["metadata"]
    assert meta["resolution"] == "1080x1920"
    assert meta["aspect_ratio"] == "9:16"
    assert meta["quality"] == "high"
    assert meta["validation"]["width"] == 1080
    assert meta["validation"]["height"] == 1920
