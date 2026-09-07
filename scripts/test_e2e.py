#!/usr/bin/env python3
import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.pipelines.text_to_video import pipeline
from app.utils.media_check import validate_video_file

async def run_end_to_end_test():
    prompt = "A small robot explores a futuristic city at sunset."
    print("==================================================================")
    print("STARTING END-TO-END PIPELINE VERIFICATION")
    print(f"Target Prompt: '{prompt}'")
    print("==================================================================")

    def progress_callback(stage, pct):
        print(f"  [Progress {pct:3d}%] -> Stage: {stage}")

    result = await pipeline.execute(
        prompt=prompt,
        duration=15,
        style="cinematic",
        voice=True,
        progress_callback=progress_callback
    )

    print("\n------------------------------------------------------------------")
    print("EXECUTION FINISHED. VERIFYING ARTIFACTS...")
    print("------------------------------------------------------------------")
    
    assert result["status"] == "success", f"Pipeline status failed: {result}"
    metadata = result["metadata"]
    video_id = result["execution_id"]
    
    video_path = BASE_DIR / "generated" / "videos" / f"{video_id}.mp4"
    meta_path = BASE_DIR / "generated" / "videos" / f"{video_id}_metadata.json"

    print(f"Checking Video File: {video_path}")
    assert os.path.exists(video_path), f"Final MP4 not found at {video_path}"
    
    print(f"Checking Metadata File: {meta_path}")
    assert os.path.exists(meta_path), f"Metadata JSON sidecar not found at {meta_path}"

    is_valid, media_info = validate_video_file(str(video_path))
    print("\nMedia Verification Results:")
    for k, v in media_info.items():
        print(f"  - {k}: {v}")

    assert is_valid, f"Media validation failed: {media_info}"
    assert media_info["frame_count"] > 0, "No frames decoded from MP4"
    assert media_info["duration"] >= 10.0, f"Duration too short: {media_info['duration']}s"
    assert media_info["file_size_bytes"] > 10000, "File size suspiciously small"

    print("\n==================================================================")
    print("SUCCESS: End-to-End Test PASSED! Real, playable MP4 verified.")
    print("==================================================================")
    return True

if __name__ == "__main__":
    asyncio.run(run_end_to_end_test())
