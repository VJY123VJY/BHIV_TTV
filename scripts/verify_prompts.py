#!/usr/bin/env python3
import asyncio
import os
import sys
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.pipelines.text_to_video import pipeline
from app.utils.media_check import validate_video_file

PROMPTS = [
    "A golden retriever dog running joyfully on a sunny tropical beach with ocean waves",
    "An astronaut walking slowly across the red dusty dunes of planet Mars under a dark sky",
    "A futuristic sports car racing through neon rain on a cybernetic highway at midnight"
]

async def verify_pipeline_differentiation():
    print("=" * 80)
    print("RUNNING 3-PROMPT DIFFERENTIATION VERIFICATION TEST")
    print("=" * 80)

    results = []

    for idx, prompt in enumerate(PROMPTS, 1):
        print(f"\n[Test Run {idx}/3] Prompt: '{prompt}'")
        res = await pipeline.execute(
            prompt=prompt,
            duration=10,
            style="cinematic",
            voice=True
        )
        assert res["status"] == "success", f"Run {idx} failed: {res}"
        meta = res["metadata"]
        video_id = res["execution_id"]
        video_path = BASE_DIR / "generated" / "videos" / f"{video_id}.mp4"
        meta_path = BASE_DIR / "generated" / "videos" / f"{video_id}_metadata.json"

        assert os.path.exists(video_path), f"Video missing: {video_path}"
        assert os.path.exists(meta_path), f"Metadata missing: {meta_path}"

        is_valid, media_info = validate_video_file(str(video_path))
        assert is_valid, f"Media validation failed for {video_path}: {media_info}"

        # Collect first keyframe image
        scenes = meta.get("scenes", [])
        first_keyframe = scenes[0].get("image_path") if scenes else None
        assert first_keyframe and os.path.exists(first_keyframe), f"Keyframe missing: {first_keyframe}"

        kf_img = cv2.imread(first_keyframe)
        assert kf_img is not None, f"Failed to read keyframe image: {first_keyframe}"

        results.append({
            "index": idx,
            "prompt": prompt,
            "execution_id": video_id,
            "video_path": str(video_path),
            "keyframe_path": first_keyframe,
            "keyframe_mean_bgr": np.mean(kf_img, axis=(0, 1)).tolist(),
            "scenes": scenes,
            "file_size": os.path.getsize(video_path),
            "media_info": media_info
        })
        print(f"  -> Generated Video ID: {video_id}")
        print(f"  -> Output File: {video_path.name} ({os.path.getsize(video_path)} bytes)")
        print(f"  -> Keyframe Mean Color (BGR): {[round(x, 1) for x in results[-1]['keyframe_mean_bgr']]}")

    print("\n" + "=" * 80)
    print("VERIFYING UNIQUENESS & DIFFERENTIATION ACROSS RUNS")
    print("=" * 80)

    # 1. Execution IDs must all be unique
    exec_ids = [r["execution_id"] for r in results]
    assert len(set(exec_ids)) == 3, f"Execution IDs not unique: {exec_ids}"
    print("[PASS] All execution IDs and filenames are completely unique.")

    # 2. Output filenames must all be unique
    video_files = [r["video_path"] for r in results]
    assert len(set(video_files)) == 3, f"Video paths not unique: {video_files}"
    print("[PASS] All video file paths are distinct.")

    # 3. Keyframe visuals must be visually distinct
    kf_colors = [r["keyframe_mean_bgr"] for r in results]
    # Check that keyframe color signatures differ significantly
    diff_0_1 = np.linalg.norm(np.array(kf_colors[0]) - np.array(kf_colors[1]))
    diff_1_2 = np.linalg.norm(np.array(kf_colors[1]) - np.array(kf_colors[2]))
    diff_0_2 = np.linalg.norm(np.array(kf_colors[0]) - np.array(kf_colors[2]))
    print(f"Keyframe visual color distances: Run1-Run2={diff_0_1:.2f}, Run2-Run3={diff_1_2:.2f}, Run1-Run3={diff_0_2:.2f}")
    assert diff_0_1 > 20.0, "Keyframes between prompt 1 and 2 are too similar!"
    assert diff_1_2 > 20.0, "Keyframes between prompt 2 and 3 are too similar!"
    assert diff_0_2 > 20.0, "Keyframes between prompt 1 and 3 are too similar!"
    print("[PASS] Keyframe imagery is visually distinct and tailored to each prompt.")

    # 4. Narratives and scene titles must reflect the prompt
    p1_scenes = " ".join([s["narrative"] + " " + s["visual_description"] for s in results[0]["scenes"]]).lower()
    p2_scenes = " ".join([s["narrative"] + " " + s["visual_description"] for s in results[1]["scenes"]]).lower()
    p3_scenes = " ".join([s["narrative"] + " " + s["visual_description"] for s in results[2]["scenes"]]).lower()

    assert any(w in p1_scenes for w in ["dog", "beach", "golden retriever", "waves", "ocean"]), f"Prompt 1 narrative missing keywords: {p1_scenes}"
    assert any(w in p2_scenes for w in ["astronaut", "mars", "dunes", "red", "dusty"]), f"Prompt 2 narrative missing keywords: {p2_scenes}"
    assert any(w in p3_scenes for w in ["car", "neon", "rain", "cybernetic", "highway", "racing"]), f"Prompt 3 narrative missing keywords: {p3_scenes}"
    print("[PASS] Story and scene generation dynamically parsed and reflected prompt entities and settings.")

    print("\n" + "=" * 80)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    asyncio.run(verify_pipeline_differentiation())
