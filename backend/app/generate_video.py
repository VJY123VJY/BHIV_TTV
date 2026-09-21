"""
CLI tool for direct Text-to-Video generation.
Usage:
  python -m app.generate_video --prompt "A farmer in a lush field" --character rahul_farmer --language mr --duration 15
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR.parent))

from app.pipelines.text_to_video import pipeline


async def run_cli():
    parser = argparse.ArgumentParser(description="BHIV TTV Studio CLI Video Generator")
    parser.add_argument("--prompt", type=str, required=True, help="Text description / script")
    parser.add_argument("--duration", type=int, default=15, help="Video duration in seconds")
    parser.add_argument("--style", type=str, default="cinematic", help="Visual style preset")
    parser.add_argument("--character", type=str, default=None, help="Character profile ID (e.g. rahul_farmer)")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g. mr, hi, en)")
    parser.add_argument("--voice", action="store_true", default=True, help="Enable speech narration")
    parser.add_argument("--aspect-ratio", type=str, default="16:9", help="16:9 or 9:16")
    parser.add_argument("--quality", type=str, default="standard", help="standard, high, ultra")
    parser.add_argument("--subtitles", action="store_true", default=True, help="Generate subtitles")
    parser.add_argument("--lipsync", action="store_true", default=True, help="Apply lip synchronization")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")

    args = parser.parse_args()

    print(f"=== Starting TTV Generation ===")
    print(f"Prompt:       {args.prompt}")
    print(f"Character:    {args.character or 'None'}")
    print(f"Language:     {args.language}")
    print(f"Duration:     {args.duration}s")
    print(f"Aspect/Res:   {args.aspect_ratio} ({args.quality})")

    result = await pipeline.execute(
        prompt=args.prompt,
        duration=args.duration,
        style=args.style,
        voice=args.voice,
        aspect_ratio=args.aspect_ratio,
        quality=args.quality,
        language=args.language,
        lipsync=args.lipsync,
        fps=args.fps,
        character_id=args.character,
        subtitles=args.subtitles,
    )

    meta = result.get("metadata", {})
    print("=== Video Generation Completed Successfully ===")
    print(f"Video URL:    {meta.get('video_url')}")
    print(f"Resolution:   {meta.get('resolution')}")
    print(f"Aspect Ratio: {meta.get('aspect_ratio')}")
    print(f"Language:     {meta.get('language')}")
    print(f"Total Scenes: {len(meta.get('scenes', []))}")


def main():
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
