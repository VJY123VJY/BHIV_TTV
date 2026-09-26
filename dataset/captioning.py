"""Video-aware captioning with an optional local vision-language model.

Captions are generated from sampled frames, never from file names, search
queries, or URL text.  Every generated caption is marked ``needs_review`` so
style labels and licences are not silently treated as ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dataset.ttv_prepare import _jsonl, _write_jsonl


INSTRUCTION = (
    "Describe this video for text-to-video training in one factual sentence. "
    "Include subject, visible action, environment, camera framing or movement, "
    "apparent motion, lighting, and visual style only when visible. Do not guess "
    "identity, age, nationality, or facts not visible in the frames."
)


def _load_images(frame_paths: List[str]):
    from PIL import Image
    return [Image.open(path).convert("RGB") for path in frame_paths]


class TransformersVideoCaptioner:
    """Lazy Hugging Face VLM adapter; model download is explicit CLI opt-in."""
    def __init__(self, model_id: str, device: str = "auto") -> None:
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError as exc:
            raise RuntimeError("Captioning requires transformers. Install the optional training dependencies first.") from exc
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id)
        if device != "auto": self.model.to(device)

    def caption(self, frame_paths: List[str]) -> str:
        if not frame_paths:
            raise ValueError("No extracted frames available for VLM captioning")
        # Most image-to-text VLMs accept a batch of sampled frames. A multi-frame
        # model can use all eight; single-image models still receive an honest
        # representative middle frame instead of a filename-derived prompt.
        images = _load_images(frame_paths)
        try:
            inputs = self.processor(images=images, text=INSTRUCTION, return_tensors="pt", padding=True)
            generated = self.model.generate(**inputs, max_new_tokens=96)
            text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        except Exception as exc:
            raise RuntimeError(f"The selected VLM could not caption sampled frames: {exc}") from exc
        if not text or len(text.split()) < 4:
            raise RuntimeError("VLM returned an unusable caption; retain the sample for manual annotation.")
        return text


def apply_captions(records: Iterable[Dict[str, Any]], captioner: TransformersVideoCaptioner) -> List[Dict[str, Any]]:
    output = []
    for record in records:
        updated = dict(record)
        try:
            caption = captioner.caption(list(record.get("frames") or []))
            updated.update({"caption": caption, "caption_provider": "transformers_vlm", "caption_status": "needs_review"})
        except (RuntimeError, ValueError) as exc:
            updated.update({"caption": None, "caption_status": "needs_manual_caption", "caption_error": str(exc)})
        output.append(updated)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate review-required temporal captions from extracted video frames")
    parser.add_argument("--input", required=True, help="Frame-enriched input JSONL")
    parser.add_argument("--output", required=True, help="Output metadata JSONL")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct", help="Explicitly chosen VLM ID/path")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    captioner = TransformersVideoCaptioner(args.model, args.device)
    rows = apply_captions(_jsonl(Path(args.input)), captioner)
    _write_jsonl(Path(args.output), rows)
    print(json.dumps({"records": len(rows), "needs_review": sum(row.get("caption_status") == "needs_review" for row in rows), "output": args.output}, indent=2))


if __name__ == "__main__": main()
