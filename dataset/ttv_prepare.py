"""Reproducible, provenance-first TTV dataset preparation commands.

This module only processes local files listed in a reviewed JSONL manifest or
assets returned by the registry-gated Wikimedia adapter.  It intentionally has
no generic URL downloader and never infers rights from public visibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import cv2
import numpy as np

from dataset.quality import calculate_phash, calculate_sha256, hamming_distance
from dataset.registry import DatasetPolicyError, license_is_allowed, load_registry, url_is_prohibited
from dataset.video_processor import video_processor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CATEGORY_NAMES = {"people", "nature", "animals", "vehicles", "urban", "rural", "objects", "cinematic"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _record_rejection(data_root: Path, record: Dict[str, Any], reason: str) -> None:
    rejected = data_root / "rejected"
    rejected.mkdir(parents=True, exist_ok=True)
    asset_id = record.get("id") or record.get("asset_id") or hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()[:12]
    payload = {"record": record, "reason": reason, "rejected_at": _utcnow()}
    (rejected / f"{asset_id}_{reason}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_provenance(record: Dict[str, Any]) -> None:
    """Require item-level rights and local attestation before processing."""
    source_id = record.get("source_id", "")
    license_value = record.get("license")
    if not license_is_allowed(license_value):
        raise DatasetPolicyError(f"license is not allowlisted: {license_value!r}")
    if not record.get("source_url") or not record.get("license_url") or not record.get("creator"):
        raise DatasetPolicyError("source_url, license_url, and creator are mandatory per-item provenance")
    if url_is_prohibited(record["source_url"]):
        raise DatasetPolicyError("prohibited social/copyrighted host")
    if source_id == "user_authorized" and not record.get("rights_attestation_id"):
        raise DatasetPolicyError("user-authorized media needs a rights_attestation_id")
    if source_id and source_id not in {s["id"] for s in load_registry().get("sources", [])}:
        raise DatasetPolicyError(f"unknown source_id: {source_id}")


def _sample_indices(frame_count: int, sample_count: int = 24) -> set[int]:
    if frame_count <= sample_count:
        return set(range(frame_count))
    return {round(i * (frame_count - 1) / (sample_count - 1)) for i in range(sample_count)}


def audit_video(path: Path, min_width: int = 1280, min_height: int = 720) -> Dict[str, Any]:
    """Decode representative frames and calculate training quality signals."""
    result: Dict[str, Any] = {"file": str(path), "valid": False, "failure_reasons": []}
    if not path.exists() or path.stat().st_size < 1024:
        result["failure_reasons"].append("missing_or_too_small")
        return result
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        result["failure_reasons"].append("decode_open_failed")
        return result
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frame_count = float(cap.get(cv2.CAP_PROP_FPS) or 0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        cap.release(); result["failure_reasons"].append("invalid_stream_metadata"); return result
    selected, prior_gray = _sample_indices(frame_count), None
    blur_scores: List[float] = []; brightness: List[float] = []; motion: List[float] = []; hashes: List[str] = []
    decoded = 0
    for index in range(frame_count):
        ok, frame = cap.read()
        if not ok or frame is None:
            result["failure_reasons"].append(f"decode_failed_at_{index}")
            break
        if index not in selected:
            continue
        decoded += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brightness.append(float(gray.mean()))
        hashes.append(calculate_phash(gray))
        if prior_gray is not None:
            motion.append(float(np.mean(cv2.absdiff(gray, prior_gray))))
        prior_gray = gray
    cap.release()
    duration = frame_count / fps
    result.update({
        "width": width, "height": height, "fps": round(fps, 3), "frame_count": frame_count,
        "duration": round(duration, 3), "aspect_ratio": "16:9" if width >= height else "9:16",
        "blur_score": round(float(np.mean(blur_scores or [0])), 3),
        "brightness": round(float(np.mean(brightness or [0])), 3),
        "motion_score": round(float(np.mean(motion or [0])), 3),
        "frame_hashes": hashes, "decoded_samples": decoded,
        "file_size_bytes": path.stat().st_size,
    })
    if width < min_width or height < min_height: result["failure_reasons"].append("below_min_resolution")
    if duration < 2: result["failure_reasons"].append("too_short")
    if result["blur_score"] < 45: result["failure_reasons"].append("excessive_blur")
    if result["brightness"] < 12: result["failure_reasons"].append("extremely_dark")
    if decoded < min(3, len(selected)): result["failure_reasons"].append("insufficient_decoded_samples")
    # OCR/logo detection is intentionally not claimed without a configured OCR
    # model. Such clips remain manual-review candidates.
    result["watermark_review_required"] = True
    result["valid"] = not result["failure_reasons"]
    result["quality_score"] = round(max(0.0, min(1.0, 0.4 + min(result["blur_score"], 250) / 500 + min(result["motion_score"], 40) / 200)), 3)
    return result


def _fit_with_padding(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize without stretching; pad neutral black bars where crop is unsafe."""
    original_h, original_w = frame.shape[:2]
    scale = min(width / original_w, height / original_h)
    resized = cv2.resize(frame, (round(original_w * scale), round(original_h * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def extract_clips(record: Dict[str, Any], data_root: Path, target_fps: int = 24, target_width: int = 1280, target_height: int = 720, clip_seconds: int = 4) -> List[Dict[str, Any]]:
    """Shot-aware temporal clip creation; clips retain source-group provenance."""
    input_path = Path(record["local_path"])
    audit = audit_video(input_path)
    if not audit["valid"]:
        _record_rejection(data_root, record, "video_quality_failed")
        return []
    cap = cv2.VideoCapture(str(input_path)); fps = float(cap.get(cv2.CAP_PROP_FPS) or target_fps)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    shot_ranges = video_processor.detect_scenes(str(input_path)) or [(0, source_frames - 1)]
    output_root = data_root / "clips" / str(record.get("id", input_path.stem)); output_root.mkdir(parents=True, exist_ok=True)
    clips: List[Dict[str, Any]] = []; clip_index = 0
    source_frames_per_clip = max(1, math.ceil(clip_seconds * fps))
    output_frames_per_clip = max(1, target_fps * clip_seconds)
    for shot_start, shot_end in shot_ranges:
        start = shot_start
        while start + source_frames_per_clip <= shot_end + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            output_path = output_root / f"{record.get('id', input_path.stem)}_{clip_index:04d}.mp4"
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (target_width, target_height))
            frame_count = 0
            source_position = start
            while frame_count < output_frames_per_clip:
                source_position = min(shot_end, round(start + frame_count * fps / target_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_position)
                ok, frame = cap.read()
                if not ok or frame is None: break
                writer.write(_fit_with_padding(frame, target_width, target_height))
                frame_count += 1
            writer.release()
            if frame_count == output_frames_per_clip:
                clip = dict(record)
                clip.update({"id": f"{record.get('id', input_path.stem)}_clip_{clip_index:04d}", "video": str(output_path.resolve()), "local_path": str(output_path.resolve()), "source_group_id": record.get("source_group_id") or record.get("id"), "parent_asset_id": record.get("id"), "clip_start_seconds": round(start / fps, 3), "duration": clip_seconds, "fps": target_fps, "width": target_width, "height": target_height, "aspect_ratio": "16:9" if target_width >= target_height else "9:16", "quality": audit})
                clips.append(clip); clip_index += 1
            else:
                output_path.unlink(missing_ok=True)
            start += source_frames_per_clip
    cap.release()
    return clips


def extract_frames(records: Iterable[Dict[str, Any]], data_root: Path, frames_per_clip: int = 8) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for record in records:
        path = Path(record["local_path"]); cap = cv2.VideoCapture(str(path)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_dir = data_root / "frames" / record["id"]; frame_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for number, index in enumerate(sorted(_sample_indices(count, frames_per_clip))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = cap.read()
            if ok and frame is not None:
                destination = frame_dir / f"frame_{number:03d}.jpg"; cv2.imwrite(str(destination), frame); paths.append(str(destination.resolve()))
        cap.release(); updated = dict(record); updated["frames"] = paths; output.append(updated)
    return output


def _fingerprint(record: Dict[str, Any]) -> List[str]:
    return list(record.get("quality", {}).get("frame_hashes", []))


def deduplicate(records: Iterable[Dict[str, Any]], data_root: Path, max_hamming: int = 4) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []; seen_sha: Dict[str, Dict[str, Any]] = {}
    for record in records:
        path = Path(record["local_path"]); sha = calculate_sha256(str(path)) if path.exists() else ""
        duplicate_of = None
        if sha and sha in seen_sha: duplicate_of = seen_sha[sha].get("id")
        if not duplicate_of:
            hashes = _fingerprint(record)
            for prior in kept:
                prior_hashes = _fingerprint(prior)
                if hashes and prior_hashes and min(hamming_distance(a, b) for a in hashes for b in prior_hashes) <= max_hamming:
                    duplicate_of = prior.get("id"); break
        if duplicate_of:
            _record_rejection(data_root, record, f"near_duplicate_of_{duplicate_of}")
            continue
        updated = dict(record); updated["sha256"] = sha; kept.append(updated); seen_sha[sha] = updated
    return kept


def grouped_split(records: Sequence[Dict[str, Any]], seed: str = "ttv-studio-v001") -> Dict[str, List[Dict[str, Any]]]:
    """70/15/15 split by parent/source group to prevent clip leakage."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records: groups[str(record.get("source_group_id") or record.get("parent_asset_id") or record.get("id"))].append(record)
    ordered = sorted(groups.items(), key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest())
    targets = {"train": len(records) * .70, "validation": len(records) * .15, "test": len(records) * .15}
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for _, group in ordered:
        split = min(splits, key=lambda name: len(splits[name]) / max(targets[name], 1))
        splits[split].extend(group)
    return splits


def dataset_report(records: Sequence[Dict[str, Any]], data_root: Path, dataset_version: str) -> Dict[str, Any]:
    categories = Counter((item.get("category") or "unclassified") for item in records)
    styles = Counter((item.get("style") or "unreviewed") for item in records)
    motion = Counter((item.get("motion") or "unreviewed") for item in records)
    videos = [r for r in records if r.get("local_path", "").lower().endswith(tuple(VIDEO_EXTENSIONS))]
    total_hours = sum(float(r.get("duration", 0)) for r in videos) / 3600
    rejected = len(list((data_root / "rejected").glob("*.json")))
    return {"dataset_version": dataset_version, "generated_at": _utcnow(), "total_images": sum(1 for r in records if r.get("media_type") == "image"), "total_videos": len(videos), "total_clips": len(records), "total_hours": round(total_hours, 4), "average_resolution": {"width": round(float(np.mean([r.get("width", 0) for r in records] or [0])), 1), "height": round(float(np.mean([r.get("height", 0) for r in records] or [0])), 1)}, "licensed_sources": sorted({r.get("source_id", r.get("source_name", "unknown")) for r in records}), "rejected_files": rejected, "categories": dict(categories), "category_percentages": {key: round(value / max(len(records), 1) * 100, 2) for key, value in categories.items()}, "styles": dict(styles), "motion": dict(motion), "underrepresented_categories": [name for name in CATEGORY_NAMES if categories[name] / max(len(records), 1) < .08]}


def cmd_validate(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root); valid = []
    for record in _jsonl(Path(args.input)):
        try:
            validate_provenance(record)
            path = Path(record.get("local_path", ""))
            if not path.exists(): raise ValueError("local_path does not exist")
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                quality = audit_video(path, args.min_width, args.min_height)
                if not quality["valid"]: raise ValueError(",".join(quality["failure_reasons"]))
                record["quality"] = quality
            valid.append(record)
        except (DatasetPolicyError, ValueError) as exc: _record_rejection(data_root, record, "validation_failed")
    _write_jsonl(Path(args.output), valid); print(json.dumps({"valid": len(valid), "output": args.output}, indent=2))


def cmd_clips(args: argparse.Namespace) -> None:
    output = []
    for record in _jsonl(Path(args.input)):
        output.extend(extract_clips(record, Path(args.data_root), args.fps, args.width, args.height, args.seconds))
    _write_jsonl(Path(args.output), output); print(json.dumps({"clips": len(output), "output": args.output}, indent=2))


def cmd_frames(args: argparse.Namespace) -> None:
    output = extract_frames(_jsonl(Path(args.input)), Path(args.data_root), args.count)
    _write_jsonl(Path(args.output), output); print(json.dumps({"records": len(output), "output": args.output}, indent=2))


def cmd_dedupe(args: argparse.Namespace) -> None:
    output = deduplicate(list(_jsonl(Path(args.input))), Path(args.data_root), args.max_hamming)
    _write_jsonl(Path(args.output), output); print(json.dumps({"kept": len(output), "output": args.output}, indent=2))


def cmd_splits(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root); records = list(_jsonl(Path(args.input))); splits = grouped_split(records, args.seed)
    for name, items in splits.items(): _write_jsonl(data_root / name / "metadata.jsonl", items)
    report = dataset_report(records, data_root, args.dataset_version); report["splits"] = {name: len(items) for name, items in splits.items()}
    (data_root / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    report = dataset_report(list(_jsonl(Path(args.input))), Path(args.data_root), args.dataset_version)
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTV Studio provenance-first data preparation")
    sub = p.add_subparsers(dest="command", required=True)
    def common(command: str):
        q = sub.add_parser(command); q.add_argument("--data-root", default=str(DEFAULT_DATA)); q.add_argument("--input", required=True); q.add_argument("--output", required=True); return q
    q = common("validate"); q.add_argument("--min-width", type=int, default=1280); q.add_argument("--min-height", type=int, default=720); q.set_defaults(func=cmd_validate)
    q = common("clips"); q.add_argument("--fps", type=int, default=24); q.add_argument("--width", type=int, default=1280); q.add_argument("--height", type=int, default=720); q.add_argument("--seconds", type=int, choices=[2, 4, 6, 8], default=4); q.set_defaults(func=cmd_clips)
    q = common("frames"); q.add_argument("--count", type=int, default=8); q.set_defaults(func=cmd_frames)
    q = common("dedupe"); q.add_argument("--max-hamming", type=int, default=4); q.set_defaults(func=cmd_dedupe)
    q = sub.add_parser("splits"); q.add_argument("--data-root", default=str(DEFAULT_DATA)); q.add_argument("--input", required=True); q.add_argument("--dataset-version", default="dataset_v001"); q.add_argument("--seed", default="ttv-studio-v001"); q.set_defaults(func=cmd_splits)
    q = sub.add_parser("report"); q.add_argument("--data-root", default=str(DEFAULT_DATA)); q.add_argument("--input", required=True); q.add_argument("--output", required=True); q.add_argument("--dataset-version", default="dataset_v001"); q.set_defaults(func=cmd_report)
    return p


def main() -> None:
    args = parser().parse_args(); args.func(args)


if __name__ == "__main__": main()
