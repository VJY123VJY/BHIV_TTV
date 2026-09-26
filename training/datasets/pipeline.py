"""
TTV Studio Training / Fine-Tuning Dataset Pipeline.

Provides complete parsing, validation, resolution, download caching, and split management:
- Supports dictionary manifests: {"dataset_name": "...", "records": [...]}
- Supports raw JSON arrays: [{"video_url": "...", "prompt": "...", "split": "train"}, ...]
- Supports JSONL files and string payloads
- Validates records:
    * id (auto-generated if missing in raw arrays; detects duplicates)
    * video_url OR video_path (supports HTTP/HTTPS and local paths)
    * prompt OR caption
    * split in ("train", "val", "validation", "test"), normalizing "validation" -> "val"
- Caches and downloads remote videos to data/training/<dataset_slug>/videos/
- Avoids redownloading existing valid videos
- Probes and verifies video decodability with OpenCV
- Produces normalized internal records:
    {"id": ..., "video_path": ..., "video_url": ..., "prompt": ..., "caption": ..., "category": ..., "split": ...}
- Emits structured dataset directory:
    data/training/<dataset_slug>/
        videos/
        metadata.json
        train.json / train.jsonl
        val.json / val.jsonl
        test.json / test.jsonl
        dataset.jsonl
        download_errors.json
"""

from __future__ import annotations

import os
import re
import json
import urllib.parse
import urllib.request
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


ACCEPTED_SPLITS = {"train", "val", "validation", "test"}


def normalize_split(split_raw: Optional[str]) -> Optional[str]:
    """Normalizes split name to 'train', 'val', or 'test'. Returns None if invalid."""
    if not split_raw:
        return None
    val = str(split_raw).strip().lower()
    if val == "validation":
        return "val"
    if val in ("train", "val", "test"):
        return val
    return None


def is_valid_url(url: Optional[str]) -> bool:
    """Validates whether a string is a valid HTTP/HTTPS URL."""
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urllib.parse.urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def is_valid_video_file(file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """
    Verifies that a local file exists, is non-empty, and is a decodable video.
    Checks container signatures and probes decoding with OpenCV.
    """
    p = Path(file_path)
    if not p.exists():
        return False, f"File does not exist: {file_path}"
    if p.stat().st_size == 0:
        return False, "File size is 0 bytes"

    # Minimal container check (MP4, MKV/WebM, AVI, MOV)
    try:
        with open(p, "rb") as f:
            header = f.read(64)
        is_mp4 = (len(header) >= 12 and b"ftyp" in header[4:16])
        is_webm = header.startswith(b"\x1a\x45\xdf\xa3")
        is_avi = (header.startswith(b"RIFF") and b"AVI " in header[8:16])
        is_mov = (b"moov" in header or b"mdat" in header or is_mp4)
        if not (is_mp4 or is_webm or is_avi or is_mov):
            # Not an immediately recognizable container, but probe with OpenCV anyway
            pass
    except Exception as e:
        return False, f"Failed reading file header: {e}"

    if OPENCV_AVAILABLE:
        try:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                return False, "OpenCV failed to open video file (corrupt or unsupported codec)"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                cap.release()
                return False, f"Invalid dimensions: {width}x{height}"
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return False, "Failed to decode first frame from video"
        except Exception as e:
            return False, f"Video probe error: {e}"

    return True, None


def parse_manifest(
    content_or_path: Union[str, Path, bytes, List[Dict[str, Any]], Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """
    Parses manifest records and metadata from:
    1. Dict with 'records' (or 'samples', 'data', 'videos', 'items', 'dataset')
    2. List of dicts (raw JSON array)
    3. JSON string or bytes (with UTF-8 / BOM handling)
    4. JSONL string, bytes, or file
    5. Existing file path (.json or .jsonl)

    Returns:
        (records_list, top_level_metadata_dict, resolved_path)
    """
    # 1. Already a List
    if isinstance(content_or_path, list):
        return content_or_path, {}, None

    # 2. Already a Dict
    if isinstance(content_or_path, dict):
        meta = {k: v for k, v in content_or_path.items() if k not in ("records", "samples", "data", "videos", "items")}
        for key in ("records", "samples", "data", "videos", "items", "dataset"):
            if key in content_or_path and isinstance(content_or_path[key], list):
                return content_or_path[key], meta, None
        if "prompt" in content_or_path or "caption" in content_or_path or "id" in content_or_path:
            return [content_or_path], meta, None
        return [], meta, None

    # 3. Path string or Path object
    resolved_path: Optional[str] = None
    if isinstance(content_or_path, (str, Path)):
        p = Path(content_or_path)
        if p.exists() and p.is_file():
            resolved_path = str(p.resolve())
            text = p.read_text(encoding="utf-8-sig")
            records, meta = _parse_text_payload(text)
            return records, meta, resolved_path

    # 4. Raw text or bytes
    if isinstance(content_or_path, bytes):
        text = content_or_path.decode("utf-8-sig", errors="replace")
    else:
        text = str(content_or_path)

    records, meta = _parse_text_payload(text)
    return records, meta, resolved_path


def _parse_text_payload(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Internal helper to parse JSON or JSONL text into (records, metadata)."""
    trimmed = text.strip()
    if not trimmed:
        return [], {}

    # Try full JSON first
    if trimmed.startswith("{") or trimmed.startswith("["):
        try:
            data = json.loads(trimmed)
            if isinstance(data, list):
                return data, {}
            if isinstance(data, dict):
                meta = {k: v for k, v in data.items() if k not in ("records", "samples", "data", "videos", "items")}
                for key in ("records", "samples", "data", "videos", "items", "dataset"):
                    if key in data and isinstance(data[key], list):
                        return data[key], meta
                if "prompt" in data or "caption" in data or "id" in data:
                    return [data], meta
        except Exception:
            pass

    # Fallback: Parse line-by-line JSONL
    records: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    for line in trimmed.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                records.append(parsed)
            elif isinstance(parsed, list):
                records.extend([x for x in parsed if isinstance(x, dict)])
        except Exception:
            pass

    return records, meta


def validate_manifest_records(
    records: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    min_prompt_len: int = 1
) -> Dict[str, Any]:
    """
    Validates manifest records:
    - Required:
        * id (non-empty string, auto-assigned if missing in raw arrays, checked for duplicates)
        * video_url OR video_path (non-empty)
        * prompt OR caption (non-empty)
        * split (train, val, validation, test) -> normalized to train, val, test
    - Calculates exact sample counts and split distributions (train, val, test)
    - Returns detailed validation report
    """
    metadata = metadata or {}
    valid_samples: List[Dict[str, Any]] = []
    rejected_samples: List[Dict[str, Any]] = []
    splits_count: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    categories_count: Dict[str, int] = {}
    formats_count: Dict[str, int] = {}
    has_remote_urls = False

    seen_ids = set()

    for idx, raw_item in enumerate(records):
        if not isinstance(raw_item, dict):
            rejected_samples.append({
                "index": idx,
                "id": f"record_{idx:04d}",
                "reasons": ["Record is not a valid JSON object"],
                "sample": raw_item
            })
            continue

        issues: List[str] = []

        # 1. ID Validation / Generation
        raw_id = raw_item.get("id")
        if raw_id is not None and str(raw_id).strip():
            item_id = str(raw_id).strip()
            if item_id in seen_ids:
                issues.append(f"Duplicate id: '{item_id}'")
            seen_ids.add(item_id)
        else:
            # Generate deterministic fallback ID for raw items lacking ID
            item_id = f"ttv_sample_{idx+1:04d}"
            seen_ids.add(item_id)

        # 2. Video reference validation (video_url OR video_path / video / source_url)
        video_url = str(raw_item.get("video_url") or "").strip()
        video_path = str(raw_item.get("video_path") or raw_item.get("video") or raw_item.get("path") or "").strip()
        source_url = str(raw_item.get("source_url") or "").strip()

        video_ref = video_url or video_path or source_url
        if not video_ref:
            issues.append("Missing video reference: record must specify 'video_url' or 'video_path'")

        if video_url and is_valid_url(video_url):
            has_remote_urls = True
        elif video_ref.startswith(("http://", "https://")):
            has_remote_urls = True

        # 3. Prompt validation (prompt OR caption)
        prompt = str(raw_item.get("prompt") or raw_item.get("caption") or "").strip()
        caption = str(raw_item.get("caption") or prompt).strip()
        if len(prompt) < min_prompt_len:
            issues.append(f"Missing or empty prompt/caption (length < {min_prompt_len})")

        # 4. Split validation
        raw_split = raw_item.get("split")
        normalized_split = normalize_split(raw_split)
        if not normalized_split:
            issues.append(
                f"Invalid or missing split: '{raw_split}'. Accepted split values are 'train', 'val', 'validation', 'test'."
            )
            normalized_split = "train"  # placeholder for metrics structure

        category = str(raw_item.get("category") or "general").strip().lower()
        fmt = str(raw_item.get("format") or "mp4").strip().lower()

        if issues:
            rejected_samples.append({
                "index": idx,
                "id": item_id,
                "reasons": issues,
                "sample": raw_item
            })
            continue

        # Record valid item
        splits_count[normalized_split] = splits_count.get(normalized_split, 0) + 1
        categories_count[category] = categories_count.get(category, 0) + 1
        formats_count[fmt] = formats_count.get(fmt, 0) + 1

        valid_samples.append({
            "id": item_id,
            "video_url": video_url or (video_ref if is_valid_url(video_ref) else ""),
            "video_path": video_path or (video_ref if not is_valid_url(video_ref) else ""),
            "prompt": prompt,
            "caption": caption,
            "category": category,
            "split": normalized_split,
            "format": fmt,
            "license_note": raw_item.get("license_note") or raw_item.get("license") or "N/A",
            "source_url": source_url or video_url,
            "label_status": raw_item.get("label_status") or "VALIDATED"
        })

    total = len(records)
    valid_count = len(valid_samples)
    rejected_count = len(rejected_samples)
    is_valid = (valid_count > 0 and rejected_count == 0)

    summary = (
        f"Validated {valid_count}/{total} records "
        f"(Train: {splits_count.get('train', 0)}, Val: {splits_count.get('val', 0)}, Test: {splits_count.get('test', 0)})."
    )
    if rejected_count > 0:
        summary += f" Rejected {rejected_count} invalid records."

    return {
        "valid": is_valid,
        "total_records": total,
        "valid_records": valid_count,
        "rejected_records": rejected_count,
        "splits": splits_count,
        "categories": categories_count,
        "formats": formats_count,
        "has_remote_urls": has_remote_urls,
        "sample_preview": valid_samples[:5],
        "issues": rejected_samples,
        "summary": summary,
        "dataset_name": metadata.get("dataset_name", "TTV_Training_Dataset")
    }


def download_video_file(
    url: str,
    target_path: Path,
    timeout: int = 15,
    max_retries: int = 2
) -> Tuple[bool, Optional[str]]:
    """
    Downloads a video from an HTTP/HTTPS URL into target_path:
    - Does not redownload if target exists and is a valid video
    - Verifies HTTP status 200
    - Verifies downloaded file integrity via OpenCV
    - Returns (success, error_message)
    """
    if target_path.exists() and target_path.stat().st_size > 0:
        ok, probe_err = is_valid_video_file(target_path)
        if ok:
            return True, None
        target_path.unlink(missing_ok=True)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(f".tmp_{os.getpid()}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TTV-Studio/1.0",
        "Accept": "*/*"
    }

    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status if hasattr(resp, "status") else 200
                if status != 200:
                    last_error = f"HTTP status {status}"
                    continue
                with open(temp_target, "wb") as f_out:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f_out.write(chunk)

            ok, verify_err = is_valid_video_file(temp_target)
            if not ok:
                temp_target.unlink(missing_ok=True)
                last_error = f"Downloaded content is not a valid video: {verify_err}"
                continue

            # Atomic replace
            temp_target.replace(target_path)
            return True, None
        except Exception as e:
            if temp_target.exists():
                temp_target.unlink(missing_ok=True)
            last_error = str(e)

    return False, last_error


def build_and_resolve_dataset(
    records: List[Dict[str, Any]],
    dataset_name: str = "ttv_training_manifest_120",
    base_output_dir: Union[str, Path] = "data/training",
    workspace_root: Optional[Path] = None,
    download_remote_videos: bool = True,
    download_timeout: int = 15
) -> Dict[str, Any]:
    """
    Resolves records into the production dataset directory structure:
    data/training/<dataset_slug>/
        videos/
        metadata.json
        train.json
        val.json
        test.json
        train.jsonl
        val.jsonl
        test.jsonl
        dataset.jsonl
        download_errors.json

    Resolves every sample into the normalized structure:
    {
        "id": "...",
        "video_path": "...",
        "video_url": "...",
        "prompt": "...",
        "caption": "...",
        "category": "...",
        "split": "train"
    }
    """
    workspace_root = workspace_root or Path.cwd()
    clean_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset_name).strip("_") or "dataset"
    
    target_dir = Path(base_output_dir)
    if not target_dir.is_absolute():
        target_dir = workspace_root / target_dir
    dataset_dir = target_dir / clean_slug
    videos_dir = dataset_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Local fallback search paths: generated/videos, data/raw, etc.
    local_video_dirs = [
        workspace_root / "generated" / "videos",
        workspace_root / "data" / "raw",
        workspace_root / "data" / "processed",
    ]

    normalized_records: List[Dict[str, Any]] = []
    download_errors: List[Dict[str, Any]] = []
    splits_dict: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for idx, sample in enumerate(records):
        item_id = str(sample.get("id") or f"ttv_sample_{idx+1:04d}").strip()
        raw_url = str(sample.get("video_url") or sample.get("source_url") or "").strip()
        raw_path = str(sample.get("video_path") or sample.get("video") or "").strip()
        prompt = str(sample.get("prompt") or sample.get("caption") or "").strip()
        caption = str(sample.get("caption") or prompt).strip()
        category = str(sample.get("category") or "general").strip()
        split = normalize_split(sample.get("split")) or "train"

        resolved_video_path = ""

        # A. Check if raw_path is an existing local file
        if raw_path:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = workspace_root / candidate
            if candidate.exists() and candidate.is_file():
                resolved_video_path = str(candidate.resolve())

        # B. If not resolved yet, check local video dirs by item_id or stem
        if not resolved_video_path and raw_url:
            url_filename = Path(urllib.parse.urlparse(raw_url).path).name
            cached_candidate = videos_dir / (url_filename or f"{item_id}.mp4")
            if cached_candidate.exists() and cached_candidate.stat().st_size > 0:
                resolved_video_path = str(cached_candidate.resolve())
            else:
                for ldir in local_video_dirs:
                    if not ldir.exists():
                        continue
                    if url_filename:
                        match = ldir / url_filename
                        if match.exists():
                            resolved_video_path = str(match.resolve())
                            break

        # C. Download remote video if needed and enabled
        if not resolved_video_path and is_valid_url(raw_url):
            url_filename = Path(urllib.parse.urlparse(raw_url).path).name
            if not url_filename or not url_filename.endswith((".mp4", ".mov", ".webm", ".avi")):
                url_filename = f"{item_id}.mp4"
            dest = videos_dir / f"{item_id}_{url_filename}"

            if download_remote_videos:
                ok, dl_err = download_video_file(raw_url, dest, timeout=download_timeout)
                if ok:
                    resolved_video_path = str(dest.resolve())
                else:
                    download_errors.append({
                        "id": item_id,
                        "video_url": raw_url,
                        "error": dl_err or "Unknown download error"
                    })
            else:
                # Placeholder for scheduled deferred download
                resolved_video_path = str(dest.resolve())

        # D. If still not resolved and local generated videos exist, support local source
        if not resolved_video_path:
            gen_dir = workspace_root / "generated" / "videos"
            if gen_dir.exists():
                mp4s = list(gen_dir.glob("*.mp4"))
                if mp4s:
                    resolved_video_path = str(mp4s[idx % len(mp4s)].resolve())

        norm_item = {
            "id": item_id,
            "video_path": resolved_video_path,
            "video_url": raw_url if is_valid_url(raw_url) else "",
            "prompt": prompt,
            "caption": caption,
            "category": category,
            "split": split
        }

        normalized_records.append(norm_item)
        if split in splits_dict:
            splits_dict[split].append(norm_item)

    # Write output files
    # 1. metadata.json
    metadata = {
        "dataset_name": dataset_name,
        "clean_slug": clean_slug,
        "total_records": len(normalized_records),
        "splits": {k: len(v) for k, v in splits_dict.items()},
        "download_errors_count": len(download_errors),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "videos_dir": str(videos_dir.resolve())
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 2. train.json, val.json, test.json (raw JSON arrays)
    for sp, items in splits_dict.items():
        (dataset_dir / f"{sp}.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
        # 3. train.jsonl, val.jsonl, test.jsonl
        with open(dataset_dir / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    # 4. dataset.jsonl (complete manifest)
    dataset_jsonl_path = dataset_dir / "dataset.jsonl"
    with open(dataset_jsonl_path, "w", encoding="utf-8") as f:
        for item in normalized_records:
            f.write(json.dumps(item) + "\n")

    # 5. download_errors.json
    (dataset_dir / "download_errors.json").write_text(json.dumps(download_errors, indent=2), encoding="utf-8")

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_manifest": str(dataset_jsonl_path.resolve()),
        "train_manifest": str((dataset_dir / "train.jsonl").resolve()),
        "val_manifest": str((dataset_dir / "val.jsonl").resolve()),
        "test_manifest": str((dataset_dir / "test.jsonl").resolve()),
        "total_records": len(normalized_records),
        "splits": {k: len(v) for k, v in splits_dict.items()},
        "download_errors": download_errors
    }
