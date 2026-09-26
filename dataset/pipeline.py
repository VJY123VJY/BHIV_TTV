"""
Unified Dataset Pipeline Orchestrator.
Coordinates ingestion, licensing verification, quality assurance, deduplication, captioning, and manifest building.
"""
import os
import json
import shutil
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

from dataset.models import AssetMetadata, QualityReport
from dataset.quality import (
    verify_license,
    calculate_sha256,
    assess_image_quality,
    hamming_distance,
)
from dataset.caption import auto_captioner
from dataset.sources.wikimedia import wikimedia_source
from dataset.sources.archive_org import archive_org_source
from dataset.sources.synthetic import synthetic_source
from dataset.downloader import download_resumable, extension_for_url, DownloadError
from dataset.video_processor import video_processor
from dataset.registry import DatasetPolicyError, assert_candidate_allowed


class DatasetPipeline:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.images_dir = self.base_dir / "images"
        self.videos_dir = self.base_dir / "videos"
        self.frames_dir = self.base_dir / "frames"
        self.staging_dir = self.base_dir / "staging"
        self.processed_dir = self.base_dir / "processed"
        self.rejected_dir = self.base_dir / "rejected"
        self.manifests_dir = self.base_dir / "manifests"
        self.metadata_dir = self.base_dir / "metadata"
        self.embeddings_dir = self.base_dir / "embeddings"
        self.captions_dir = self.base_dir / "captions"
        self.annotations_dir = self.base_dir / "annotations"

        self.ensure_directories()

    def ensure_directories(self):
        for d in [
            self.raw_dir,
            self.images_dir,
            self.videos_dir,
            self.frames_dir,
            self.staging_dir,
            self.processed_dir,
            self.rejected_dir,
            self.manifests_dir,
            self.metadata_dir,
            self.embeddings_dir,
            self.captions_dir,
            self.annotations_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Create dataset category subsets
        dataset_subsets = Path("datasets")
        for cat in [
            "human_actions",
            "talking_faces",
            "environments",
            "objects",
            "vehicles",
            "nature",
            "cinematic",
            "multilingual_talking",
        ]:
            (dataset_subsets / cat).mkdir(parents=True, exist_ok=True)
            (dataset_subsets / "manifests").mkdir(parents=True, exist_ok=True)

    def ingest_assets(
        self,
        source: str = "synthetic",
        query: str = "farmer speaking",
        limit: int = 10,
        license_filter: Optional[str] = None,
        media_type: str = "image",
        min_resolution: Optional[tuple[int, int]] = None,
        max_download_gb: int = 100,
    ) -> List[AssetMetadata]:
        """
        Ingests media assets from a specified source (wikimedia | archive_org | synthetic),
        evaluates licensing, applies quality filters, deduplicates, captions, and commits to processed.
        """
        ingested = []

        if source == "wikimedia":
            raw_candidates = wikimedia_source.search_assets(query, limit=limit, media_type=media_type)
        elif source == "archive_org":
            raw_candidates = archive_org_source.search_assets(query, limit=limit, media_type=media_type)
        else:
            # Synthetic / local bootstrap
            raw_candidates = []
            categories = ["talking_faces", "human_actions", "environments", "nature", "multilingual_talking"]
            for i in range(1, limit + 1):
                cat = categories[(i - 1) % len(categories)]
                dest = str(self.raw_dir / f"syn_{cat}_{i:03d}.jpg")
                info = synthetic_source.generate_synthetic_image(cat, dest, index=i)
                raw_candidates.append(info)

        # Process each candidate through the quality & provenance pipeline.
        # Only Wikimedia has API-level, per-file metadata sufficient for this
        # automated path. Archive/NASA/local assets must arrive through the
        # reviewed local-import command with a provenance attestation.
        for cand in raw_candidates:
            if source not in {"wikimedia", "synthetic"}:
                self._reject(cand.get("asset_id", "unknown"), "manual_review_required", source)
                continue
            asset = self.process_candidate(
                cand, query=query, license_filter=license_filter,
                min_resolution=min_resolution, max_download_bytes=max_download_gb * 1024 ** 3,
            )
            if asset:
                ingested.append(asset)

        return ingested

    def process_candidate(
        self, cand: Dict[str, Any], query: str = "", license_filter: Optional[str] = None,
        min_resolution: Optional[tuple[int, int]] = None, max_download_bytes: int = 100 * 1024 ** 3,
    ) -> Optional[AssetMetadata]:
        asset_id = cand.get("asset_id", "asset_unknown")
        src_url = cand.get("source_url", "")
        license_str = cand.get("license", "unknown")

        # 1. License and provenance check. This happens before download so a
        # public URL can never bypass the dataset registry.
        if cand.get("source_name") != "internal-synthetic":
            try:
                assert_candidate_allowed(cand, "wikimedia_commons")
            except DatasetPolicyError as exc:
                self._reject(asset_id, "provenance_policy_failed", str(exc))
                return None

        # 2. License Check
        lic_info = verify_license(license_str)
        requested_license = (license_filter or "").strip().lower()
        matches_requested_license = not requested_license or requested_license in license_str.lower()
        if not lic_info.training_eligible or not matches_requested_license:
            # Quarantine in rejected
            rej_path = self.rejected_dir / f"{asset_id}_rejected.json"
            rej_path.write_text(json.dumps({"asset_id": asset_id, "reason": "unsupported_license", "details": lic_info.__dict__}))
            return None

        # 3. Local asset file resolution (or download if remote)
        media_type = cand.get("media_type", "image")
        suffix = extension_for_url(cand.get("download_url") or src_url, media_type)
        local_path = cand.get("local_path") or str(self.raw_dir / f"{asset_id}{suffix}")
        if not os.path.exists(local_path):
            download_url = cand.get("download_url") or src_url
            if cand.get("source_name") == "internal-synthetic":
                synthetic_source.generate_synthetic_image(cand.get("category", "general"), local_path)
            elif not download_url or not download_url.startswith(("http://", "https://")):
                self._reject(asset_id, "missing_download_url")
                return None
            else:
                try:
                    download_resumable(download_url, Path(local_path), max_download_bytes)
                except DownloadError as exc:
                    self._reject(asset_id, "download_failed", str(exc))
                    return None

        # 4. Cryptographic Hash
        sha256 = calculate_sha256(local_path)

        # 5. Quality Audit
        if media_type == "video":
            video_info = video_processor.process_video(
                local_path, str(self.staging_dir / asset_id), target_duration=4
            )
            if not video_info.get("valid"):
                self._reject(asset_id, "invalid_video", video_info.get("error"))
                return None
            width, height = int(video_info["width"]), int(video_info["height"])
            quality_metrics = None
            if min_resolution and (width < min_resolution[0] or height < min_resolution[1]):
                self._reject(asset_id, "below_minimum_resolution", f"{width}x{height}")
                return None
            phash = None
            duration = float(video_info["duration"])
        else:
            quality_metrics = assess_image_quality(
                local_path,
                min_width=(min_resolution or (512, 512))[0],
                min_height=(min_resolution or (512, 512))[1],
            )
            width, height = quality_metrics.width, quality_metrics.height
            phash = quality_metrics.phash
            duration = 0.0
        if quality_metrics and not quality_metrics.passed_qc:
            rej_path = self.rejected_dir / f"{asset_id}_qc_failed.json"
            rej_path.write_text(json.dumps({"asset_id": asset_id, "reasons": quality_metrics.failure_reasons}))
            return None

        # 6. Deduplication against processed assets
        for existing_meta in self.metadata_dir.glob("*.json"):
            try:
                data = json.loads(existing_meta.read_text())
                if data.get("sha256") == sha256:
                    return None  # Exact duplicate
                if phash and data.get("phash"):
                    dist = hamming_distance(phash, data["phash"])
                    if dist <= 3:
                        return None  # Perceptual duplicate
            except Exception:
                continue

        # 7. Automated Captioning
        category = cand.get("category", "general")
        caption_meta = auto_captioner.generate_caption(query or category, context={"subject": category})

        # 8. Commit to Processed and write metadata manifest
        proc_dest = self.processed_dir / f"{asset_id}{Path(local_path).suffix.lower()}"
        if str(proc_dest) != local_path:
            shutil.copy2(local_path, proc_dest)

        asset = AssetMetadata(
            asset_id=asset_id,
            source_url=src_url,
            source_name=cand.get("source_name", "synthetic"),
            license=lic_info.license,
            license_url=cand.get("license_url"),
            creator=cand.get("creator", "Unknown"),
            media_type=media_type,
            resolution=f"{width}x{height}",
            duration=duration,
            sha256=sha256,
            phash=phash,
            allowed_for_training=True,
            category=category,
            local_path=str(proc_dest),
            quality_metrics=quality_metrics.__dict__ if quality_metrics else {"video": video_info},
            caption_metadata=caption_meta.__dict__,
        )

        meta_dest = self.metadata_dir / f"{asset_id}.json"
        meta_dest.write_text(json.dumps(asset.to_dict(), indent=2))
        self._write_annotation(asset)

        return asset

    def _write_annotation(self, asset: AssetMetadata) -> None:
        """Persist prompt-oriented labels independently from source provenance."""
        caption = asset.caption_metadata or {}
        annotation = {
            "asset_id": asset.asset_id,
            "caption": caption.get("caption", ""),
            "objects": caption.get("objects") or caption.get("subjects", []),
            "actions": caption.get("actions", []),
            "environment": caption.get("environment", "general"),
            "weather": caption.get("weather", "unknown"),
            "time": caption.get("time", "unknown"),
            "camera": caption.get("camera", "unknown"),
            "motion": caption.get("motion", "unknown"),
        }
        (self.annotations_dir / f"{asset.asset_id}.json").write_text(
            json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _reject(self, asset_id: str, reason: str, details: Optional[str] = None) -> None:
        payload = {"asset_id": asset_id, "reason": reason}
        if details:
            payload["details"] = details
        (self.rejected_dir / f"{asset_id}_{reason}.json").write_text(json.dumps(payload), encoding="utf-8")

    def generate_manifests(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> QualityReport:
        """
        Partitions all processed metadata into train.jsonl, validation.jsonl, and test.jsonl.
        Generates dataset_report.json summary.
        """
        all_metas = []
        for f in self.metadata_dir.glob("*.json"):
            try:
                all_metas.append(json.loads(f.read_text()))
            except Exception:
                continue

        # Split by source parent, rather than individual clips/assets. This
        # prevents adjacent clips or re-encodes of a source video leaking into
        # validation/test. A stable hash makes results reproducible.
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in all_metas:
            group_id = str(item.get("source_group_id") or item.get("parent_asset_id") or item.get("asset_id"))
            groups.setdefault(group_id, []).append(item)
        ordered_groups = sorted(groups.items(), key=lambda pair: __import__("hashlib").sha256(f"ttv-v001:{pair[0]}".encode()).hexdigest())
        targets = {"train": len(all_metas) * train_ratio, "validation": len(all_metas) * val_ratio, "test": len(all_metas) * test_ratio}
        split_map: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        for _, group in ordered_groups:
            bucket = min(split_map, key=lambda name: len(split_map[name]) / max(targets[name], 1))
            split_map[bucket].extend(group)
        train_set, val_set, test_set = split_map["train"], split_map["validation"], split_map["test"]

        manifest_paths = [
            (self.manifests_dir / "train.jsonl", train_set),
            (self.manifests_dir / "validation.jsonl", val_set),
            (self.manifests_dir / "test.jsonl", test_set),
            (Path("datasets/manifests/train.jsonl"), train_set),
            (Path("datasets/manifests/validation.jsonl"), val_set),
            (Path("datasets/manifests/test.jsonl"), test_set),
        ]

        for p, data_list in manifest_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as out_f:
                for item in data_list:
                    manifest_record = {
                        "id": item.get("asset_id"),
                        "media_path": item.get("local_path"),
                        "caption": item.get("caption_metadata", {}).get("caption", ""),
                        "language": "en",
                        "resolution": item.get("resolution", "1280x720"),
                        "duration": item.get("duration", 0.0),
                        "fps": item.get("quality_metrics", {}).get("video", {}).get("fps"),
                        "license": item.get("license"),
                        "source": item.get("source_name"),
                        "training_eligible": item.get("allowed_for_training", True),
                    }
                    caption = item.get("caption_metadata", {})
                    manifest_record.update({
                        "video": item.get("local_path") if item.get("media_type") == "video" else None,
                        "frames": item.get("quality_metrics", {}).get("video", {}).get("keyframes", []),
                        "objects": caption.get("objects") or caption.get("subjects", []),
                        "actions": caption.get("actions", []),
                        "environment": caption.get("environment"),
                        "weather": caption.get("weather"),
                        "time": caption.get("time"),
                        "camera": caption.get("camera"),
                        "motion": caption.get("motion"),
                    })
                    out_f.write(json.dumps(manifest_record) + "\n")

        # Compile Quality Report
        rejected_count = len(list(self.rejected_dir.glob("*.json")))
        categories_count = {}
        for item in all_metas:
            c = item.get("category", "general")
            categories_count[c] = categories_count.get(c, 0) + 1

        report = QualityReport(
            total_downloaded=n + rejected_count,
            valid=n,
            duplicates=0,
            low_quality=rejected_count,
            license_unknown=0,
            corrupted=0,
            training_ready=n,
            categories=categories_count,
        )

        report_dest = Path("dataset_report.json")
        report_dest.write_text(json.dumps(report.to_dict(), indent=2))
        (self.base_dir / "dataset_report.json").write_text(json.dumps(report.to_dict(), indent=2))

        return report


dataset_pipeline = DatasetPipeline()
