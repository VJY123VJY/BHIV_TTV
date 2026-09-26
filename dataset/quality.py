"""
Quality control, license verification, perceptual hashing, and deduplication engine.
"""
import os
import hashlib
from typing import Tuple, Optional, List
import cv2
import numpy as np
from PIL import Image
from dataset.registry import license_is_allowed

from dataset.models import LicenseInfo, QualityMetrics

# Permissive training licenses
PERMISSIVE_LICENSES = {
    "cc0",
    "public domain",
    "cc-by",
    "cc-by-sa",
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "open-database",
    "internal-synthetic",
    "user-authorized",
}

NON_TRAINING_LICENSES = {
    "cc-by-nc",
    "cc-by-nd",
    "cc-by-nc-sa",
    "cc-by-nc-nd",
    "all rights reserved",
    "copyrighted",
    "unknown",
}


def verify_license(license_str: Optional[str]) -> LicenseInfo:
    """Verifies whether an asset's license is legally permitted for AI model training."""
    if not license_str:
        return LicenseInfo(
            license="unknown",
            commercial_allowed=False,
            training_eligible=False,
        )

    norm = license_str.strip().lower()
    # The registry is the policy authority.  Keep the local sets for backwards
    # compatible diagnostics, but never expand the registry allowlist here.
    is_eligible = license_is_allowed(license_str)

    return LicenseInfo(
        license=license_str,
        commercial_allowed=is_eligible,
        training_eligible=is_eligible,
    )


def calculate_sha256(filepath: str) -> str:
    """Calculates cryptographic SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def calculate_phash(image_or_path, hash_size: int = 8) -> str:
    """
    Computes a 64-bit DCT perceptual hash (pHash) for image duplicate detection.
    Robust against resizing, slight compression, and watermarks.
    """
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_or_path, np.ndarray):
        img = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY) if len(image_or_path.shape) == 3 else image_or_path
    else:
        # PIL image
        img = np.array(image_or_path.convert("L"))

    if img is None:
        return "0" * (hash_size * hash_size // 4)

    # 1. Resize to 32x32
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)

    # 2. Compute 2D Discrete Cosine Transform (DCT)
    dct = cv2.dct(resized)

    # 3. Extract the top-left 8x8 lowest frequencies (excluding DC component at 0,0)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)

    # 4. Binary hash based on median
    bits = (dct_low > med).flatten()
    hash_hex = "".join(f"{int(''.join(map(str, bits[i:i+4].astype(int))), 2):x}" for i in range(0, len(bits), 4))
    return hash_hex


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculates bitwise Hamming distance between two hex hashes."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 64
    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    return bin(val1 ^ val2).count("1")


def assess_image_quality(
    image_path: str,
    min_width: int = 512,
    min_height: int = 512,
    blur_threshold: float = 80.0,
) -> QualityMetrics:
    """
    Audits image technical quality:
    - Verifies file integrity
    - Measures resolution
    - Evaluates blur via Laplacian variance
    - Detects blank / near-black frames
    """
    metrics = QualityMetrics()

    if not os.path.exists(image_path) or os.path.getsize(image_path) < 100:
        metrics.is_corrupt = True
        metrics.passed_qc = False
        metrics.failure_reasons.append("File corrupt or empty")
        return metrics

    img = cv2.imread(image_path)
    if img is None:
        metrics.is_corrupt = True
        metrics.passed_qc = False
        metrics.failure_reasons.append("Failed to decode image")
        return metrics

    h, w = img.shape[:2]
    metrics.width = w
    metrics.height = h
    metrics.aspect_ratio = "16:9" if w >= h else "9:16"

    # Resolution check
    if w < min_width or h < min_height:
        metrics.passed_qc = False
        metrics.failure_reasons.append(f"Resolution {w}x{h} below threshold {min_width}x{min_height}")

    # Blur detection via Laplacian variance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics.blur_score = round(blur_score, 2)
    if blur_score < blur_threshold:
        metrics.passed_qc = False
        metrics.failure_reasons.append(f"Image blurred (score {metrics.blur_score} < {blur_threshold})")

    # Black frame detection
    mean_luminance = float(np.mean(gray))
    if mean_luminance < 3.0:
        metrics.is_black_frame = True
        metrics.passed_qc = False
        metrics.failure_reasons.append("Pure black or underexposed frame")

    # Perceptual hash
    metrics.phash = calculate_phash(gray)

    return metrics
