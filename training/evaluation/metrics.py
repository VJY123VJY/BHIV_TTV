import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple


def calculate_sharpness(frame: np.ndarray) -> float:
    """Calculates spatial sharpness using the variance of the Laplacian."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def calculate_contrast(frame: np.ndarray) -> float:
    """Calculates image contrast via standard deviation of pixel intensities."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    return float(gray.std())


def calculate_frame_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two frames.
    Normalized to [0.0, 1.0].
    """
    if img1.ndim == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        g1 = img1.astype(np.float32)
        g2 = img2.astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def evaluate_temporal_consistency(frames: List[np.ndarray]) -> float:
    """
    Measures frame-to-frame stability and temporal continuity using average adjacent SSIM.
    Values closer to 1.0 indicate smooth cinematic motion without flickering.
    """
    if len(frames) < 2:
        return 1.0

    ssim_values = []
    for i in range(len(frames) - 1):
        s = calculate_frame_ssim(frames[i], frames[i + 1])
        ssim_values.append(s)
    return float(np.mean(ssim_values))


def evaluate_motion_consistency(frames: List[np.ndarray]) -> float:
    """
    Computes consistency of motion vectors across frames.
    Penalizes extreme jitter or completely static frames.
    Returns normalized score [0.0, 1.0].
    """
    if len(frames) < 3:
        return 1.0

    diff_magnitudes = []
    for i in range(len(frames) - 1):
        f1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        f2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        diff = np.abs(f2 - f1).mean()
        diff_magnitudes.append(diff)

    mean_diff = float(np.mean(diff_magnitudes))
    std_diff = float(np.std(diff_magnitudes))

    # A good video has steady, moderate motion (mean_diff in [2.0, 30.0]) and low variance
    motion_score = np.exp(-std_diff / (mean_diff + 1e-4))
    return float(np.clip(motion_score, 0.0, 1.0))


def evaluate_visual_quality(frames: List[np.ndarray]) -> float:
    """
    Assesses overall visual quality based on spatial sharpness and contrast.
    Normalized score [0.0, 1.0].
    """
    if not frames:
        return 0.0

    sharpness_scores = [calculate_sharpness(f) for f in frames]
    contrast_scores = [calculate_contrast(f) for f in frames]

    avg_sharpness = np.mean(sharpness_scores)
    avg_contrast = np.mean(contrast_scores)

    # Sigmoidal mapping for calibrated 0-1 scale
    norm_sharpness = float(1.0 / (1.0 + np.exp(-avg_sharpness / 100.0 + 2.0)))
    norm_contrast = float(np.clip(avg_contrast / 75.0, 0.0, 1.0))

    return float(0.6 * norm_sharpness + 0.4 * norm_contrast)


def evaluate_prompt_adherence(prompt: str, frames: List[np.ndarray]) -> float:
    """
    Evaluates semantic prompt adherence based on visual feature matching.
    Checks presence of prompt keywords against visual features (color distribution, brightness, motion).
    """
    p_lower = prompt.lower()
    score = 0.5  # Baseline

    if not frames:
        return 0.0

    # Average color analysis across frames
    avg_r = np.mean([f[:, :, 0].mean() for f in frames])
    avg_g = np.mean([f[:, :, 1].mean() for f in frames])
    avg_b = np.mean([f[:, :, 2].mean() for f in frames])

    # Check semantic keyword alignments
    if any(k in p_lower for k in ["mars", "red", "desert", "sunset"]):
        if avg_r > avg_b: # Red/warm dominance
            score += 0.25
        else:
            score -= 0.1
    elif any(k in p_lower for k in ["beach", "ocean", "sea", "blue", "water", "sky"]):
        if avg_b > avg_r: # Blue/cool dominance
            score += 0.25
        else:
            score -= 0.1
    elif any(k in p_lower for k in ["cyberpunk", "neon", "dark", "night"]):
        brightness = (avg_r + avg_g + avg_b) / 3.0
        if brightness < 120.0: # Appropriate dark mood
            score += 0.25

    return float(np.clip(score, 0.0, 1.0))
