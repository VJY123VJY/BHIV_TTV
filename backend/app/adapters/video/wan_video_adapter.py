"""
Wan 2.2 TI2V-5B Video Generation Adapter
==========================================
Uses HuggingFace diffusers WanImageToVideoPipeline (Wan-AI/Wan2.2-TI2V-5B-Diffusers).

Hardware requirements
---------------------
* NVIDIA GPU with CUDA is REQUIRED for inference.
* This adapter cannot run on CPU or Intel/AMD integrated graphics.
* On a local machine without NVIDIA GPU, a clear VisualGenerationError is raised.
* Inference is intended to run on a cloud NVIDIA GPU (e.g. RunPod, Lambda Labs, Colab Pro).

Cloud deployment
----------------
Set the following in your cloud .env:
    VIDEO_PROVIDER=wan
    WAN_MODEL_ID=Wan-AI/Wan2.2-TI2V-5B-Diffusers
    WAN_DEVICE=cuda
    WAN_DTYPE=bfloat16
    HF_TOKEN=<your_huggingface_token>       # optional if model is public

The model is loaded lazily on the first request — NOT at application startup.

Supported modes
---------------
* Image-to-Video (primary): uses scene keyframe image + text prompt
* Text-to-Video (fallback): synthesises a neutral background PIL image from the
  scene's visual description when no keyframe image is available.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from app.adapters.base import BaseVideoAdapter
from app.models.scene import Scene
from app.core.config import settings
from app.core.logging import telemetry, logger
from app.core.exceptions import VisualGenerationError

# torch is imported at module level so tests can mock torch.cuda.is_available.
# If torch is not installed, the ImportError is surfaced at inference time
# (inside generate_scene_video) rather than at application startup.
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from diffusers import WanImageToVideoPipeline  # noqa: F401 — type hint only



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUDA_ERROR_MESSAGE = (
    "\n"
    "╔══════════════════════════════════════════════════════════════════════╗\n"
    "║          Wan 2.2 TI2V-5B — NVIDIA GPU Required                      ║\n"
    "╠══════════════════════════════════════════════════════════════════════╣\n"
    "║  Wan inference requires an NVIDIA CUDA GPU.                         ║\n"
    "║  Your current environment does NOT have a CUDA-capable GPU.         ║\n"
    "║                                                                      ║\n"
    "║  Recommended options:                                                ║\n"
    "║  1. Deploy on a cloud NVIDIA GPU (RunPod, Lambda Labs, Colab Pro).  ║\n"
    "║  2. Switch to the local OpenCV provider for development:            ║\n"
    "║       Set VIDEO_PROVIDER=opencv in your .env file.                  ║\n"
    "╚══════════════════════════════════════════════════════════════════════╝\n"
)


def _check_cuda_available() -> None:
    """Raise VisualGenerationError with a clear message if CUDA is not available."""
    if not _TORCH_AVAILABLE or _torch is None:
        raise VisualGenerationError(
            "torch is not installed. "
            "Install it for your CUDA version before using the Wan adapter.\n"
            "See: https://pytorch.org/get-started/locally/"
        )
    if not _torch.cuda.is_available():
        raise VisualGenerationError(_CUDA_ERROR_MESSAGE)



def _build_synthetic_image(scene: Scene, width: int, height: int):
    """
    Create a solid-colour PIL Image representing the scene's mood when no
    keyframe is available (text-to-video mode).
    The colour is derived heuristically from the scene description keywords.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise VisualGenerationError(
            "Pillow is required for Wan adapter. Install with: pip install Pillow"
        ) from exc

    desc = (scene.visual_description or "").lower()
    # Heuristic colour palette based on common scene types
    if any(k in desc for k in ["night", "dark", "space", "galaxy", "cosmos"]):
        colour = (10, 10, 30)
    elif any(k in desc for k in ["desert", "sand", "mars", "arid"]):
        colour = (194, 148, 90)
    elif any(k in desc for k in ["ocean", "sea", "water", "beach"]):
        colour = (30, 80, 160)
    elif any(k in desc for k in ["forest", "jungle", "tree", "green"]):
        colour = (30, 90, 40)
    elif any(k in desc for k in ["snow", "ice", "arctic", "winter", "blizzard"]):
        colour = (220, 235, 255)
    elif any(k in desc for k in ["sunset", "golden", "dusk", "twilight"]):
        colour = (200, 110, 50)
    else:
        colour = (80, 80, 100)  # neutral cinematic grey-blue

    return Image.new("RGB", (width, height), colour)


def _frames_to_mp4(frames, output_path: str, fps: int, width: int, height: int) -> None:
    """
    Write a list of PIL Images or numpy arrays to an MP4 file using OpenCV.
    Falls back to imageio if OpenCV VideoWriter fails to open.
    """
    try:
        from PIL import Image as _PILImage
        _pil_available = True
    except ImportError:
        _pil_available = False

    # --- Convert frames to numpy BGR arrays ---
    bgr_frames = []
    for f in frames:
        if _pil_available and isinstance(f, _PILImage.Image):
            arr = np.array(f.convert("RGB"))
        elif isinstance(f, np.ndarray):
            arr = f
        else:
            raise VisualGenerationError(f"Unknown frame type from Wan pipeline: {type(f)}")
        # Resize to target resolution
        import cv2
        arr_resized = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LANCZOS4)
        bgr_frames.append(cv2.cvtColor(arr_resized, cv2.COLOR_RGB2BGR))

    if not bgr_frames:
        raise VisualGenerationError("Wan pipeline returned zero frames.")

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not out.isOpened():
            raise VisualGenerationError(
                f"OpenCV VideoWriter failed to open output path: {output_path}"
            )

    for frame in bgr_frames:
        out.write(frame)
    out.release()


def _interpolate_frames(frames: list, target_count: int) -> list:
    """
    Temporally interpolate a frame list to reach `target_count` frames.
    Uses linear index mapping (same strategy as NeuralVideoAdapter).
    """
    if len(frames) == 0:
        raise VisualGenerationError("Cannot interpolate — frame list is empty.")
    if target_count <= 0:
        raise VisualGenerationError("target_count must be positive.")
    if len(frames) == 1:
        return frames * target_count

    import cv2
    from PIL import Image as _PILImage

    indices = np.linspace(0, len(frames) - 1, target_count)
    result = []
    for idx_f in indices:
        lo = int(np.floor(idx_f))
        hi = min(len(frames) - 1, int(np.ceil(idx_f)))
        alpha = idx_f - lo
        if lo == hi or alpha == 0.0:
            result.append(frames[lo])
        else:
            # Blend two PIL frames
            a = np.array(frames[lo].convert("RGB")).astype(np.float32)
            b = np.array(frames[hi].convert("RGB")).astype(np.float32)
            blended = np.clip((1 - alpha) * a + alpha * b, 0, 255).astype(np.uint8)
            result.append(_PILImage.fromarray(blended))
    return result


# ---------------------------------------------------------------------------
# Main Adapter
# ---------------------------------------------------------------------------

class WanVideoAdapter(BaseVideoAdapter):
    """
    Wan 2.2 TI2V-5B Image-to-Video adapter for the BHIV TTV system.

    * Implements the same BaseVideoAdapter interface as OpenCVVideoAdapter.
    * Model is loaded lazily on the first inference call (NOT at startup).
    * Requires an NVIDIA CUDA GPU — raises VisualGenerationError otherwise.
    * Supports both image-to-video (primary) and text-to-video (synthetic bg).
    * Output is an MP4 file at the path expected by video_service.py.
    """

    def __init__(self) -> None:
        self._pipeline = None  # Loaded lazily
        self._model_id: str = settings.WAN_MODEL_ID
        self._device: str = settings.WAN_DEVICE
        self._dtype_str: str = settings.WAN_DTYPE
        self._num_frames: int = settings.WAN_NUM_FRAMES
        self._inference_steps: int = settings.WAN_INFERENCE_STEPS
        self._guidance_scale: float = settings.WAN_GUIDANCE_SCALE
        logger.info(
            f"[WanVideoAdapter] Initialised (model={self._model_id}, "
            f"device={self._device}, dtype={self._dtype_str}). "
            "Model will be loaded on first inference call."
        )

    # ------------------------------------------------------------------
    # Pipeline loading (called lazily)
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """
        Load the Wan 2.2 TI2V-5B pipeline from HuggingFace Hub or local path.
        This method is called once on the first inference request.
        """
        if self._pipeline is not None:
            return  # Already loaded

        logger.info(
            f"[WanVideoAdapter] Loading pipeline: {self._model_id} "
            f"on {self._device} / {self._dtype_str} ..."
        )

        try:
            import torch
            from diffusers import WanImageToVideoPipeline
            from diffusers.utils import export_to_video  # noqa: F401 — imported for validation
        except ImportError as exc:
            raise VisualGenerationError(
                "Required packages for Wan are not installed.\n"
                "Run: pip install diffusers>=0.33.0 transformers>=4.51.0 "
                "accelerate>=1.6.0 safetensors>=0.5.3"
            ) from exc

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype_str.lower(), torch.bfloat16)

        hf_token: Optional[str] = getattr(settings, "HF_TOKEN", None) or None

        try:
            self._pipeline = WanImageToVideoPipeline.from_pretrained(
                self._model_id,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
            self._pipeline.to(self._device)
            logger.info("[WanVideoAdapter] Pipeline loaded and moved to device successfully.")
        except Exception as exc:
            self._pipeline = None
            raise VisualGenerationError(
                f"Failed to load Wan pipeline '{self._model_id}': {exc}\n"
                "Check that:\n"
                "  1. diffusers>=0.33.0 is installed.\n"
                "  2. The model ID is correct and accessible on HuggingFace.\n"
                "  3. HF_TOKEN is set if the model requires authentication.\n"
                "  4. Sufficient VRAM is available (≥16GB recommended for 480p, "
                "≥24GB for 720p)."
            ) from exc

    # ------------------------------------------------------------------
    # Public interface (matches BaseVideoAdapter)
    # ------------------------------------------------------------------

    async def generate_scene_video(
        self,
        scene: Scene,
        keyframe_path: str,
        output_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
    ) -> str:
        """
        Generate a scene video using Wan 2.2 TI2V-5B.

        Parameters
        ----------
        scene : Scene
            Scene dataclass containing visual_description, narrative, duration,
            camera_motion, and metadata (including enriched_prompt if available).
        keyframe_path : str
            Absolute path to the keyframe/reference image (image-to-video mode).
            If the file does not exist, a synthetic background is used (text-to-video mode).
        output_path : str
            Absolute path where the output MP4 will be written.
        width, height : int
            Target output resolution. Note: Wan VRAM requirements scale with resolution.
        fps : int
            Frames per second for the output video.

        Returns
        -------
        str
            The output_path of the generated MP4.

        Raises
        ------
        VisualGenerationError
            If CUDA is unavailable, pipeline fails to load, or inference fails.
        """
        # ── 1. CUDA guard ──────────────────────────────────────────────
        _check_cuda_available()

        # ── 2. Prepare output directory ────────────────────────────────
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ── 3. Determine prompt ────────────────────────────────────────
        prompt: str = (
            scene.metadata.get("enriched_prompt")
            or scene.visual_description
            or scene.narrative
            or "cinematic video scene"
        )

        # ── 4. Load input image ────────────────────────────────────────
        image_mode: str
        if keyframe_path and os.path.exists(keyframe_path):
            try:
                from PIL import Image
                input_image = Image.open(keyframe_path).convert("RGB").resize(
                    (width, height)
                )
                image_mode = "image_to_video"
            except Exception as exc:
                raise VisualGenerationError(
                    f"[WanVideoAdapter] Failed to open keyframe image '{keyframe_path}': {exc}"
                ) from exc
        else:
            logger.warning(
                "[WanVideoAdapter] No valid keyframe found at '%s'. "
                "Falling back to synthetic background (text-to-video mode).",
                keyframe_path,
            )
            input_image = _build_synthetic_image(scene, width, height)
            image_mode = "text_to_video_synthetic"

        # ── 5. Lazy-load the pipeline ──────────────────────────────────
        self._load_pipeline()

        # ── 6. Run inference ───────────────────────────────────────────
        telemetry.emit(
            "wan_video_generation",
            scene.title,
            {
                "engine": "wan2.2-ti2v-5b",
                "mode": image_mode,
                "device": self._device,
                "dtype": self._dtype_str,
                "num_frames": self._num_frames,
                "resolution": f"{width}x{height}",
            },
        )

        logger.info(
            "[WanVideoAdapter] Running inference for scene '%s' "
            "(prompt=%r, mode=%s, frames=%d, steps=%d).",
            scene.title,
            prompt[:80],
            image_mode,
            self._num_frames,
            self._inference_steps,
        )

        try:
            import torch
            with torch.inference_mode():
                output = self._pipeline(
                    image=input_image,
                    prompt=prompt,
                    num_frames=self._num_frames,
                    num_inference_steps=self._inference_steps,
                    guidance_scale=self._guidance_scale,
                )
            wan_frames = output.frames[0]  # list of PIL Images
        except Exception as exc:
            raise VisualGenerationError(
                f"[WanVideoAdapter] Inference failed for scene '{scene.title}': {exc}"
            ) from exc

        # ── 7. Temporal interpolation to match scene duration ──────────
        target_frame_count = max(12, int(scene.duration * fps))
        if len(wan_frames) != target_frame_count:
            logger.info(
                "[WanVideoAdapter] Interpolating %d Wan frames → %d frames "
                "(scene duration=%.1fs @ %d fps).",
                len(wan_frames),
                target_frame_count,
                scene.duration,
                fps,
            )
            wan_frames = _interpolate_frames(wan_frames, target_frame_count)

        # ── 8. Write frames to MP4 ─────────────────────────────────────
        _frames_to_mp4(wan_frames, output_path, fps=fps, width=width, height=height)

        logger.info(
            "[WanVideoAdapter] Scene '%s' written to: %s",
            scene.title,
            output_path,
        )
        return output_path
