import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app.adapters.base import BaseVideoAdapter
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.models.scene import Scene
from app.core.config import settings
from app.core.logging import telemetry, logger
from app.core.exceptions import VisualGenerationError

from training.fine_tuning.model import SpatialTemporalTTVModel
from training.fine_tuning.lora import apply_lora_to_model, load_lora_state_dict
from training.preprocessing.video_preprocessor import VideoPreprocessor
from training.preprocessing.text_preprocessor import TextPreprocessor


class NeuralVideoAdapter(BaseVideoAdapter):
    """
    Fine-Tuned Neural Video Generation Adapter.
    Uses trained SpatialTemporalTTVModel with LoRA weights to synthesize
    neural motion trajectories from scene keyframes and prompts.
    Falls back gracefully to OpenCVVideoAdapter if weights cannot be loaded.
    """
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.fallback = OpenCVVideoAdapter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or settings.FINE_TUNED_CHECKPOINT_PATH
        self.model: Optional[SpatialTemporalTTVModel] = None
        self.preprocessor = VideoPreprocessor(num_frames=8, height=128, width=128)
        self.text_preprocessor = TextPreprocessor(max_length=32)

        self._init_model()

    def _find_default_checkpoint(self) -> Optional[str]:
        """Locates the latest trained checkpoint if none explicitly configured."""
        lora_dir = settings.get_absolute_path("models/lora")
        if lora_dir.exists():
            for p in sorted(lora_dir.glob("*/adapter_weights.pt"), reverse=True):
                return str(p)
        ckpt_dir = settings.get_absolute_path("training/checkpoints")
        if ckpt_dir.exists():
            for p in sorted(ckpt_dir.glob("*.pt"), reverse=True):
                return str(p)
        return None

    def _init_model(self):
        target_ckpt = self.checkpoint_path or self._find_default_checkpoint()
        if not target_ckpt or not os.path.exists(target_ckpt):
            logger.warning(f"[NeuralVideoAdapter] No checkpoint found at '{target_ckpt}'. Operating in procedural fallback mode.")
            return

        try:
            # Instantiate model architecture
            self.model = SpatialTemporalTTVModel(
                in_channels=3,
                latent_channels=64,
                num_frames=8,
                vocab_size=10000,
                max_prompt_length=32,
                text_embed_dim=128
            ).to(self.device)

            ckpt = torch.load(target_ckpt, map_location=self.device)
            if ckpt.get("use_lora", True):
                apply_lora_to_model(self.model, rank=ckpt.get("config", {}).get("lora_rank", 4))
                if "lora_state_dict" in ckpt:
                    load_lora_state_dict(self.model, ckpt["lora_state_dict"])
                elif "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(ckpt["model_state_dict"])

            self.model.eval()
            logger.info(f"[NeuralVideoAdapter] Successfully loaded fine-tuned checkpoint: {target_ckpt}")
        except Exception as e:
            logger.error(f"[NeuralVideoAdapter] Error loading checkpoint {target_ckpt}: {e}. Falling back to OpenCV engine.")
            self.model = None

    async def generate_scene_video(
        self,
        scene: Scene,
        keyframe_path: str,
        output_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24
    ) -> str:
        if self.model is None:
            telemetry.emit("neural_video_generation", scene.title, {"status": "fallback_opencv"})
            return await self.fallback.generate_scene_video(scene, keyframe_path, output_path, width, height, fps)

        telemetry.emit("neural_video_generation", scene.title, {"engine": "spatial_temporal_lora", "device": str(self.device)})
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not os.path.exists(keyframe_path):
            raise VisualGenerationError(f"Keyframe image missing: {keyframe_path}")

        img_bgr = cv2.imread(keyframe_path)
        if img_bgr is None:
            raise VisualGenerationError(f"Could not load keyframe image: {keyframe_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Preprocess keyframe to model input resolution
        resized_kf = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        kf_tensor = torch.from_numpy(np.transpose((resized_kf.astype(np.float32) / 127.5) - 1.0, (2, 0, 1))).unsqueeze(0).float().to(self.device)

        # 2. Tokenize prompt
        prompt_text = scene.metadata.get("enriched_prompt", scene.visual_description)
        tokens = self.text_preprocessor.tokenize_to_tensor(prompt_text).unsqueeze(0).to(self.device)

        # 3. Model inference
        with torch.no_grad():
            pred_frames, _, _ = self.model(kf_tensor, tokens)
            pred_frames = pred_frames.squeeze(0) # (T, C, H, W)

        # 4. Denormalize frames
        model_frames = [
            self.preprocessor.denormalize_tensor(pred_frames[t])
            for t in range(pred_frames.shape[0])
        ]

        # 5. Temporal interpolation to match requested scene duration
        total_output_frames = max(12, int(scene.duration * fps))
        indices = np.linspace(0, len(model_frames) - 1, total_output_frames)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
            if not out.isOpened():
                raise VisualGenerationError(f"Failed to open VideoWriter for {output_path}")

        for idx_float in indices:
            low_idx = int(np.floor(idx_float))
            high_idx = min(len(model_frames) - 1, int(np.ceil(idx_float)))
            alpha = idx_float - low_idx

            frame_a = model_frames[low_idx]
            frame_b = model_frames[high_idx]
            interp = cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0.0)

            # Upscale to requested target resolution
            upscaled = cv2.resize(interp, (width, height), interpolation=cv2.INTER_LANCZOS4)
            out_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
            out.write(out_bgr)

        out.release()
        return output_path
