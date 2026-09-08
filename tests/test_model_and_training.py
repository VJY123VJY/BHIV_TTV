import os
import sys
import tempfile
import torch
import torch.nn as nn
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from training.fine_tuning.model import SpatialTemporalTTVModel
from training.fine_tuning.lora import apply_lora_to_model, get_lora_state_dict, load_lora_state_dict
from training.fine_tuning.trainer import TTVLoss, TTVTrainer
from training.datasets.loader import create_dataloader


def test_spatial_temporal_model_forward():
    model = SpatialTemporalTTVModel(
        in_channels=3,
        latent_channels=16,
        num_frames=4,
        vocab_size=500,
        max_prompt_length=16,
        text_embed_dim=32
    )

    batch_size = 2
    keyframe = torch.randn(batch_size, 3, 64, 64)
    tokens = torch.randint(0, 500, (batch_size, 16))

    pred_frames, video_emb, text_emb = model(keyframe, tokens)
    assert pred_frames.shape == (batch_size, 4, 3, 64, 64)
    assert video_emb.shape == (batch_size, 32)
    assert text_emb.shape == (batch_size, 32)
    assert pred_frames.min() >= -1.0
    assert pred_frames.max() <= 1.0


def test_lora_parameter_freezing_and_gradient_flow():
    model = SpatialTemporalTTVModel(
        in_channels=3,
        latent_channels=16,
        num_frames=4,
        vocab_size=500,
        max_prompt_length=16,
        text_embed_dim=32
    )

    # Before LoRA, all parameters require grad
    assert all(p.requires_grad for p in model.parameters())

    # Apply LoRA
    apply_lora_to_model(model, rank=4, alpha=8.0)

    # Base parameters should be frozen, LoRA parameters trainable
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert len(trainable) > 0
    assert len(frozen) > 0
    assert len(trainable) < len(frozen)

    # Perform forward + backward pass and verify gradients
    keyframe = torch.randn(1, 3, 64, 64)
    tokens = torch.randint(0, 500, (1, 16))
    pred_frames, video_emb, text_emb = model(keyframe, tokens)

    criterion = TTVLoss()
    target = torch.randn(1, 4, 3, 64, 64)
    loss, metrics = criterion(pred_frames, target, video_emb, text_emb)

    loss.backward()

    # Verify that gradients exist on trainable LoRA weights
    has_nonzero_grad = False
    for p in trainable:
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_nonzero_grad = True
            break
    assert has_nonzero_grad is True, "LoRA parameters received zero gradients!"


def test_checkpoint_saving_and_resuming():
    model = SpatialTemporalTTVModel(
        in_channels=3,
        latent_channels=16,
        num_frames=4,
        vocab_size=500,
        max_prompt_length=16,
        text_embed_dim=32
    )
    apply_lora_to_model(model, rank=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pt")
        state = {
            "epoch": 2,
            "global_step": 50,
            "model_state_dict": model.state_dict(),
            "use_lora": True,
            "lora_state_dict": get_lora_state_dict(model),
            "config": {"test": True}
        }
        torch.save(state, ckpt_path)
        assert os.path.exists(ckpt_path)

        # Load into fresh model
        new_model = SpatialTemporalTTVModel(
            in_channels=3,
            latent_channels=16,
            num_frames=4,
            vocab_size=500,
            max_prompt_length=16,
            text_embed_dim=32
        )
        apply_lora_to_model(new_model, rank=4)
        loaded = torch.load(ckpt_path)
        load_lora_state_dict(new_model, loaded["lora_state_dict"])
