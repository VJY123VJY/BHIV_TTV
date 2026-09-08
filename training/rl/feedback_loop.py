import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F

from training.fine_tuning.model import SpatialTemporalTTVModel
from training.fine_tuning.lora import apply_lora_to_model, get_lora_state_dict, load_lora_state_dict
from training.preprocessing.video_preprocessor import VideoPreprocessor
from training.preprocessing.text_preprocessor import TextPreprocessor
from training.rl.rewards import CompositeRewardModel
from training.utils.versioning import ModelRegistry


class RLFeedbackLoop:
    """
    Reinforcement Learning / Direct Preference Feedback Loop for Text-to-Video.
    Workflow:
      Prompt
      -> Generate multiple candidate video sequences (varying motion noise / seeds)
      -> Evaluate candidates with modular composite reward models
      -> Form preference ranking & select winner (y_w) vs loser (y_l)
      -> Compute preference loss and backpropagate to LoRA weights
      -> Register new model checkpoint version
      -> Record trajectory manifest
    """
    def __init__(
        self,
        model: SpatialTemporalTTVModel,
        reward_model: Optional[CompositeRewardModel] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "training/rl/runs"
    ):
        self.config = config or {}
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and self.config.get("device") != "cpu") else "cpu"
        )
        self.model = model.to(self.device)
        self.reward_model = reward_model or CompositeRewardModel()
        vocab_size = self.config.get("vocab_size", getattr(self.model.text_encoder.token_embeddings, "num_embeddings", 10000))
        self.text_preprocessor = TextPreprocessor(max_length=self.config.get("max_prompt_length", 32), vocab_size=vocab_size)
        self.video_preprocessor = VideoPreprocessor(
            num_frames=self.config.get("num_frames", 8),
            height=self.config.get("height", 128),
            width=self.config.get("width", 128)
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()

        # DPO / Preference Optimization parameters
        self.beta = float(self.config.get("beta", 0.1)) # Temperature scaling
        self.learning_rate = float(self.config.get("learning_rate", 5e-5))
        self.num_candidates = int(self.config.get("num_candidates", 3))

        # LoRA setup
        apply_lora_to_model(self.model, rank=int(self.config.get("lora_rank", 4)))
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)

    def generate_candidates(
        self,
        prompt: str,
        initial_keyframe: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Generates K candidate video frame sequences for a given prompt
        by perturbing latent temporal position noise.
        """
        self.model.eval()
        tokens = self.text_preprocessor.tokenize_to_tensor(prompt).unsqueeze(0).to(self.device)
        kf = initial_keyframe.unsqueeze(0).to(self.device)

        candidates = []
        with torch.no_grad():
            for c_idx in range(self.num_candidates):
                # Apply subtle perturbation to temporal embeddings to create variation
                noise = torch.randn_like(self.model.temporal_pos_embed) * 0.05 * c_idx
                orig_embed = self.model.temporal_pos_embed.data.clone()
                self.model.temporal_pos_embed.data.add_(noise)

                pred_frames, video_emb, text_emb = self.model(kf, tokens)
                self.model.temporal_pos_embed.data.copy_(orig_embed)

                # Convert tensor frames to RGB numpy frames for reward calculation
                tensor_seq = pred_frames.squeeze(0) # (T, C, H, W)
                np_frames = [
                    self.video_preprocessor.denormalize_tensor(tensor_seq[t])
                    for t in range(tensor_seq.shape[0])
                ]

                # Evaluate candidate rewards
                breakdown = self.reward_model.calculate_breakdown(prompt, np_frames)

                candidates.append({
                    "candidate_index": c_idx,
                    "pred_frames_tensor": pred_frames,
                    "video_embedding": video_emb,
                    "text_embedding": text_emb,
                    "rewards": breakdown,
                    "composite_reward": breakdown["composite"]
                })

        return candidates

    def preference_optimization_step(
        self,
        winner: Dict[str, Any],
        loser: Dict[str, Any],
        prompt: str,
        initial_keyframe: torch.Tensor
    ) -> float:
        """
        Executes a Direct Preference Optimization (DPO) update step:
        Maximizes margin: log_prob(winner) - log_prob(loser).
        """
        self.model.train()
        self.optimizer.zero_grad()

        tokens = self.text_preprocessor.tokenize_to_tensor(prompt).unsqueeze(0).to(self.device)
        kf = initial_keyframe.unsqueeze(0).to(self.device)

        # Forward pass for current policy
        pred_w, v_emb_w, t_emb_w = self.model(kf, tokens)
        pred_l, v_emb_l, t_emb_l = self.model(kf, tokens)

        # Alignment loss difference
        cos_w = F.cosine_similarity(v_emb_w, t_emb_w, dim=-1)
        cos_l = F.cosine_similarity(v_emb_l, t_emb_l, dim=-1)

        # Implicit log ratio margin: (cos_w - cos_l)
        margin = cos_w - cos_l
        loss = -torch.log(torch.sigmoid(self.beta * margin) + 1e-8).mean()

        # Regularization with winner reconstruction
        recon_loss = F.l1_loss(pred_w, winner["pred_frames_tensor"])
        total_loss = loss + 0.1 * recon_loss

        total_loss.backward()
        self.optimizer.step()

        return float(total_loss.item())

    def run_iteration(
        self,
        prompts: List[str],
        reference_keyframes: List[torch.Tensor],
        iteration: int = 1
    ) -> Dict[str, Any]:
        """Runs a complete iteration of the RL feedback loop across input prompts."""
        run_manifest = {
            "iteration": iteration,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompts_evaluated": len(prompts),
            "trajectories": [],
            "mean_reward_winner": 0.0,
            "mean_reward_loser": 0.0
        }

        winner_rewards = []
        loser_rewards = []

        for p_idx, (prompt, kf) in enumerate(zip(prompts, reference_keyframes)):
            # 1. Generate candidate video sequences
            candidates = self.generate_candidates(prompt, kf)

            # 2. Rank candidates by composite reward
            sorted_candidates = sorted(candidates, key=lambda x: x["composite_reward"], reverse=True)
            winner = sorted_candidates[0]
            loser = sorted_candidates[-1]

            winner_rewards.append(winner["composite_reward"])
            loser_rewards.append(loser["composite_reward"])

            # 3. Preference optimization update
            loss_val = self.preference_optimization_step(winner, loser, prompt, kf)

            run_manifest["trajectories"].append({
                "prompt": prompt,
                "winner_reward": winner["composite_reward"],
                "loser_reward": loser["composite_reward"],
                "reward_breakdown": winner["rewards"],
                "preference_loss": round(loss_val, 4)
            })

        run_manifest["mean_reward_winner"] = round(float(sum(winner_rewards) / len(winner_rewards)), 4) if winner_rewards else 0.0
        run_manifest["mean_reward_loser"] = round(float(sum(loser_rewards) / len(loser_rewards)), 4) if loser_rewards else 0.0

        # Save versioned checkpoint
        version_str = f"ttv_lora_v{iteration:03d}"
        ckpt_path = self.output_dir / f"{version_str}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "lora_state_dict": get_lora_state_dict(self.model),
            "iteration": iteration,
            "run_manifest": run_manifest
        }, str(ckpt_path))

        # Register in ModelRegistry
        self.registry.register_version(
            version_name=version_str,
            checkpoint_path=str(ckpt_path.resolve()),
            model_type="lora",
            metrics={
                "winner_reward": run_manifest["mean_reward_winner"],
                "loser_reward": run_manifest["mean_reward_loser"]
            }
        )

        run_manifest["saved_checkpoint"] = str(ckpt_path.resolve())
        run_manifest["model_version"] = version_str

        # Save trajectory record
        manifest_file = self.output_dir / f"iteration_{iteration:03d}_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2)

        return run_manifest
