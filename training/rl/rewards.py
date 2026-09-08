from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from training.evaluation.metrics import (
    evaluate_prompt_adherence,
    evaluate_temporal_consistency,
    evaluate_visual_quality,
    evaluate_motion_consistency
)


class BaseRewardModel(ABC):
    """Abstract base class for modular video reward calculation."""
    @abstractmethod
    def calculate_reward(self, prompt: str, frames: List[np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> float:
        """Returns normalized scalar reward in range [0.0, 1.0]."""
        pass


class PromptAdherenceRewardModel(BaseRewardModel):
    """Rewards semantic alignment between prompt and visual frames."""
    def calculate_reward(self, prompt: str, frames: List[np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> float:
        return evaluate_prompt_adherence(prompt, frames)


class TemporalSmoothnessRewardModel(BaseRewardModel):
    """Rewards cinematic temporal consistency and penalizes erratic jitter."""
    def calculate_reward(self, prompt: str, frames: List[np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> float:
        s_temporal = evaluate_temporal_consistency(frames)
        s_motion = evaluate_motion_consistency(frames)
        return float(0.6 * s_temporal + 0.4 * s_motion)


class AestheticVisualRewardModel(BaseRewardModel):
    """Rewards high spatial sharpness, rich contrast, and color balance."""
    def calculate_reward(self, prompt: str, frames: List[np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> float:
        return evaluate_visual_quality(frames)


class HumanPreferenceRewardModel(BaseRewardModel):
    """
    Incorporates human preference feedback (ratings 1 to 5, normalized to [0, 1]).
    If no human score is provided in metadata, returns default neutral score.
    """
    def calculate_reward(self, prompt: str, frames: List[np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> float:
        if metadata and "human_rating" in metadata:
            rating = float(metadata["human_rating"])
            # Map 1-5 scale to 0.0 - 1.0
            return float(np.clip((rating - 1.0) / 4.0, 0.0, 1.0))
        return 0.5


class CompositeRewardModel(BaseRewardModel):
    """
    Weighted combination of individual reward models:
    R = w_prompt * R_prompt + w_temp * R_temporal + w_aesthetic * R_aesthetic + w_human * R_human
    """
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        self.weights = weights or {
            "prompt": 0.35,
            "temporal": 0.35,
            "aesthetic": 0.20,
            "human": 0.10
        }
        self.models: Dict[str, BaseRewardModel] = {
            "prompt": PromptAdherenceRewardModel(),
            "temporal": TemporalSmoothnessRewardModel(),
            "aesthetic": AestheticVisualRewardModel(),
            "human": HumanPreferenceRewardModel()
        }

    def register_reward_model(self, name: str, model: BaseRewardModel, weight: float):
        """Allows plugging in additional external reward models dynamically."""
        self.models[name] = model
        self.weights[name] = weight

    def calculate_reward(self, prompt: str, frames: List[np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> float:
        total = 0.0
        weight_sum = 0.0
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w > 0:
                r = model.calculate_reward(prompt, frames, metadata)
                total += w * r
                weight_sum += w
        return float(total / weight_sum) if weight_sum > 0 else 0.0

    def calculate_breakdown(self, prompt: str, frames: List[np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Returns individual sub-rewards and final composite reward."""
        breakdown = {}
        for name, model in self.models.items():
            breakdown[name] = round(model.calculate_reward(prompt, frames, metadata), 4)
        breakdown["composite"] = round(self.calculate_reward(prompt, frames, metadata), 4)
        return breakdown
