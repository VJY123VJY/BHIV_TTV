"""Model evaluation package for Text-to-Video."""
from training.evaluation.metrics import (
    evaluate_temporal_consistency,
    evaluate_motion_consistency,
    evaluate_visual_quality,
    evaluate_prompt_adherence
)
from training.evaluation.evaluator import TTVEvaluator
