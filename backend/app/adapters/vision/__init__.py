"""
Vision and visual consistency adapters.
"""
from app.adapters.vision.consistency_adapter import VisualConsistencyAdapter

def get_vision_adapter(provider: str = "standard"):
    return VisualConsistencyAdapter()
