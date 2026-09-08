import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 checksum of a file."""
    if not os.path.exists(file_path):
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ModelRegistry:
    """
    Manages versioned model artifacts and checkpoint manifests.
    Layout:
      models/
        base/
          version_info.json
        lora/
          ttv_lora_v001/
            adapter_weights.pt
            version_info.json
          ttv_lora_v002/
    """
    def __init__(self, models_root: Optional[str] = None):
        if models_root is None:
            # Default to text_to_video_final/models
            self.models_root = Path(__file__).resolve().parents[2] / "models"
        else:
            self.models_root = Path(models_root)
        self.models_root.mkdir(parents=True, exist_ok=True)
        (self.models_root / "base").mkdir(parents=True, exist_ok=True)
        (self.models_root / "lora").mkdir(parents=True, exist_ok=True)

    def register_version(
        self,
        version_name: str,
        checkpoint_path: str,
        base_model: str = "spatial_temporal_ttv_v1",
        dataset_version: str = "dataset_v1",
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        model_type: str = "lora"
    ) -> Dict[str, Any]:
        """Registers a trained model checkpoint with full provenance and metadata."""
        dest_dir = self.models_root / model_type / version_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Compute checksum
        weights_sha = compute_file_sha256(checkpoint_path) if checkpoint_path else ""

        manifest = {
            "version": version_name,
            "model_type": model_type,
            "base_model": base_model,
            "dataset_version": dataset_version,
            "checkpoint_path": str(checkpoint_path),
            "weights_sha256": weights_sha,
            "registered_at": datetime.utcnow().isoformat() + "Z",
            "training_config": config or {},
            "metrics": metrics or {}
        }

        manifest_path = dest_dir / "version_info.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def list_versions(self, model_type: str = "lora") -> List[Dict[str, Any]]:
        """Lists all registered versions for a given model type."""
        type_dir = self.models_root / model_type
        if not type_dir.exists():
            return []
        versions = []
        for p in type_dir.iterdir():
            if p.is_dir():
                info_path = p / "version_info.json"
                if info_path.exists():
                    try:
                        with open(info_path, "r", encoding="utf-8") as f:
                            versions.append(json.load(f))
                    except Exception:
                        pass
        return sorted(versions, key=lambda x: x.get("registered_at", ""))

    def get_version(self, version_name: str, model_type: str = "lora") -> Optional[Dict[str, Any]]:
        """Retrieves metadata manifest for a specific model version."""
        info_path = self.models_root / model_type / version_name / "version_info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
