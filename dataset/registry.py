"""License-gated source registry used by dataset collection commands.

The registry deliberately permits automatic acquisition only from sources whose
API returns enough *per-item* provenance to enforce the allowlist.  A public
URL is never treated as a training right.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "data" / "dataset_registry.json"


class DatasetPolicyError(ValueError):
    """Raised before a non-approved source or license can be acquired."""


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> Dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        raise DatasetPolicyError(f"Dataset registry is missing: {registry_path}")
    return json.loads(registry_path.read_text(encoding="utf-8"))


def source_config(source_id: str, registry_path: str | Path = DEFAULT_REGISTRY) -> Dict[str, Any]:
    for source in load_registry(registry_path).get("sources", []):
        if source.get("id") == source_id:
            return source
    raise DatasetPolicyError(f"Source '{source_id}' is not present in the approved dataset registry.")


def is_auto_collection_allowed(source_id: str, registry_path: str | Path = DEFAULT_REGISTRY) -> bool:
    return source_config(source_id, registry_path).get("collection_mode") == "api_allowlisted"


def _normalise_license(value: str | None) -> str:
    return "-".join((value or "").strip().lower().replace("_", "-").split())


def license_is_allowed(value: str | None, registry_path: str | Path = DEFAULT_REGISTRY) -> bool:
    normalised = _normalise_license(value)
    policy = load_registry(registry_path).get("policy", {})
    rejected = [_normalise_license(item) for item in policy.get("rejected_item_licenses", [])]
    allowed = [_normalise_license(item) for item in policy.get("allowed_item_licenses", [])]
    if not normalised or any(token in normalised for token in rejected):
        return False
    return any(token in normalised for token in allowed)


def url_is_prohibited(url: str, registry_path: str | Path = DEFAULT_REGISTRY) -> bool:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return any(host == banned or host.endswith("." + banned)
               for banned in load_registry(registry_path).get("policy", {}).get("prohibited_hosts", []))


def assert_candidate_allowed(candidate: Dict[str, Any], source_id: str) -> None:
    """Validate provenance before any network request is made for an asset."""
    if not is_auto_collection_allowed(source_id):
        raise DatasetPolicyError(
            f"{source_id} is manual-review-only. Add a reviewed local asset with an authorization record instead."
        )
    for field in ("source_url", "download_url"):
        url = candidate.get(field) or ""
        if not url or url_is_prohibited(url):
            raise DatasetPolicyError(f"Candidate has missing or prohibited {field}.")
    if not license_is_allowed(candidate.get("license")):
        raise DatasetPolicyError(f"Candidate license is not in the commercial/derivative-use allowlist: {candidate.get('license')!r}")
    required = ("source_url", "download_url", "license", "license_url", "creator")
    missing = [field for field in required if not candidate.get(field)]
    if missing:
        raise DatasetPolicyError(f"Candidate lacks required provenance fields: {', '.join(missing)}")
