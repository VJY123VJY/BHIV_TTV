import hashlib
import uuid
from datetime import datetime

def compute_hash(data: str) -> str:
    """Compute deterministic SHA-256 hash for arbitrary string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def generate_execution_id(prefix: str = "exec") -> str:
    """Generate a unique, trace-safe execution ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rand_suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{rand_suffix}"
