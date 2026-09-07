import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for production observability."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "execution_id"):
            log_data["execution_id"] = record.execution_id
        if hasattr(record, "event_type"):
            log_data["event_type"] = record.event_type
        if hasattr(record, "payload"):
            log_data["payload"] = record.payload
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def setup_logger(name: str = "ttv", level: str = "INFO") -> logging.Logger:
    """Setup and configure unified logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    return logger

logger = setup_logger()

class TelemetryEmitter:
    """Telemetry system capturing formal execution lifecycle events."""
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger

    def emit(self, event_type: str, execution_id: str, data: Optional[Dict[str, Any]] = None, level: str = "info"):
        payload = data or {}
        extra = {
            "execution_id": execution_id,
            "event_type": event_type,
            "payload": payload
        }
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn(f"[{event_type}] execution={execution_id}", extra=extra)

telemetry = TelemetryEmitter()
