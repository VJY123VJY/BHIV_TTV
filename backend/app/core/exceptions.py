"""
Exception hierarchy for the unified Text-to-Video system.
"""

class TTVException(Exception):
    """Base exception for all TTV system errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class ValidationError(TTVException):
    """Raised when request payload or parameters fail validation."""
    pass

class GovernanceViolationError(TTVException):
    """Raised when governance lock or authorization token checks fail."""
    pass

class ProviderError(TTVException):
    """Raised when an external or local AI provider fails."""
    pass

class PromptEngineeringError(TTVException):
    """Raised when prompt analysis or breakdown fails."""
    pass

class VisualGenerationError(TTVException):
    """Raised when keyframe or scene video rendering fails."""
    pass

class AudioGenerationError(TTVException):
    """Raised when speech synthesis or audio mixing fails."""
    pass

class FFmpegProcessingError(TTVException):
    """Raised when FFmpeg assembly, transcoding, or concatenation fails."""
    pass

class StorageError(TTVException):
    """Raised when artifact storage or retrieval fails."""
    pass

class JobNotFoundError(TTVException):
    """Raised when a requested generation job does not exist."""
    pass
