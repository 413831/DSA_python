"""Custom domain exceptions for the TTS service."""


class TTSError(Exception):
    """Base exception for TTS service errors."""


class SSMLParseError(TTSError):
    """Raised when the input text contains invalid or malformed SSML-like tags."""


class VoiceNotFoundError(TTSError):
    """Raised when a requested voice_id or one of its dependencies does not exist."""


class CyclicVoiceDependencyError(TTSError):
    """Raised when a cycle is detected in the voice dependency graph."""


class JobNotFoundError(TTSError):
    """Raised when a job_id cannot be found in the state store."""

