"""Domain event definitions for the event-driven TTS architecture."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar, List, Optional


@dataclass(frozen=True)
class BaseEvent:
    """Base domain event with timestamp and event_type discriminator."""

    event_type: ClassVar[str] = "base.event"
    job_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        kw_only=True,
    )


@dataclass(frozen=True)
class TTSRequested(BaseEvent):
    """Published when a client requests asynchronous speech synthesis (`tts.requested`)."""

    event_type: ClassVar[str] = "tts.requested"
    text: str
    voice_id: str
    webhook_url: Optional[str] = None
    attempt: int = 1


@dataclass(frozen=True)
class TTSCompleted(BaseEvent):
    """Published when a worker finishes synthesizing all chunks (`tts.completed`)."""

    event_type: ClassVar[str] = "tts.completed"
    voice_id: str
    chunks: List[str]
    resolved_voices: List[str]
    webhook_url: Optional[str] = None


@dataclass(frozen=True)
class TTSFailed(BaseEvent):
    """Published when a job permanently fails after exhausting retries (`tts.failed`)."""

    event_type: ClassVar[str] = "tts.failed"
    voice_id: str
    error: str
    attempts: int
    webhook_url: Optional[str] = None

