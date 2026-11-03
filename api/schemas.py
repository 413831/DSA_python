"""Pydantic v2 request and response schemas for the TTS API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.state import JobStatus


class SynthesizeRequest(BaseModel):
    """Payload for `POST /synthesize`."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Input text with optional simplified SSML tags (<emphasis>, <break time='500ms'/>).",
        examples=[
            "Hello world! <break time='500ms'/> Welcome to <emphasis level='strong'>ElevenLabs</emphasis> streaming TTS."
        ],
    )
    voice_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the target voice or cloned voice.",
        examples=["cloned_narrator_v1"],
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Optional callback URL invoked when the job completes or fails.",
    )


class SynthesizeResponse(BaseModel):
    """Response returned by `POST /synthesize` (HTTP 202 Accepted)."""

    job_id: str
    status: JobStatus
    idempotent_reused: bool = Field(
        default=False,
        description="True if an existing in-flight job_id was returned via idempotency cache.",
    )


class JobResponse(BaseModel):
    """Detailed job status returned by `GET /jobs/{job_id}`."""

    job_id: str
    status: JobStatus
    voice_id: str
    chunks: List[str] = Field(default_factory=list)
    resolved_voices: List[str] = Field(default_factory=list)
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    retries: int = 0
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class DLQEntryResponse(BaseModel):
    """Schema for inspecting Dead Letter Queue entries via `GET /dlq`."""

    job_id: str
    voice_id: str
    text: str
    attempts: int
    reason: str
    failed_at: str

