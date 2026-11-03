"""In-memory state store using Hash Maps for O(1) job lookup and idempotency caching."""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
import uuid

from core.exceptions import JobNotFoundError


class JobStatus(str, Enum):
    """Lifecycle states of an asynchronous TTS synthesis job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobState:
    """Mutable state record for a synthesis job."""

    job_id: str
    text: str
    voice_id: str
    request_hash: str
    webhook_url: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    chunks: List[str] = field(default_factory=list)
    resolved_voices: List[str] = field(default_factory=list)
    annotations: List[dict] = field(default_factory=list)
    retries: int = 0
    error_message: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(frozen=True)
class DLQEntry:
    """Dead Letter Queue record for jobs that exhausted all retry attempts."""

    job_id: str
    voice_id: str
    text: str
    attempts: int
    reason: str
    failed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class JobStore:
    """Thread/coroutine-safe in-memory state store backed by Python Hash Maps (`dict`).

    Data Structures:
    1. `_jobs` (`dict[str, JobState]`): Hash Map providing O(1) average-time lookup
       and mutation of job status by `job_id`.
    2. `_idempotency_index` (`dict[str, str]`): Hash Map mapping `sha256(voice_id + text)`
       to an active `job_id` for O(1) duplicate request detection.
    3. `_dlq` (`list[DLQEntry]`): Dead Letter Queue storing permanently failed jobs, plus
       `_dlq_job_ids` (`set[str]`) to guarantee O(1) deduplication in the DLQ.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._idempotency_index: Dict[str, str] = {}
        self._dlq: List[DLQEntry] = []
        self._dlq_job_ids: set[str] = set()
        self._lock = asyncio.Lock()

    @staticmethod
    def compute_request_hash(text: str, voice_id: str) -> str:
        """Compute deterministic SHA-256 hash for `(voice_id, normalized_text)`."""
        normalized = f"{voice_id.strip()}::{text.strip()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def get_or_create_job(
        self,
        text: str,
        voice_id: str,
        webhook_url: Optional[str] = None,
    ) -> tuple[JobState, bool]:
        """Return `(job_state, created_new)` using the idempotency Hash Map.

        Idempotency Rule:
        If a job with the exact same `(text, voice_id)` hash is currently in `PENDING`
        or `PROCESSING` state, return that existing job instead of spawning duplicate work.
        """
        req_hash = self.compute_request_hash(text=text, voice_id=voice_id)

        async with self._lock:
            existing_job_id = self._idempotency_index.get(req_hash)
            if existing_job_id is not None:
                existing_job = self._jobs.get(existing_job_id)
                if existing_job and existing_job.status in (
                    JobStatus.PENDING,
                    JobStatus.PROCESSING,
                ):
                    return existing_job, False

            job_id = str(uuid.uuid4())
            job = JobState(
                job_id=job_id,
                text=text,
                voice_id=voice_id,
                request_hash=req_hash,
                webhook_url=webhook_url,
                status=JobStatus.PENDING,
            )
            self._jobs[job_id] = job
            self._idempotency_index[req_hash] = job_id
            return job, True

    def get_job(self, job_id: str) -> JobState:
        """Retrieve a job by ID in O(1) time or raise `JobNotFoundError`."""
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Job '{job_id}' not found.")
        return job

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        chunks: Optional[List[str]] = None,
        resolved_voices: Optional[List[str]] = None,
        annotations: Optional[List[dict]] = None,
        error_message: Optional[str] = None,
    ) -> JobState:
        """Update job state fields and release the idempotency lock when terminal."""
        async with self._lock:
            job = self.get_job(job_id)
            job.status = status
            job.updated_at = datetime.now(timezone.utc).isoformat()

            if chunks is not None:
                job.chunks = chunks
            if resolved_voices is not None:
                job.resolved_voices = resolved_voices
            if annotations is not None:
                job.annotations = annotations
            if error_message is not None:
                job.error_message = error_message

            # Once a job finishes (COMPLETED or FAILED), remove it from the active
            # idempotency cache so future requests with the same payload can run again.
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                if self._idempotency_index.get(job.request_hash) == job_id:
                    del self._idempotency_index[job.request_hash]

            return job

    async def increment_retry(self, job_id: str, error_message: str) -> int:
        """Increment retry counter for a job and return the new count."""
        async with self._lock:
            job = self.get_job(job_id)
            job.retries += 1
            job.error_message = error_message
            job.updated_at = datetime.now(timezone.utc).isoformat()
            return job.retries

    async def push_to_dlq(self, job: JobState, reason: str) -> None:
        """Record a permanently failed job in the Dead Letter Queue (idempotent per job_id)."""
        async with self._lock:
            if job.job_id in self._dlq_job_ids:
                return
            self._dlq_job_ids.add(job.job_id)
            self._dlq.append(
                DLQEntry(
                    job_id=job.job_id,
                    voice_id=job.voice_id,
                    text=job.text,
                    attempts=job.retries,
                    reason=reason,
                )
            )

    @property
    def dlq(self) -> List[DLQEntry]:
        """Return a snapshot copy of the Dead Letter Queue."""
        return list(self._dlq)

