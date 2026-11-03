"""FastAPI route definitions for the asynchronous TTS synthesis service."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas import (
    DLQEntryResponse,
    JobResponse,
    SynthesizeRequest,
    SynthesizeResponse,
)
from core.exceptions import (
    CyclicVoiceDependencyError,
    JobNotFoundError,
    SSMLParseError,
    VoiceNotFoundError,
)
from core.parser import parse_ssml_text
from core.state import JobStore
from core.voice_graph import VoiceDependencyGraph
from events.bus import InMemoryEventBus
from events.models import TTSRequested

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_components(
    request: Request,
) -> tuple[JobStore, InMemoryEventBus, VoiceDependencyGraph]:
    """Extract initialized domain components from FastAPI `app.state`."""
    return (
        request.app.state.job_store,
        request.app.state.event_bus,
        request.app.state.voice_graph,
    )


@router.post(
    "/synthesize",
    response_model=SynthesizeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue an asynchronous TTS synthesis job",
)
async def synthesize_speech(
    payload: SynthesizeRequest,
    request: Request,
) -> SynthesizeResponse:
    """Validate input, check idempotency cache, and enqueue a `tts.requested` event."""
    store, bus, voice_graph = _get_components(request)

    # Fast-fail validation (400 / 404) before queuing background work:
    # 1. Verify voice exists and dependency graph is cycle-free
    try:
        voice_graph.resolve_dependency_chain_dfs(payload.voice_id)
    except VoiceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except CyclicVoiceDependencyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    # 2. Verify SSML tags are syntactically valid and text is non-empty
    try:
        parsed = parse_ssml_text(payload.text)
        if not parsed.clean_text:
            raise SSMLParseError("Input text contains no speakable words.")
    except SSMLParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    # 3. Idempotency check (O(1) Hash Map lookup for active jobs with same text + voice_id)
    job, created_new = await store.get_or_create_job(
        text=payload.text,
        voice_id=payload.voice_id,
        webhook_url=payload.webhook_url,
    )

    if not created_new:
        logger.info(
            "Idempotent hit: returning existing in-flight job | job_id=%s voice_id=%s",
            job.job_id,
            job.voice_id,
        )
        return SynthesizeResponse(
            job_id=job.job_id,
            status=job.status,
            idempotent_reused=True,
        )

    # 4. Publish `tts.requested` to the in-memory event bus
    await bus.publish(
        TTSRequested(
            job_id=job.job_id,
            text=payload.text,
            voice_id=payload.voice_id,
            webhook_url=payload.webhook_url,
        )
    )

    return SynthesizeResponse(
        job_id=job.job_id,
        status=job.status,
        idempotent_reused=False,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve the status and synthesized chunks of a job",
)
async def get_job_status(job_id: str, request: Request) -> JobResponse:
    """Lookup job status in O(1) time from the `JobStore` Hash Map."""
    store, _, _ = _get_components(request)
    try:
        job = store.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        voice_id=job.voice_id,
        chunks=job.chunks,
        resolved_voices=job.resolved_voices,
        annotations=job.annotations,
        retries=job.retries,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get(
    "/dlq",
    response_model=List[DLQEntryResponse],
    status_code=status.HTTP_200_OK,
    summary="Inspect the Dead Letter Queue (failed jobs after max retries)",
)
async def get_dlq_entries(request: Request) -> List[DLQEntryResponse]:
    """Return all permanently failed jobs recorded in the Dead Letter Queue."""
    store, _, _ = _get_components(request)
    return [
        DLQEntryResponse(
            job_id=entry.job_id,
            voice_id=entry.voice_id,
            text=entry.text,
            attempts=entry.attempts,
            reason=entry.reason,
            failed_at=entry.failed_at,
        )
        for entry in store.dlq
    ]

