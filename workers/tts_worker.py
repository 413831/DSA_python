"""Asynchronous background worker that processes TTS jobs and dispatches webhooks."""

import asyncio
from dataclasses import asdict
import logging
from typing import Optional

import httpx

from core.chunker import chunk_text_sliding_window
from core.exceptions import TTSError
from core.parser import parse_ssml_text
from core.state import JobStatus, JobStore
from core.voice_graph import VoiceDependencyGraph
from events.bus import InMemoryEventBus
from events.models import TTSCompleted, TTSFailed, TTSRequested

logger = logging.getLogger(__name__)


class TTSWorker:
    """Consumes `tts.requested` events, executes the TTS pipeline, and emits completion/failure events.

    Pipeline stages per job:
    1. Resolve voice dependency chain via Graph Traversal (DFS with cycle detection).
    2. Parse SSML tags (`<emphasis>`, `<break time="..."/>`) via String Parsing.
    3. Split cleaned text into streaming segments via Sliding Window (`chunk_text_sliding_window`).
    4. Simulate acoustic model inference latency (`asyncio.sleep`).
    5. Handle retries and Dead Letter Queue (DLQ) if failures exceed `max_retries` (default: 2 retries).
    6. Emit `tts.completed` or `tts.failed` and trigger optional HTTP webhook via `httpx`.
    """

    def __init__(
        self,
        bus: InMemoryEventBus,
        store: JobStore,
        voice_graph: VoiceDependencyGraph,
        *,
        window_size: int = 8,
        overlap: int = 0,
        synthesis_latency_sec: float = 0.05,
        max_retries: int = 2,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.bus = bus
        self.store = store
        self.voice_graph = voice_graph
        self.window_size = window_size
        self.overlap = overlap
        self.synthesis_latency_sec = synthesis_latency_sec
        self.max_retries = max_retries
        self._external_client = http_client
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

        # Register event handlers on the bus
        self.bus.subscribe(TTSRequested.event_type, self.handle_tts_requested)
        self.bus.subscribe(TTSCompleted.event_type, self.handle_tts_completed)
        self.bus.subscribe(TTSFailed.event_type, self.handle_tts_failed)

    async def start(self) -> None:
        """Start the background event loop consumer task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="tts-worker-loop")
        logger.info("TTSWorker started background loop.")

    async def stop(self) -> None:
        """Gracefully stop the background consumer loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("TTSWorker stopped.")

    async def _run_loop(self) -> None:
        """Continuously dispatch events from the `InMemoryEventBus` queue."""
        while self._running:
            try:
                await self.bus.dispatch_next()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Unexpected error in worker dispatch loop: %s", exc)

    async def handle_tts_requested(self, event: TTSRequested) -> None:
        """Process a `tts.requested` event with retry and Dead Letter Queue (DLQ) support."""
        job_id = event.job_id
        voice_id = event.voice_id

        logger.info(
            "Processing job | job_id=%s voice_id=%s attempt=%d",
            job_id,
            voice_id,
            event.attempt,
        )

        try:
            await self.store.update_status(job_id, JobStatus.PROCESSING)

            # 1. Graph Traversal (DFS): Resolve voice dependency chain
            resolved_voices = self.voice_graph.resolve_dependency_chain_dfs(voice_id)

            # 2. String Parsing: Validate & extract SSML style tags
            parsed = parse_ssml_text(event.text)
            if not parsed.clean_text:
                raise TTSError("Text is empty after stripping SSML tags.")

            # Simulate acoustic synthesis failure for unstable test voice or explicit fail marker
            if voice_id == "unstable_experimental_voice" or "[FAIL_SYNTHESIS]" in parsed.clean_text:
                raise RuntimeError(
                    f"Simulated acoustic model failure for voice_id='{voice_id}'"
                )

            # 3. Sliding Window: Split text into streaming chunks respecting punctuation
            chunks = chunk_text_sliding_window(
                parsed.clean_text,
                window_size=self.window_size,
                overlap=self.overlap,
            )

            # 4. Simulate GPU inference latency per chunk
            await asyncio.sleep(self.synthesis_latency_sec * max(1, len(chunks)))

            # 5. Update state store to COMPLETED
            annotations_dict = [asdict(a) for a in parsed.annotations]
            await self.store.update_status(
                job_id,
                JobStatus.COMPLETED,
                chunks=chunks,
                resolved_voices=resolved_voices,
                annotations=annotations_dict,
            )

            # 6. Publish `tts.completed` event
            await self.bus.publish(
                TTSCompleted(
                    job_id=job_id,
                    voice_id=voice_id,
                    chunks=chunks,
                    resolved_voices=resolved_voices,
                    webhook_url=event.webhook_url,
                )
            )

        except Exception as exc:
            retries = await self.store.increment_retry(job_id, str(exc))
            logger.warning(
                "Job processing error | job_id=%s voice_id=%s retries=%d error=%s",
                job_id,
                voice_id,
                retries,
                exc,
            )

            # DLQ Rule: If a job fails more than `max_retries` (2) times (i.e. > 2 failures),
            # mark as FAILED, push to DLQ, and emit `tts.failed`.
            if retries > self.max_retries:
                job = await self.store.update_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message=str(exc),
                )
                await self.store.push_to_dlq(job, reason=str(exc))
                logger.error(
                    "Job moved to DLQ | job_id=%s voice_id=%s attempts=%d reason=%s",
                    job_id,
                    voice_id,
                    retries,
                    exc,
                )
                await self.bus.publish(
                    TTSFailed(
                        job_id=job_id,
                        voice_id=voice_id,
                        error=str(exc),
                        attempts=retries,
                        webhook_url=event.webhook_url,
                    )
                )
            else:
                # Re-enqueue event for retry
                await self.bus.publish(
                    TTSRequested(
                        job_id=job_id,
                        text=event.text,
                        voice_id=voice_id,
                        webhook_url=event.webhook_url,
                        attempt=retries + 1,
                    )
                )

    async def handle_tts_completed(self, event: TTSCompleted) -> None:
        """Handle `tts.completed` by logging and firing the optional webhook."""
        logger.info(
            "Job completed | job_id=%s voice_id=%s chunks=%d",
            event.job_id,
            event.voice_id,
            len(event.chunks),
        )
        if event.webhook_url:
            await self._send_webhook(
                event.webhook_url,
                {
                    "event": event.event_type,
                    "job_id": event.job_id,
                    "status": JobStatus.COMPLETED.value,
                    "voice_id": event.voice_id,
                    "chunks": event.chunks,
                    "resolved_voices": event.resolved_voices,
                },
            )

    async def handle_tts_failed(self, event: TTSFailed) -> None:
        """Handle `tts.failed` by logging and firing the optional webhook."""
        logger.error(
            "Job permanently failed | job_id=%s voice_id=%s attempts=%d error=%s",
            event.job_id,
            event.voice_id,
            event.attempts,
            event.error,
        )
        if event.webhook_url:
            await self._send_webhook(
                event.webhook_url,
                {
                    "event": event.event_type,
                    "job_id": event.job_id,
                    "status": JobStatus.FAILED.value,
                    "voice_id": event.voice_id,
                    "error": event.error,
                    "attempts": event.attempts,
                },
            )

    async def _send_webhook(self, url: str, payload: dict) -> None:
        """Send an asynchronous HTTP POST webhook notification using `httpx`."""
        try:
            if self._external_client is not None:
                resp = await self._external_client.post(url, json=payload, timeout=5.0)
                resp.raise_for_status()
            else:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, timeout=5.0)
                    resp.raise_for_status()
            logger.info(
                "Webhook delivered | job_id=%s url=%s status_code=%d",
                payload.get("job_id"),
                url,
                resp.status_code,
            )
        except Exception as exc:
            # Webhook delivery failure should not crash the worker or corrupt job state
            logger.warning(
                "Webhook delivery failed | job_id=%s url=%s error=%s",
                payload.get("job_id"),
                url,
                exc,
            )

