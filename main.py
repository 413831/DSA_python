"""FastAPI application entrypoint and lifespan manager for the Async TTS API."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import httpx

from api.routes import router
from core.exceptions import TTSError
from core.state import JobStore
from core.voice_graph import VoiceDependencyGraph
from events.bus import InMemoryEventBus
from workers.tts_worker import TTSWorker

# Configure structured-style standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(
    *,
    synthesis_latency_sec: float = 0.05,
    window_size: int = 8,
    overlap: int = 0,
    max_retries: int = 2,
    http_client: Optional[httpx.AsyncClient] = None,
) -> FastAPI:
    """Application factory creating the FastAPI app with its event bus, state store, and worker."""
    job_store = JobStore()
    event_bus = InMemoryEventBus()
    voice_graph = VoiceDependencyGraph()
    worker = TTSWorker(
        bus=event_bus,
        store=job_store,
        voice_graph=voice_graph,
        window_size=window_size,
        overlap=overlap,
        synthesis_latency_sec=synthesis_latency_sec,
        max_retries=max_retries,
        http_client=http_client,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await worker.start()
        try:
            yield
        finally:
            await worker.stop()

    app = FastAPI(
        title="ElevenLabs Async TTS Interview Simulation API",
        version="1.0.0",
        description=(
            "Event-driven asynchronous Text-to-Speech backend featuring SSML parsing, "
            "sliding-window chunking, DFS/BFS voice dependency resolution, idempotency, and DLQ."
        ),
        lifespan=lifespan,
    )

    # Attach core singletons to `app.state` for clean Dependency Injection
    app.state.job_store = job_store
    app.state.event_bus = event_bus
    app.state.voice_graph = voice_graph
    app.state.worker = worker

    @app.exception_handler(TTSError)
    async def tts_domain_exception_handler(
        request: Request, exc: TTSError
    ) -> JSONResponse:
        logger.error("Unhandled domain error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    app.include_router(router)
    return app


app = create_app()

