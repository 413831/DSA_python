"""Integration tests for idempotency, worker pipeline, webhooks, and Dead Letter Queue (DLQ)."""

from typing import List
from fastapi.testclient import TestClient
import httpx
import pytest

from core.state import JobStatus
from main import create_app


@pytest.mark.asyncio
async def test_idempotency_returns_same_job_id_while_pending_or_processing() -> None:
    # Set a longer synthesis latency so the first job stays in-flight during the 2nd request
    app = create_app(synthesis_latency_sec=0.15, window_size=5)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "text": "Idempotency check on <emphasis>ElevenLabs</emphasis> synthesis.",
                "voice_id": "cloned_narrator_v1",
            }

            r1 = await client.post("/synthesize", json=payload)
            r2 = await client.post("/synthesize", json=payload)

            assert r1.status_code == 202
            assert r2.status_code == 202

            data1 = r1.json()
            data2 = r2.json()
            assert data1["job_id"] == data2["job_id"]
            assert data1["idempotent_reused"] is False
            assert data2["idempotent_reused"] is True

            # Wait for the event bus queue to finish processing
            await app.state.event_bus.join()

            job_resp = await client.get(f"/jobs/{data1['job_id']}")
            assert job_resp.status_code == 200
            job_data = job_resp.json()
            assert job_data["status"] == JobStatus.COMPLETED.value
            assert len(job_data["chunks"]) >= 1
            assert job_data["resolved_voices"] == [
                "eleven_multilingual_v2",
                "rachel_base",
                "cloned_narrator_v1",
            ]

            # Once COMPLETED, a new identical request should create a fresh job_id
            r3 = await client.post("/synthesize", json=payload)
            assert r3.status_code == 202
            assert r3.json()["job_id"] != data1["job_id"]
            await app.state.event_bus.join()


@pytest.mark.asyncio
async def test_failed_job_retries_and_moves_to_dlq_with_webhook() -> None:
    webhook_calls: List[dict] = []

    async def mock_webhook_handler(request: httpx.Request) -> httpx.Response:
        import json

        webhook_calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"ok": True})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_webhook_handler))
    app = create_app(
        synthesis_latency_sec=0.01,
        max_retries=2,
        http_client=mock_client,
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/synthesize",
                json={
                    "text": "Testing retry and DLQ behavior.",
                    "voice_id": "unstable_experimental_voice",
                    "webhook_url": "https://client.example.com/webhook",
                },
            )
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            # Wait for initial attempt + 2 retries + tts.failed event handling
            await app.state.event_bus.join()

            status_resp = await client.get(f"/jobs/{job_id}")
            assert status_resp.status_code == 200
            job_data = status_resp.json()
            assert job_data["status"] == JobStatus.FAILED.value
            # Initial attempt + 2 retries = 3 total failed attempts (> max_retries=2)
            assert job_data["retries"] == 3

            dlq_resp = await client.get("/dlq")
            assert dlq_resp.status_code == 200
            dlq_entries = dlq_resp.json()
            assert len(dlq_entries) == 1
            assert dlq_entries[0]["job_id"] == job_id
            assert dlq_entries[0]["attempts"] == 3

            assert len(webhook_calls) == 1
            assert webhook_calls[0]["event"] == "tts.failed"
            assert webhook_calls[0]["job_id"] == job_id

    await mock_client.aclose()


def test_http_error_responses_for_invalid_inputs() -> None:
    app = create_app(synthesis_latency_sec=0.01)
    with TestClient(app) as client:
        # 404 on unknown voice
        r_voice = client.post(
            "/synthesize",
            json={"text": "Hello world", "voice_id": "missing_voice"},
        )
        assert r_voice.status_code == 404

        # 400 on malformed SSML
        r_ssml = client.post(
            "/synthesize",
            json={"text": "Hello <emphasis>unclosed", "voice_id": "rachel_base"},
        )
        assert r_ssml.status_code == 400

        # 404 on unknown job_id
        r_job = client.get("/jobs/non-existent-id")
        assert r_job.status_code == 404

