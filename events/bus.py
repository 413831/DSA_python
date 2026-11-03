"""In-memory asynchronous Event Bus combining `asyncio.Queue` and pub/sub handlers."""

import asyncio
from collections import defaultdict
import logging
from typing import Any, Awaitable, Callable, Dict, List

from events.models import BaseEvent

logger = logging.getLogger(__name__)

EventHandler = Callable[[ Any], Awaitable[None]]


class InMemoryEventBus:
    """Asynchronous Event Bus built on `asyncio.Queue` + topic subscriber registry.

    Architecture Pattern:
    - Producers call `await bus.publish(event)` which enqueues the event in O(1) time
      and returns immediately (decoupling HTTP latency from worker processing).
    - Subscribers register via `bus.subscribe("tts.requested", handler)`.
    - The internal dispatcher loop pulls events from `asyncio.Queue` and invokes
      all subscribed async handlers for that `event_type`.
    """

    def __init__(self, max_queue_size: int = 1000) -> None:
        self._queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=max_queue_size)
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe an async handler coroutine to a specific event type."""
        self._subscribers[event_type].append(handler)
        logger.debug("Subscribed handler %s to event '%s'", handler.__name__, event_type)

    async def publish(self, event: BaseEvent) -> None:
        """Enqueue an event onto the internal `asyncio.Queue`."""
        await self._queue.put(event)
        logger.info(
            "Event published | event_type=%s job_id=%s queue_size=%d",
            event.event_type,
            event.job_id,
            self._queue.qsize(),
        )

    async def dispatch_next(self) -> None:
        """Consume a single event from the queue and invoke all registered handlers."""
        event = await self._queue.get()
        try:
            handlers = self._subscribers.get(event.event_type, [])
            for handler in handlers:
                await handler(event)
        finally:
            self._queue.task_done()

    async def join(self) -> None:
        """Block until all currently queued events have been processed (useful in tests)."""
        await self._queue.join()

    @property
    def pending_events(self) -> int:
        """Return current number of unprocessed events in the queue."""
        return self._queue.qsize()

