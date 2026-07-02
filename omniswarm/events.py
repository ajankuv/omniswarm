"""In-process pub/sub for live job events (consumed by the SSE /stream endpoint)."""
import asyncio

_subs: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subs.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subs.discard(q)


def publish(event: dict) -> None:
    for q in list(_subs):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass
