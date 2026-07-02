import asyncio
import pytest
from omniswarm import events


@pytest.mark.asyncio
async def test_publish_reaches_subscriber():
    q = events.subscribe()
    try:
        events.publish({"type": "job", "job_id": "x"})
        ev = await asyncio.wait_for(q.get(), timeout=1)
        assert ev["job_id"] == "x"
    finally:
        events.unsubscribe(q)


def test_unsubscribe_removes():
    q = events.subscribe()
    events.unsubscribe(q)
    events.publish({"a": 1})  # must not raise with no/removed subscribers
