import httpx
import pytest
import respx
from omniswarm.adapters import generate, health, OmniRouteError

BASE = "http://omni/v1"


def _chat_json(content):
    return {"choices": [{"message": {"content": content}}]}


@respx.mock
@pytest.mark.asyncio
async def test_generate_non_stream_returns_content():
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_json("hello world"))
    )
    async with httpx.AsyncClient() as client:
        out = await generate(client, BASE, "mistral/mistral-large-latest", "sys", "hi")
    assert out == "hello world"


@respx.mock
@pytest.mark.asyncio
async def test_stream_skips_chunks_with_empty_choices():
    # Real DeepSeek-R1 SSE includes chunks with an empty choices list (e.g. usage).
    # These must be skipped, not crash with IndexError.
    sse_body = (
        'data: {"choices": [], "usage": {"total_tokens": 5}}\n'
        "\n"
        'data: {"choices": [{"delta": {"content": "real answer"}}]}\n'
        "\n"
        "data: [DONE]\n"
        "\n"
    )
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, text=sse_body))
    async with httpx.AsyncClient() as client:
        out = await generate(client, BASE, "ghm/deepseek/DeepSeek-R1", "s", "u")
    assert out == "real answer"


@respx.mock
@pytest.mark.asyncio
async def test_reasoning_falls_back_to_reasoning_field():
    sse_body = (
        'data: {"choices": [{"delta": {"content": "", "reasoning": "the answer"}}]}\n'
        "\n"
        "data: [DONE]\n"
        "\n"
    )
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, text=sse_body))
    async with httpx.AsyncClient() as client:
        out = await generate(client, BASE, "ghm/deepseek/DeepSeek-R1", "s", "u")
    assert out == "the answer"


@respx.mock
@pytest.mark.asyncio
async def test_retries_then_raises_on_persistent_error():
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient() as client:
        with pytest.raises(OmniRouteError):
            await generate(client, BASE, "mistral/mistral-large-latest", "s", "u", max_retries=1)


@respx.mock
@pytest.mark.asyncio
async def test_health_true_on_200():
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    async with httpx.AsyncClient() as client:
        assert await health(client, BASE) is True


@respx.mock
@pytest.mark.asyncio
async def test_generate_emits_telemetry():
    from omniswarm import adapters as ad
    calls = []
    ad.set_sink(lambda model, ok, lat, status: calls.append((model, ok, status)))
    try:
        respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json=_chat_json("hi")))
        async with httpx.AsyncClient() as client:
            await generate(client, BASE, "mistral/mistral-large-latest", "s", "u")
        assert calls and calls[-1][0] == "mistral/mistral-large-latest"
        assert calls[-1][1] is True and calls[-1][2] == "ok"
    finally:
        ad.set_sink(None)
