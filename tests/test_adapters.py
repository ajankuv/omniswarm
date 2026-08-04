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


@respx.mock
@pytest.mark.asyncio
async def test_permanent_error_fails_fast_no_retry():
    """403/429/etc are quota/auth/gone — retrying only burns a dead provider.
    Must raise after exactly ONE call, not max_retries+1."""
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(403))
    async with httpx.AsyncClient() as client:
        with pytest.raises(OmniRouteError):
            await generate(client, BASE, "gemini/x", "s", "u", max_retries=3)
    assert route.call_count == 1          # failed fast, did NOT retry 4x


@respx.mock
@pytest.mark.asyncio
async def test_transient_error_still_retries():
    """5xx/timeout stay retryable — must attempt max_retries+1 times."""
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(503))
    async with httpx.AsyncClient() as client:
        with pytest.raises(OmniRouteError):
            await generate(client, BASE, "prov/x", "s", "u", max_retries=2)
    assert route.call_count == 3          # 1 + 2 retries


def test_is_permanent_classification():
    from omniswarm.adapters import _is_permanent
    for c in (401, 402, 403, 404, 410, 429):
        assert _is_permanent(f"HTTP {c}") is True
    for c in (500, 502, 503, 504):
        assert _is_permanent(f"HTTP {c}") is False
    assert _is_permanent("Timeout") is False and _is_permanent("empty") is False


@respx.mock
@pytest.mark.asyncio
async def test_incident_replay_failfast_feeds_provider_breaker():
    """End-to-end: a provider stuck on 429 (quota) must (a) fail fast — 1 call per
    request, not 3 — and (b) drive its provider breaker to 'exhausted'. This is the
    Aug-4 quota-burn scenario the whole Tier-1 change targets."""
    from omniswarm import adapters as ad, failover
    breaker = failover.ProviderBreaker(threshold=3, cooldown=600)
    ad.set_sink(lambda model, ok, lat, status: breaker.record(model, ok, status))
    try:
        route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(429))
        async with httpx.AsyncClient() as client:
            for _ in range(3):                       # 3 failed requests
                with pytest.raises(OmniRouteError):
                    await generate(client, BASE, "gemini/flash", "s", "u", max_retries=2)
        assert route.call_count == 3                 # 1 call per request (fail-fast), NOT 9
        assert "gemini" in breaker.exhausted()       # provider tripped from the 3 quota errors
    finally:
        ad.set_sink(None)


def test_exhaustion_body_detection():
    from omniswarm.adapters import _is_exhaustion_body, is_exhaustion
    assert _is_exhaustion_body('{"error":{"message":"No active credentials for provider: mistral"}}')
    assert _is_exhaustion_body('{"error":{"code":"rate_limit_exceeded"}}')
    assert _is_exhaustion_body('{"error":{"type":"rate_limit_error"}}')
    assert not _is_exhaustion_body('{"error":{"message":"model xyz not found"}}')
    assert not _is_exhaustion_body("")
    assert is_exhaustion("HTTP 404 EXHAUSTED") and not is_exhaustion("HTTP 404")


@respx.mock
@pytest.mark.asyncio
async def test_explicit_exhaustion_status_surfaced():
    """A 'no active credentials' body must surface an EXHAUSTED status via the sink."""
    from omniswarm import adapters as ad
    seen = []
    ad.set_sink(lambda m, ok, lat, status: seen.append(status))
    try:
        respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(
            404, json={"error": {"message": "No active credentials for provider: mistral",
                                 "code": "model_not_found"}}))
        async with httpx.AsyncClient() as client:
            with pytest.raises(OmniRouteError):
                await generate(client, BASE, "mistral/codestral-latest", "s", "u", max_retries=2)
        assert seen and "EXHAUSTED" in seen[-1]     # explicit signal surfaced
    finally:
        ad.set_sink(None)
