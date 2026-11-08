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


# --- v1.3.1: "can't run here" error bodies are permanent ------------------------------

@pytest.mark.parametrize("code,message", [
    (502, "Auggie CLI not found"),
    (503, "Cloudflare Playground needs Playwright and its Chromium browser"),
    (500, "DEVIN_AGENTIC_HOME must be an absolute path"),
    (503, "Codex app-server transport is not configured"),
    (503, "The FOO-TOOL binary is NOT INSTALLED on this host"),
    (500, "This provider requires installing an extra package"),
])
@respx.mock
@pytest.mark.asyncio
async def test_unavailable_body_is_permanent_not_retried(code, message):
    from omniswarm import adapters as ad
    seen = []
    ad.set_sink(lambda m, ok, lat, status: seen.append(status))
    try:
        route = respx.post(f"{BASE}/chat/completions").mock(
            return_value=httpx.Response(code, json={"error": {"message": message}}))
        async with httpx.AsyncClient() as client:
            with pytest.raises(OmniRouteError):
                await generate(client, BASE, "prov/x", "s", "u", max_retries=3)
        assert route.call_count == 1                      # no retries: it can't ever work
        assert seen == [f"HTTP {code} UNAVAILABLE"]
        assert ad._is_permanent(seen[0]) and ad.is_unavailable(seen[0])
    finally:
        ad.set_sink(None)


@respx.mock
@pytest.mark.asyncio
async def test_generic_503_still_transient():
    from omniswarm import adapters as ad
    seen = []
    ad.set_sink(lambda m, ok, lat, status: seen.append(status))
    try:
        route = respx.post(f"{BASE}/chat/completions").mock(
            return_value=httpx.Response(503, json={"error": {"message": "upstream overloaded, try again"}}))
        async with httpx.AsyncClient() as client:
            with pytest.raises(OmniRouteError):
                await generate(client, BASE, "prov/x", "s", "u", max_retries=2)
        assert route.call_count == 3                      # still retried
        assert seen == ["HTTP 503"] * 3
        assert not ad._is_permanent("HTTP 503") and not ad.is_unavailable("HTTP 503")
        # "model not found" is a 404 about ONE model, not a can't-run provider
        assert not ad._is_unavailable_body('{"error":{"message":"model xyz not found"}}')
    finally:
        ad.set_sink(None)


@respx.mock
@pytest.mark.asyncio
async def test_probe_is_one_tiny_call():
    from omniswarm import adapters as ad
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(503))
    async with httpx.AsyncClient() as client:
        ok, err = await ad.probe(client, BASE, "prov/x")
        assert ok is False and "503" in err
        assert route.call_count == 1                      # max_retries=0
        assert __import__("json").loads(route.calls[0].request.content)["max_tokens"] == 5
        route.mock(return_value=httpx.Response(200, json=_chat_json("pong")))
        assert await ad.probe(client, BASE, "prov/x") == (True, "")


@respx.mock
@pytest.mark.asyncio
async def test_probe_counts_empty_200_as_reachable():
    # Reasoning models spend a tiny max_tokens budget on hidden reasoning and come back
    # 200 with empty content (finish_reason=length). They answer fine with a real
    # budget, so a probe must not call them dead. Normal calls still treat empty as a
    # failure and retry.
    from omniswarm import adapters as ad
    seen = []
    ad.set_sink(lambda m, ok, lat, status: seen.append((ok, status)))
    try:
        route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}))
        async with httpx.AsyncClient() as client:
            assert await ad.probe(client, BASE, "nvidia/some/reasoning-model") == (True, "")
            assert route.call_count == 1
            assert seen == [(True, "probe:ok")]     # sink learns it is reachable (as a probe)
            with pytest.raises(OmniRouteError):    # a real job still needs content
                await generate(client, BASE, "nvidia/some/reasoning-model", "s", "u", max_retries=1)
            assert route.call_count == 3
    finally:
        ad.set_sink(None)


# --- gate review fixes -----------------------------------------------------------------

async def _status_for(code, body, max_retries=0):
    """Run one generate() against a mocked reply; return the statuses the sink saw."""
    from omniswarm import adapters as ad
    seen = []
    ad.set_sink(lambda m, ok, lat, status: seen.append(status))
    try:
        reply = (httpx.Response(code, json=body) if isinstance(body, dict)
                 else httpx.Response(code, text=body))
        with respx.mock:
            respx.post(f"{BASE}/chat/completions").mock(return_value=reply)
            async with httpx.AsyncClient() as client:
                with pytest.raises(OmniRouteError):
                    await generate(client, BASE, "prov/x", "s", "u", max_retries=max_retries)
    finally:
        ad.set_sink(None)
    return seen


@pytest.mark.asyncio
async def test_unavailable_needs_5xx_and_error_message():
    from omniswarm import adapters as ad
    from omniswarm import failover
    # a 4xx that echoes the user's prompt is about the REQUEST, never "can't run"
    echo = {"error": {"message": "Invalid request: 'why is pytest not installed on my box?'"}}
    assert await _status_for(400, echo) == ["HTTP 400"]
    s404 = await _status_for(404, echo)
    assert s404 == ["HTTP 404"] and ad._is_permanent(s404[0]) and not ad.is_unavailable(s404[0])
    b = failover.ProviderBreaker(threshold=4, cooldown=600)
    b.record("prov/x", ok=False, status="HTTP 400")
    b.record("prov/x", ok=False, status=s404[0])
    assert "prov" not in b.exhausted()                    # nothing benched
    # still UNAVAILABLE: a 5xx whose error MESSAGE says it can't run
    assert await _status_for(503, {"error": {"message": "Cloudflare Playground needs Playwright"}}) \
        == ["HTTP 503 UNAVAILABLE"]
    # ...or a non-JSON 5xx body saying so
    assert await _status_for(502, "Auggie CLI not found") == ["HTTP 502 UNAVAILABLE"]
    # a marker only in a non-message field doesn't count
    other_field = {"error": {"message": "internal error"}, "detail": "pytest not installed"}
    assert await _status_for(500, other_field) == ["HTTP 500"]
    # bare "not configured" is too broad (only "transport is not configured" counts)
    assert await _status_for(503, {"error": {"message": "logging not configured, retry"}}) == ["HTTP 503"]
    # 504 (gateway timeout) is never a can't-run signal
    assert await _status_for(504, {"error": {"message": "Auggie CLI not found"}}) == ["HTTP 504"]


@respx.mock
@pytest.mark.asyncio
async def test_generate_per_call_timeout_and_probe_uses_short_one():
    from omniswarm import adapters as ad
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json=_chat_json("hi")))
    async with httpx.AsyncClient(timeout=60) as client:
        await generate(client, BASE, "prov/x", "s", "u", timeout=7)
        assert route.calls[-1].request.extensions["timeout"]["read"] == 7
        await generate(client, BASE, "prov/x", "s", "u")
        assert route.calls[-1].request.extensions["timeout"]["read"] == 60   # client default
        await ad.probe(client, BASE, "prov/x")
        assert route.calls[-1].request.extensions["timeout"]["read"] == ad.PROBE_TIMEOUT
    assert ad.PROBE_TIMEOUT <= 20


@pytest.mark.parametrize("reply,reachable", [
    ({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}, False),
    ({"choices": [{"message": {"content": ""}}]}, False),
    ({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}, True),
    ({"choices": [{"message": {"content": "", "reasoning_content": ""}}]}, True),
    ({"choices": [{"message": {"content": "", "reasoning": None}}]}, True),
    ({"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 12}}, True),
    ({"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 0}}, False),
], ids=["stop", "bare", "length", "reasoning_content", "reasoning", "usage", "zero-usage"])
@respx.mock
@pytest.mark.asyncio
async def test_probe_empty_200_needs_evidence_the_model_ran(reply, reachable):
    from omniswarm import adapters as ad
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json=reply))
    async with httpx.AsyncClient() as client:
        ok, _ = await ad.probe(client, BASE, "prov/x")
    assert ok is reachable


@respx.mock
@pytest.mark.asyncio
async def test_probe_empty_stream_needs_evidence():
    from omniswarm import adapters as ad
    ran = ('data: {"choices": [{"delta": {"content": ""}, "finish_reason": "length"}]}\n\n'
           "data: [DONE]\n\n")
    silent = 'data: {"choices": [{"delta": {"content": ""}}]}\n\ndata: [DONE]\n\n'
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, text=ran))
    async with httpx.AsyncClient() as client:
        assert (await ad.probe(client, BASE, "ghm/x"))[0] is True
        route.mock(return_value=httpx.Response(200, text=silent))
        assert (await ad.probe(client, BASE, "ghm/x"))[0] is False


@respx.mock
@pytest.mark.asyncio
async def test_probe_calls_are_tagged_for_the_sink():
    from omniswarm import adapters as ad
    seen = []
    ad.set_sink(lambda m, ok, lat, status: seen.append((ok, status)))
    try:
        route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(
            503, json={"error": {"message": "Codex app-server transport is not configured"}}))
        async with httpx.AsyncClient() as client:
            await ad.probe(client, BASE, "cxa/x")
            route.mock(return_value=httpx.Response(200, json=_chat_json("pong")))
            await ad.probe(client, BASE, "groq/x")
        assert seen == [(False, "probe:HTTP 503 UNAVAILABLE"), (True, "probe:ok")]
        assert ad._is_permanent(seen[0][1]) and ad.is_unavailable(seen[0][1])
        assert ad.status_code("probe:HTTP 404") == 404
    finally:
        ad.set_sink(None)
