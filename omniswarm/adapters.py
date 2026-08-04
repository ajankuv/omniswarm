import asyncio
import json
import random
import time

import httpx

_REASONING_PREFIXES = ("ghm/", "cerebras/")

_SINK = None  # optional callable(model:str, ok:bool, latency_ms:float, status:str)


def set_sink(fn) -> None:
    global _SINK
    _SINK = fn


def _emit(model: str, ok: bool, latency_ms: float, status: str) -> None:
    if _SINK is not None:
        try:
            _SINK(model, ok, latency_ms, status)
        except Exception:
            pass


class OmniRouteError(Exception):
    pass


# HTTP codes where retrying is pointless: the provider is out of quota, the key is
# rejected, or the model is gone. Retrying just wastes time and hammers a dead
# provider (this is what turned a quota-burn into a 54%-failure cascade). Everything
# else (5xx, timeouts, empty responses) is treated as transient and retried.
_PERMANENT_HTTP = frozenset({401, 402, 403, 404, 405, 410, 429})


def _is_permanent(status: str) -> bool:
    """True for a status string like 'HTTP 403' whose code is non-retryable."""
    if status.startswith("HTTP "):
        code = status[5:].split()[0] if status[5:] else ""
        try:
            return int(code) in _PERMANENT_HTTP
        except ValueError:
            return False
    return False


# Explicit provider-exhaustion signals the gateway puts in the error body — these are
# DEFINITIVE (the provider is out of credits / rate-limited), unlike a bare status code.
_EXHAUSTION_MARKERS = (
    "no active credentials",       # OmniRoute: provider has no working key/credits
    "insufficient_quota",
    "rate_limit_exceeded",
    "rate_limit_error",
    "quota exceeded",
    "out of credits",
)


def _is_exhaustion_body(text: str) -> bool:
    """True if the gateway's error body explicitly says the provider is exhausted."""
    if not text:
        return False
    low = text.lower()
    return any(mk in low for mk in _EXHAUSTION_MARKERS)


def is_exhaustion(status: str) -> bool:
    """The breaker's fast-trip signal: an explicit provider-exhaustion response."""
    return "EXHAUSTED" in status


def _is_reasoning(model: str) -> bool:
    return model.startswith(_REASONING_PREFIXES)


def _build_body(model: str, system: str, user: str, max_tokens: int) -> dict:
    use_stream = model.startswith("ghm/")  # ghm requires stream=true
    if _is_reasoning(model):
        max_tokens = max(max_tokens, 200)  # reasoning models need budget to think
    return {
        "model": model,
        "stream": use_stream,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def _content_from_message(msg: dict) -> str:
    content = (msg.get("content") or "").strip()
    if content:
        return content
    return (msg.get("reasoning") or "").strip()  # reasoning models footgun


def _parse_stream(text: str) -> str:
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []  # some chunks (e.g. usage) carry no choices
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        piece = delta.get("content") or delta.get("reasoning") or ""
        parts.append(piece)
    return "".join(parts).strip()


async def generate(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 512,
    max_retries: int = 2,
) -> str:
    url = f"{base_url}/chat/completions"
    body = _build_body(model, system, user, max_tokens)
    headers = {"Content-Type": "application/json"}  # NO auth header (footgun)
    last_err = "unknown"
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        out = ""
        ok = False
        status = "ok"
        try:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                status = f"HTTP {resp.status_code}"
                # the gateway tells us WHY in a structured error body — surface an
                # explicit provider-exhaustion signal so the breaker can trip on it
                # immediately (one "no active credentials" is definitive) instead of
                # counting blind HTTP codes.
                if _is_exhaustion_body(resp.text):
                    status += " EXHAUSTED"
                last_err = status
            else:
                if body["stream"]:
                    out = _parse_stream(resp.text)
                else:
                    msg = resp.json()["choices"][0]["message"]
                    out = _content_from_message(msg)
                if out:
                    ok = True
                else:
                    status = "empty"; last_err = "empty content"
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            status = type(e).__name__; last_err = f"{type(e).__name__}: {e}"
        _emit(model, ok, (time.perf_counter() - t0) * 1000.0, status)
        if ok:
            return out
        # fail fast on quota/auth/gone — retrying can't help and only burns a dead provider
        if _is_permanent(status):
            raise OmniRouteError(f"{model} failed ({status}) — not retrying (permanent/quota)")
        if attempt < max_retries:
            await asyncio.sleep((2 ** attempt) * 0.5 + random.uniform(0, 0.3))
    raise OmniRouteError(f"{model} failed after {max_retries + 1} attempts: {last_err}")


async def health(client: httpx.AsyncClient, base_url: str) -> bool:
    try:
        resp = await client.get(f"{base_url}/models")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False
