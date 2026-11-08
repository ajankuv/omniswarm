"""Live OmniRoute model catalog: fetch, normalize, filter, cache.

Reads the gateway's OpenAI-compatible /models list (NO auth header) and shapes it
into the fields the picker + recommender need. auto/* and tllm/* are excluded
(banned from hot paths); non-chat models (image/embedding only) are dropped; with
OMNISWARM_ALLOWED_PROVIDERS set, models from other providers are dropped too.

This module is the one place that decides which model ids are routable at all
(`is_routable`) and which providers are allowed (`provider_allowed`).
"""
import time
import httpx

CATALOG_TTL = 300  # seconds

# never routed to: auto/* picks a model behind our back, tllm/* is banned from hot paths
EXCLUDE_PREFIXES = ("auto/", "tllm/")


def is_routable(mid: object) -> bool:
    """True only for non-empty model ids outside EXCLUDE_PREFIXES."""
    return isinstance(mid, str) and bool(mid) and not mid.startswith(EXCLUDE_PREFIXES)


def provider_allowed(mid: str, allowed: frozenset[str]) -> bool:
    """True if `mid`'s provider prefix is in `allowed`; an empty set allows all."""
    return not allowed or mid.split("/", 1)[0] in allowed


def filter_allowed(models: list[dict], allowed: frozenset[str]) -> list[dict]:
    return [m for m in models if provider_allowed(m["id"], allowed)]


def _chat_capable(m: dict) -> bool:
    out = m.get("output_modalities")
    if isinstance(out, list) and out and "text" not in out:
        return False  # e.g. image/audio-only
    t = str(m.get("type", "")).lower()
    if t in {"embedding", "image", "audio", "video", "moderation", "rerank"}:
        return False
    return True


def normalize_models(raw: list[dict], allowed: frozenset[str] = frozenset()) -> list[dict]:
    out = []
    for m in raw:
        mid = m.get("id")
        if not is_routable(mid) or not provider_allowed(mid, allowed):
            continue
        if not _chat_capable(m):
            continue
        caps = m.get("capabilities") or {}
        out.append({
            "id": mid,
            "provider": mid.split("/", 1)[0],
            "name": m.get("name") or mid,
            "capabilities": {
                "tool_calling": bool(caps.get("tool_calling", False)),
                "reasoning": bool(caps.get("reasoning", False)),
                "thinking": bool(caps.get("thinking", False)),
            },
            "context_length": int(m.get("context_length") or 0),
            "max_output_tokens": int(m.get("max_output_tokens") or 0),
            "chat_capable": True,
        })
    out.sort(key=lambda x: x["id"])
    return out


async def fetch_catalog(client: httpx.AsyncClient, base_url: str,
                        allowed: frozenset[str] = frozenset()) -> list[dict]:
    # NO auth header — empty key is required for the full catalog.
    resp = await client.get(f"{base_url}/models", headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return normalize_models(data, allowed)


async def get_cached_catalog(app) -> list[dict]:
    now = time.time()
    cached = getattr(app.state, "catalog", None)
    cached_at = getattr(app.state, "catalog_at", 0.0)
    if cached is not None and (now - cached_at) < CATALOG_TTL:
        return cached
    try:
        client = getattr(app.state, "client", None)
        fresh = await fetch_catalog(client, app.state.settings.omniroute_base_url,
                                    getattr(app.state.settings, "allowed_providers", frozenset()))
        app.state.catalog = fresh
        app.state.catalog_at = now
        return fresh
    except Exception:
        return cached if cached is not None else []
