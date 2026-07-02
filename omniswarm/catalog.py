"""Live OmniRoute model catalog: fetch, normalize, filter, cache.

Reads the gateway's OpenAI-compatible /models list (NO auth header) and shapes it
into the fields the picker + recommender need. auto/* and tllm/* are excluded
(banned from hot paths); non-chat models (image/embedding only) are dropped.
"""
import time
import httpx

CATALOG_TTL = 300  # seconds

_EXCLUDE_PREFIXES = ("auto/", "tllm/")


def _chat_capable(m: dict) -> bool:
    out = m.get("output_modalities")
    if isinstance(out, list) and out and "text" not in out:
        return False  # e.g. image/audio-only
    t = str(m.get("type", "")).lower()
    if t in {"embedding", "image", "audio", "video", "moderation", "rerank"}:
        return False
    return True


def normalize_models(raw: list[dict]) -> list[dict]:
    out = []
    for m in raw:
        mid = m.get("id")
        if not mid or mid.startswith(_EXCLUDE_PREFIXES):
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


async def fetch_catalog(client: httpx.AsyncClient, base_url: str) -> list[dict]:
    # NO auth header — empty key is required for the full catalog.
    resp = await client.get(f"{base_url}/models", headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return normalize_models(data)


async def get_cached_catalog(app) -> list[dict]:
    now = time.time()
    cached = getattr(app.state, "catalog", None)
    cached_at = getattr(app.state, "catalog_at", 0.0)
    if cached is not None and (now - cached_at) < CATALOG_TTL:
        return cached
    try:
        client = getattr(app.state, "client", None)
        fresh = await fetch_catalog(client, app.state.settings.omniroute_base_url)
        app.state.catalog = fresh
        app.state.catalog_at = now
        return fresh
    except Exception:
        return cached if cached is not None else []
