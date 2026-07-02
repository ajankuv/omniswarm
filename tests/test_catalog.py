import time
import pytest
from omniswarm import catalog

RAW = [
    {"id": "auto/best-coding", "capabilities": {"reasoning": True}, "context_length": 1000},
    {"id": "tllm/foo", "capabilities": {}},
    {"id": "nvidia/meta/llama-4-maverick-17b-128e-instruct", "name": "Llama 4 Maverick",
     "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
     "context_length": 128000, "max_output_tokens": 8192,
     "output_modalities": ["text"]},
    {"id": "mistral/devstral-latest", "capabilities": {"tool_calling": True},
     "context_length": 32000, "output_modalities": ["text"]},
    {"id": "some/image-gen", "capabilities": {}, "output_modalities": ["image"]},
]


def test_normalize_filters_and_shapes():
    out = catalog.normalize_models(RAW)
    ids = [m["id"] for m in out]
    assert "auto/best-coding" not in ids   # auto/* dropped
    assert "tllm/foo" not in ids            # tllm/* dropped
    assert "some/image-gen" not in ids      # non-chat dropped
    assert "nvidia/meta/llama-4-maverick-17b-128e-instruct" in ids
    m = next(x for x in out if x["id"].startswith("nvidia/"))
    assert m["provider"] == "nvidia"
    assert m["capabilities"] == {"tool_calling": True, "reasoning": False, "thinking": False}
    assert m["context_length"] == 128000
    assert m["chat_capable"] is True


def test_normalize_missing_fields_default_safely():
    out = catalog.normalize_models([{"id": "mistral/mistral-medium-3-5"}])
    assert len(out) == 1
    m = out[0]
    assert m["capabilities"] == {"tool_calling": False, "reasoning": False, "thinking": False}
    assert m["context_length"] == 0 and m["max_output_tokens"] == 0
    assert m["provider"] == "mistral"


@pytest.mark.asyncio
async def test_get_cached_catalog_uses_cache(monkeypatch):
    calls = {"n": 0}

    async def fake_fetch(client, base_url):
        calls["n"] += 1
        return [{"id": "nvidia/x", "provider": "nvidia", "name": "x",
                 "capabilities": {"tool_calling": False, "reasoning": False, "thinking": False},
                 "context_length": 0, "max_output_tokens": 0, "chat_capable": True}]
    monkeypatch.setattr(catalog, "fetch_catalog", fake_fetch)

    class App:  # stand-in for FastAPI app with .state
        class state:
            settings = type("S", (), {"omniroute_base_url": "http://x/v1"})()
    app = App()
    c1 = await catalog.get_cached_catalog(app)
    c2 = await catalog.get_cached_catalog(app)
    assert calls["n"] == 1           # second call served from cache
    assert c1 == c2 and c1[0]["id"] == "nvidia/x"


@pytest.mark.asyncio
async def test_get_cached_catalog_returns_stale_on_fetch_error(monkeypatch):
    state = {"first": True}

    async def flaky_fetch(client, base_url):
        if state["first"]:
            state["first"] = False
            return [{"id": "nvidia/x", "provider": "nvidia", "name": "x",
                     "capabilities": {"tool_calling": False, "reasoning": False, "thinking": False},
                     "context_length": 0, "max_output_tokens": 0, "chat_capable": True}]
        raise RuntimeError("gateway down")
    monkeypatch.setattr(catalog, "fetch_catalog", flaky_fetch)
    monkeypatch.setattr(catalog, "CATALOG_TTL", -1)  # force refetch every call

    class App:
        class state:
            settings = type("S", (), {"omniroute_base_url": "http://x/v1"})()
    app = App()
    first = await catalog.get_cached_catalog(app)
    second = await catalog.get_cached_catalog(app)  # fetch raises -> serve stale
    assert first == second and second[0]["id"] == "nvidia/x"
