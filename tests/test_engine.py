import os
import tempfile
import pytest
from omniswarm import engine, council, store
from omniswarm.adapters import OmniRouteError
from omniswarm.config import Settings


@pytest.fixture()
def settings():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store.init_db(path)
    yield Settings("http://x/v1", path, 3, 60, 20)
    os.remove(path)


@pytest.mark.asyncio
async def test_process_job_happy_path(monkeypatch, settings):
    async def fake_generate(*a, **k):
        return "candidate"
    async def fake_review(*a, **k):
        return council.ReviewResult("pass", "high", "vetted answer",
                                    ["mistral/mistral-large-latest"])
    monkeypatch.setattr(engine, "generate", fake_generate)
    monkeypatch.setattr(engine, "review", fake_review)
    req = engine.JobRequest("general", "sys", "summarize this")
    out = await engine.process_job(None, settings, req)
    assert out["status"] == "done"
    assert out["text"] == "vetted answer"
    assert out["tokens_saved"] > 0
    assert store.get_job(settings.db_path, out["job_id"])["status"] == "done"


@pytest.mark.asyncio
async def test_process_job_marks_failed_on_omniroute_error(monkeypatch, settings):
    async def boom(*a, **k):
        raise OmniRouteError("LAN down")
    monkeypatch.setattr(engine, "generate", boom)
    req = engine.JobRequest("general", "sys", "x")
    out = await engine.process_job(None, settings, req)
    assert out["status"] == "failed"
    assert store.get_job(settings.db_path, out["job_id"])["status"] == "failed"


import json as _json


@pytest.mark.asyncio
async def test_process_job_persists_input_and_models(monkeypatch, settings):
    async def fake_generate(*a, **k):
        return "candidate"

    async def fake_review(*a, **k):
        return council.ReviewResult("pass", "high", "vetted answer", ["m1", "m2"])

    monkeypatch.setattr(engine, "generate", fake_generate)
    monkeypatch.setattr(engine, "review", fake_review)
    req = engine.JobRequest("general", "sys", "my input text")
    out = await engine.process_job(None, settings, req)

    row = store.get_job(settings.db_path, out["job_id"])
    assert row["input"] == "my input text"
    assert _json.loads(row["models_used"]) == ["m1", "m2"]


@pytest.mark.asyncio
async def test_process_job_persists_provenance(monkeypatch, settings):
    async def fake_generate(*a, **k):
        return "candidate"
    async def fake_review(*a, **k):
        rr = council.ReviewResult("pass", "high", "vetted", ["m1", "m2"])
        rr.steps = [{"stage": "draft", "model": "m1"}, {"stage": "judge", "model": "m2", "score": 0.9}]
        return rr
    monkeypatch.setattr(engine, "generate", fake_generate)
    monkeypatch.setattr(engine, "review", fake_review)
    out = await engine.process_job(None, settings, engine.JobRequest("general", "s", "u"))
    import json as _j
    row = store.get_job(settings.db_path, out["job_id"])
    steps = _j.loads(row["provenance"])
    assert [s["stage"] for s in steps] == ["draft", "judge"]


@pytest.mark.asyncio
async def test_store_mode_none_does_not_persist_text(monkeypatch, settings):
    async def fake_generate(*a, **k):
        return "candidate"
    async def fake_review(*a, **k):
        rr = council.ReviewResult("pass", "high", "secret answer", ["m1"])
        rr.steps = [{"stage": "draft", "model": "m1", "detail": "secret draft"}]
        return rr
    monkeypatch.setattr(engine, "generate", fake_generate)
    monkeypatch.setattr(engine, "review", fake_review)
    out = await engine.process_job(None, settings, engine.JobRequest("general", "s", "secret input"), store_mode="none")
    row = store.get_job(settings.db_path, out["job_id"])
    assert row["input"] == ""
    assert row["result"] == ""
    assert row["provenance"] == "[]"
    assert row["tokens_saved"] > 0  # metadata still computed
    assert row["verdict"] == "pass"


@pytest.mark.asyncio
async def test_store_mode_redact_masks_text(monkeypatch, settings):
    async def fake_generate(*a, **k):
        return "candidate"
    async def fake_review(*a, **k):
        return council.ReviewResult("pass", "high", "the answer", ["m1"])
    monkeypatch.setattr(engine, "generate", fake_generate)
    monkeypatch.setattr(engine, "review", fake_review)
    out = await engine.process_job(None, settings, engine.JobRequest("general", "s", "my input"), store_mode="redact")
    row = store.get_job(settings.db_path, out["job_id"])
    assert "redacted" in row["input"]
    assert "redacted" in row["result"]
    assert row["provenance"] == "[]"


@pytest.mark.asyncio
async def test_process_job_uses_supplied_job_id(monkeypatch, settings):
    async def fake_generate(*a, **k):
        return "candidate"
    async def fake_review(*a, **k):
        return council.ReviewResult("pass", "high", "ok", ["m"])
    monkeypatch.setattr(engine, "generate", fake_generate)
    monkeypatch.setattr(engine, "review", fake_review)
    out = await engine.process_job(None, settings, engine.JobRequest("general", "s", "u"), job_id="fixed123")
    assert out["job_id"] == "fixed123"
    assert store.get_job(settings.db_path, "fixed123") is not None


@pytest.mark.asyncio
async def test_process_job_publishes_events(monkeypatch, settings):
    from omniswarm import events as ev
    seen = []
    monkeypatch.setattr(ev, "publish", lambda e: seen.append(e))
    async def fake_generate(*a, **k):
        return "candidate"
    async def fake_review(*a, **k):
        return council.ReviewResult("pass", "high", "ok", ["m"])
    monkeypatch.setattr(engine, "generate", fake_generate)
    monkeypatch.setattr(engine, "review", fake_review)
    out = await engine.process_job(None, settings, engine.JobRequest("general", "s", "u"))
    types = [e.get("type") for e in seen]
    assert "job" in types  # at least a start and/or completion job event
    starts = [e for e in seen if e.get("type") == "job" and e.get("status") == "running"]
    assert starts and starts[0]["job_id"] == out["job_id"]
