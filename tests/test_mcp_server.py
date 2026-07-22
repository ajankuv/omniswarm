import asyncio
import pytest
from omniswarm import mcp_server


@pytest.mark.asyncio
async def test_delegate_calls_engine_and_returns_result(monkeypatch):
    captured = {}

    async def fake_process(client, settings, req):
        captured["task_type"] = req.task_type
        captured["user"] = req.user
        return {
            "job_id": "j", "text": "vetted answer", "verdict": "pass",
            "confidence": "high", "models_used": ["m"], "tokens_saved": 7, "status": "done",
        }

    monkeypatch.setattr(mcp_server.store, "init_db", lambda p: None)
    monkeypatch.setattr(mcp_server, "_get_client", lambda: None)
    monkeypatch.setattr(mcp_server.engine, "process_job", fake_process)

    out = await mcp_server.omniswarm_delegate("summarize this", "summarize")
    assert out["text"] == "vetted answer"
    assert out["verdict"] == "pass"
    assert captured == {"task_type": "summarize", "user": "summarize this"}


@pytest.mark.asyncio
async def test_stats_tool_returns_savings(monkeypatch):
    monkeypatch.setattr(mcp_server.store, "init_db", lambda p: None)
    monkeypatch.setattr(
        mcp_server.store, "stats",
        lambda p, rate=0.0: {"total_jobs": 3, "tokens_saved": 99, "by_status": {}, "by_verdict": {}},
    )
    out = await mcp_server.omniswarm_stats()
    assert out["tokens_saved"] == 99
    assert out["total_jobs"] == 3


def test_module_surface_exists():
    assert callable(mcp_server.main)
    assert mcp_server._mcp is not None
    assert callable(mcp_server.omniswarm_delegate)
    assert callable(mcp_server.omniswarm_stats)


@pytest.mark.asyncio
async def test_submit_returns_job_id_and_runs_in_background(monkeypatch):
    seen = {}

    async def fake_process(client, settings, req, **kwargs):
        seen["task_type"] = req.task_type
        seen["user"] = req.user
        seen["job_id"] = kwargs.get("job_id")
        return {"status": "done"}

    monkeypatch.setattr(mcp_server.store, "init_db", lambda p: None)
    monkeypatch.setattr(mcp_server, "_get_client", lambda: None)
    monkeypatch.setattr(mcp_server.engine, "process_job", fake_process)

    out = await mcp_server.omniswarm_submit("summarize the log", "summarize")
    assert out["status"] == "running"
    assert out["job_id"]
    # let the background task finish, then confirm it ran with our job_id
    await asyncio.gather(*list(mcp_server._bg_tasks))
    assert seen["job_id"] == out["job_id"]
    assert seen["task_type"] == "summarize"
    assert seen["user"] == "summarize the log"


@pytest.mark.asyncio
async def test_result_reads_store(monkeypatch):
    monkeypatch.setattr(
        mcp_server.store, "get_job",
        lambda path, jid: {"id": jid, "status": "done", "verdict": "pass",
                           "confidence": "high", "result": "answer",
                           "models_used": ["m"], "note": ""} if jid == "j1" else None,
    )
    ok = await mcp_server.omniswarm_result("j1")
    assert ok["status"] == "done" and ok["verdict"] == "pass" and ok["result"] == "answer"
    missing = await mcp_server.omniswarm_result("nope")
    assert missing["status"] == "not_found"


@pytest.mark.asyncio
async def test_list_jobs_returns_compact_rows(monkeypatch):
    monkeypatch.setattr(
        mcp_server.store, "list_jobs",
        lambda path, limit, status, verdict, task_type, q: [
            {"id": "j1", "task_type": "code", "status": "done", "verdict": "pass",
             "confidence": "high", "created_at": 1.0, "result": "SHOULD_BE_DROPPED"},
        ],
    )
    rows = await mcp_server.omniswarm_list_jobs(limit=5)
    assert rows[0]["id"] == "j1" and rows[0]["verdict"] == "pass"
    assert "result" not in rows[0]  # compact: heavy fields omitted


def test_startup_applies_runtime_overrides(monkeypatch):
    from omniswarm import registry
    monkeypatch.setattr(mcp_server.store, "init_db", lambda p: None)
    monkeypatch.setattr(mcp_server.runtime, "load_runtime",
                        lambda: {"judge": "test/mcp-judge", "models": {}, "synth": "", "members": []})
    try:
        mcp_server._startup()
        assert registry.JUDGE_MODEL == "test/mcp-judge"
    finally:
        registry.apply_runtime({})  # restore defaults for other tests


@pytest.mark.asyncio
async def test_submit_records_unexpected_failure(monkeypatch):
    calls = {}
    async def boom(client, settings, req, **kwargs):
        raise ValueError("kaboom")
    def fake_update(path, job_id, **fields):
        calls["status"] = fields.get("status"); calls["job_id"] = job_id
    monkeypatch.setattr(mcp_server.store, "init_db", lambda p: None)
    monkeypatch.setattr(mcp_server, "_get_client", lambda: None)
    monkeypatch.setattr(mcp_server.engine, "process_job", boom)
    monkeypatch.setattr(mcp_server.store, "update_job", fake_update)
    out = await mcp_server.omniswarm_submit("x", "general")
    await asyncio.gather(*list(mcp_server._bg_tasks))
    assert calls["status"] == "failed" and calls["job_id"] == out["job_id"]


@pytest.mark.asyncio
async def test_recommend_returns_pick(monkeypatch):
    async def fake_fetch(client, base_url):
        return [{"id": "mistral/devstral-latest", "provider": "mistral", "name": "Devstral",
                 "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 256000, "max_output_tokens": 8192, "chat_capable": True}]
    monkeypatch.setattr(mcp_server, "_get_client", lambda: None)
    monkeypatch.setattr(mcp_server.catalog, "fetch_catalog", fake_fetch)
    monkeypatch.setattr(mcp_server.store, "model_reliability", lambda p: {})
    out = await mcp_server.omniswarm_recommend("code")
    assert out["recommended"] == "mistral/devstral-latest"
    assert "ranked" in out and "why" in out


@pytest.mark.asyncio
async def test_recommend_graceful_when_catalog_down(monkeypatch):
    async def boom(client, base_url):
        raise RuntimeError("gateway down")
    monkeypatch.setattr(mcp_server, "_get_client", lambda: None)
    monkeypatch.setattr(mcp_server.catalog, "fetch_catalog", boom)
    monkeypatch.setattr(mcp_server.store, "model_reliability", lambda p: {})
    out = await mcp_server.omniswarm_recommend("code")
    assert out["recommended"] is None
    assert out["ranked"] == []


@pytest.mark.asyncio
async def test_list_task_types(monkeypatch):
    out = await mcp_server.omniswarm_list_task_types()
    names = [t["name"] for t in out]
    assert "general" in names and "reasoning" in names
    reasoning = next(t for t in out if t["name"] == "reasoning")
    assert reasoning["high_stakes"] is True
    assert isinstance(reasoning["model"], str) and reasoning["model"]


@pytest.mark.asyncio
async def test_remote_submit_proxies_to_app(monkeypatch):
    calls = {}
    async def fake_post(path, body):
        calls["path"] = path; calls["body"] = body
        return {"job_id": "R1", "status": "running", "stream": "/stream"}
    monkeypatch.setenv("OMNISWARM_REMOTE", "http://host:8100")
    monkeypatch.setattr(mcp_server, "_remote_post", fake_post)
    out = await mcp_server.omniswarm_submit("do it", "code")
    assert out == {"job_id": "R1", "status": "running"}
    assert calls["path"] == "/v1/jobs"
    assert calls["body"] == {"messages": [{"role": "user", "content": "do it"}],
                             "omniswarm": {"task_type": "code"}}


@pytest.mark.asyncio
async def test_remote_result_maps_and_404(monkeypatch):
    monkeypatch.setenv("OMNISWARM_REMOTE", "http://host:8100")
    async def fake_get(path):
        if path == "/jobs/known":
            return 200, {"status": "done", "verdict": "pass", "confidence": "high",
                         "result": "x", "models_used": ["m"], "note": ""}
        return 404, {}
    monkeypatch.setattr(mcp_server, "_remote_get", fake_get)
    ok = await mcp_server.omniswarm_result("known")
    assert ok["verdict"] == "pass" and ok["result"] == "x"
    missing = await mcp_server.omniswarm_result("nope")
    assert missing["status"] == "not_found"


@pytest.mark.asyncio
async def test_remote_list_jobs_proxies(monkeypatch):
    monkeypatch.setenv("OMNISWARM_REMOTE", "http://host:8100")
    async def fake_get(path):
        assert path.startswith("/jobs?limit=5")
        assert "status" not in path
        return 200, [{"id": "R1", "task_type": "code", "status": "done", "verdict": "pass",
                      "confidence": "high", "created_at": 1.0, "result": "DROP"}]
    monkeypatch.setattr(mcp_server, "_remote_get", fake_get)
    rows = await mcp_server.omniswarm_list_jobs(limit=5)
    assert rows[0]["id"] == "R1" and "result" not in rows[0]


@pytest.mark.asyncio
async def test_remote_unreachable_returns_clear_error(monkeypatch):
    monkeypatch.setenv("OMNISWARM_REMOTE", "http://host:8100")
    async def boom_post(path, body):
        raise mcp_server._RemoteError("connection refused")
    async def boom_get(path):
        raise mcp_server._RemoteError("connection refused")
    monkeypatch.setattr(mcp_server, "_remote_post", boom_post)
    monkeypatch.setattr(mcp_server, "_remote_get", boom_get)
    s = await mcp_server.omniswarm_submit("x", "general")
    assert s["status"] == "error" and "connection refused" in s["error"]
    r = await mcp_server.omniswarm_result("x")
    assert r["status"] == "error"
    lj = await mcp_server.omniswarm_list_jobs()
    assert lj == []


def test_remote_headers_env(monkeypatch):
    monkeypatch.delenv("OMNISWARM_REMOTE_TOKEN", raising=False)
    assert mcp_server._remote_headers() == {}
    monkeypatch.setenv("OMNISWARM_REMOTE_TOKEN", "secret")
    assert mcp_server._remote_headers() == {"Authorization": "Bearer secret"}


@pytest.mark.asyncio
async def test_local_mode_unchanged_when_remote_unset(monkeypatch):
    # OMNISWARM_REMOTE unset -> engine-direct path (Feature 1 behavior)
    monkeypatch.delenv("OMNISWARM_REMOTE", raising=False)
    seen = {}
    async def fake_process(client, settings, req, **kwargs):
        seen["job_id"] = kwargs.get("job_id"); return {"status": "done"}
    monkeypatch.setattr(mcp_server, "_get_client", lambda: None)
    monkeypatch.setattr(mcp_server.engine, "process_job", fake_process)
    out = await mcp_server.omniswarm_submit("x", "general")
    assert out["status"] == "running" and out["job_id"]
    await asyncio.gather(*list(mcp_server._bg_tasks))
    assert seen["job_id"] == out["job_id"]


@pytest.mark.asyncio
async def test_remote_delegate_submits_and_polls(monkeypatch):
    monkeypatch.setenv("OMNISWARM_REMOTE", "http://host:8100")
    monkeypatch.setenv("OMNISWARM_REMOTE_POLL_TIMEOUT", "10")
    polls = {"n": 0}
    async def fake_post(path, body):
        assert path == "/v1/jobs"
        return {"job_id": "D1", "status": "running"}
    async def fake_get(path):
        polls["n"] += 1
        if polls["n"] < 2:
            return 200, {"status": "running"}
        return 200, {"status": "done", "verdict": "pass", "confidence": "high",
                     "result": "vetted", "models_used": ["m"], "note": ""}
    async def fake_sleep(s):
        pass
    monkeypatch.setattr(mcp_server, "_remote_post", fake_post)
    monkeypatch.setattr(mcp_server, "_remote_get", fake_get)
    monkeypatch.setattr(mcp_server.asyncio, "sleep", fake_sleep)
    out = await mcp_server.omniswarm_delegate("do it", "code")
    assert out["verdict"] == "pass" and out["result"] == "vetted"
    assert polls["n"] >= 2


@pytest.mark.asyncio
async def test_remote_delegate_timeout_returns_running(monkeypatch):
    monkeypatch.setenv("OMNISWARM_REMOTE", "http://host:8100")
    monkeypatch.setenv("OMNISWARM_REMOTE_POLL_TIMEOUT", "0")  # immediate timeout
    async def fake_post(path, body):
        return {"job_id": "D2", "status": "running"}
    async def fake_get(path):
        return 200, {"status": "running"}
    async def fake_sleep(s):
        pass
    monkeypatch.setattr(mcp_server, "_remote_post", fake_post)
    monkeypatch.setattr(mcp_server, "_remote_get", fake_get)
    monkeypatch.setattr(mcp_server.asyncio, "sleep", fake_sleep)
    out = await mcp_server.omniswarm_delegate("do it", "general")
    assert out["status"] == "running" and out["job_id"] == "D2"


@pytest.mark.asyncio
async def test_delegate_local_unchanged_when_remote_unset(monkeypatch):
    monkeypatch.delenv("OMNISWARM_REMOTE", raising=False)
    async def fake_process(client, settings, req):
        return {"text": "vetted answer", "verdict": "pass", "status": "done"}
    monkeypatch.setattr(mcp_server.store, "init_db", lambda p: None)
    monkeypatch.setattr(mcp_server, "_get_client", lambda: None)
    monkeypatch.setattr(mcp_server.engine, "process_job", fake_process)
    out = await mcp_server.omniswarm_delegate("x", "summarize")
    assert out["text"] == "vetted answer"


@pytest.mark.asyncio
async def test_remote_delegate_missing_job_id_returns_error(monkeypatch):
    monkeypatch.setenv("OMNISWARM_REMOTE", "http://host:8100")
    async def fake_post(path, body):
        return {}  # malformed: no job_id
    monkeypatch.setattr(mcp_server, "_remote_post", fake_post)
    out = await mcp_server.omniswarm_delegate("x", "general")
    assert out["status"] == "error" and "job_id" in out["error"]


@pytest.mark.asyncio
async def test_feedback_tool_records(monkeypatch):
    seen = {}
    monkeypatch.setattr(mcp_server.store, "init_db", lambda p: None)
    def fake_set(path, jid, val):
        seen["call"] = (jid, val)
        return True
    monkeypatch.setattr(mcp_server.store, "set_feedback", fake_set)
    out = await mcp_server.omniswarm_feedback("j1", correct=False)
    assert out["ok"] is True and out["feedback"] == "down"
    assert seen["call"] == ("j1", "down")
