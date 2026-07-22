import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from omniswarm import app as app_module, engine, council


@pytest.fixture()
def client(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("OMNISWARM_DB_PATH", path)

    async def fake_process(client, settings, req, store_mode="full", active_roster=None, always_council=False, job_id=None):
        from omniswarm import store
        from omniswarm.util import new_id
        jid = job_id or new_id()
        store.create_job(settings.db_path, jid, req.task_type, "done")
        store.update_job(settings.db_path, jid, status="done", verdict="pass",
                         confidence="high", result="vetted", tokens_saved=7)
        return {"job_id": jid, "text": "vetted", "verdict": "pass",
                "confidence": "high", "models_used": ["m1"], "tokens_saved": 7,
                "status": "done"}

    monkeypatch.setattr(app_module.engine, "process_job", fake_process)
    with TestClient(app_module.create_app()) as c:
        yield c
    os.remove(path)


def test_chat_completion_returns_openai_shape_plus_omniswarm(client):
    resp = client.post("/v1/chat/completions", json={
        "model": "omniswarm",
        "messages": [
            {"role": "system", "content": "you summarize"},
            {"role": "user", "content": "summarize X"},
        ],
        "omniswarm": {"task_type": "summarize"},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "vetted"
    assert body["omniswarm"]["verdict"] == "pass"
    assert body["omniswarm"]["confidence"] == "high"


def test_jobs_listing_after_a_call(client):
    client.post("/v1/chat/completions", json={
        "model": "omniswarm",
        "messages": [{"role": "user", "content": "hi"}],
    })
    jobs = client.get("/jobs").json()
    assert len(jobs) >= 1
    assert jobs[0]["status"] == "done"


def test_startup_survives_failing_health_check(monkeypatch):
    """Startup must not crash when the OmniRoute health check raises an exception."""
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("OMNISWARM_DB_PATH", path)

    async def exploding_health(client, base_url):
        raise RuntimeError("simulated health-check failure")

    monkeypatch.setattr("omniswarm.app.health", exploding_health)

    try:
        with TestClient(app_module.create_app()) as c:
            # App started without raising — verify /jobs responds normally
            resp = c.get("/jobs")
            assert resp.status_code == 200
    finally:
        os.remove(path)


def test_stats_endpoint(client):
    client.post("/v1/chat/completions", json={
        "model": "omniswarm",
        "messages": [{"role": "user", "content": "hi"}],
    })
    s = client.get("/stats").json()
    assert s["total_jobs"] >= 1
    assert "tokens_saved" in s
    assert "by_status" in s and "by_verdict" in s


def test_dashboard_root_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "OMNI" in resp.text  # the dashboard heading


def test_jobs_rows_include_detail_fields(client):
    client.post("/v1/chat/completions", json={
        "model": "omniswarm",
        "messages": [{"role": "user", "content": "hi"}],
    })
    row = client.get("/jobs").json()[0]
    assert "input" in row
    assert "models_used" in row


def test_dashboard_has_detail_markup(client):
    html = client.get("/").text
    assert "Provenance" in html  # detail panel label


def test_stats_has_enriched_keys(client):
    client.post("/v1/chat/completions", json={
        "model": "omniswarm", "messages": [{"role": "user", "content": "hi"}]})
    s = client.get("/stats").json()
    for k in ("by_model", "by_task_type", "by_confidence", "most_used_model",
              "total_model_calls", "council_engaged", "avg_tokens_saved"):
        assert k in s


def test_stats_reports_dollars_saved(client):
    client.post("/v1/chat/completions", json={
        "model": "omniswarm", "messages": [{"role": "user", "content": "hi"}]})
    s = client.get("/stats").json()
    # default rate is $5/Mtok (config.py); dollars_saved is derived, non-negative
    assert s["usd_per_mtok"] == 5.0
    assert s["dollars_saved"] == round(s["tokens_saved"] / 1_000_000 * 5.0, 2)


def test_dashboard_has_dollars_tile(client):
    html = client.get("/").text
    assert 'id="dollars"' in html
    assert "saved vs premium" in html


def test_dashboard_has_analytics_markup(client):
    html = client.get("/").text
    assert "Free models at work" in html   # model leaderboard panel
    assert "Provenance" in html            # per-job timeline label


def test_dashboard_renders_member_role_in_timeline(client):
    html = client.get("/").text
    # the timeline must surface the member role (s.role) on review steps
    assert "s.role" in html


def test_settings_get_hides_token(client):
    s = client.get("/settings").json()
    assert "api_token" not in s
    assert "protected" in s and s["protected"] is False
    assert "store_mode" in s and "rate_limit_per_min" in s


def test_token_protects_endpoints(monkeypatch, tmp_path):
    import omniswarm.app as app_module
    monkeypatch.setenv("OMNISWARM_DB_PATH", str(tmp_path / "d.db"))
    monkeypatch.setenv("OMNISWARM_RUNTIME", str(tmp_path / "rt.json"))
    from fastapi.testclient import TestClient
    with TestClient(app_module.create_app()) as c:
        # set a token via POST (open while no token set yet)
        assert c.post("/settings", json={"api_token": "sk-test"}).status_code == 200
        # now data endpoints require it
        assert c.get("/jobs").status_code == 401
        assert c.get("/jobs", headers={"Authorization": "Bearer sk-test"}).status_code == 200
        assert c.get("/jobs?token=sk-test").status_code == 200


def test_dashboard_links_to_control_panel(client):
    html = client.get("/").text
    # dashboard no longer embeds the panel — it links to the dedicated page via the nav
    assert 'href="/control-panel"' in html
    assert 'class="nav"' in html


def test_control_panel_page_served(client):
    resp = client.get("/control-panel")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    html = resp.text
    assert "Control Panel" in html
    assert "store_mode" in html       # privacy control wired
    assert 'data-mode="redact"' in html  # the polished mode cards


def test_shared_css_served(client):
    resp = client.get("/static/omniswarm.css")
    assert resp.status_code == 200
    assert ".nav" in resp.text


def test_settings_exposes_members_and_active_roster(client):
    s = client.get("/settings").json()
    assert "members" in s and isinstance(s["members"], list)
    assert any(m["role"] == "Security Engineer" for m in s["members"])
    assert "active_roster" in s


def test_settings_accepts_active_roster(monkeypatch, tmp_path):
    import omniswarm.app as app_module
    monkeypatch.setenv("OMNISWARM_DB_PATH", str(tmp_path / "d.db"))
    monkeypatch.setenv("OMNISWARM_RUNTIME", str(tmp_path / "rt.json"))
    from fastapi.testclient import TestClient
    with TestClient(app_module.create_app()) as c:
        r = c.post("/settings", json={"active_roster": ["Fact-Checker", "User Advocate"]})
        assert r.status_code == 200
        assert c.get("/settings").json()["active_roster"] == ["Fact-Checker", "User Advocate"]


def test_control_panel_has_roster_section(client):
    html = client.get("/control-panel").text
    assert "Council Roster" in html
    assert "active_roster" in html


def test_dashboard_has_run_console(client):
    html = client.get("/").text
    assert "Run a job" in html
    assert 'id="c-prompt"' in html


def test_jobs_filter_and_export(client):
    client.post("/v1/chat/completions", json={"model": "omniswarm",
        "messages": [{"role": "user", "content": "hi"}]})
    # filter passthrough
    r = client.get("/jobs?status=done")
    assert r.status_code == 200
    # export json + csv
    j = client.get("/export?format=json")
    assert j.status_code == 200 and "application/json" in j.headers["content-type"]
    c = client.get("/export?format=csv")
    assert c.status_code == 200 and "text/csv" in c.headers["content-type"]
    assert "attachment" in c.headers.get("content-disposition", "")


def test_dashboard_has_filter_bar(client):
    html = client.get("/").text
    assert 'id="f-search"' in html
    assert "Escalated/failed" in html
    assert "Export CSV" in html


def test_reliability_endpoint(client):
    r = client.get("/reliability")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_dashboard_has_reliability_panel(client):
    html = client.get("/").text
    assert "Model reliability" in html
    assert 'id="p-reliability"' in html


def test_settings_exposes_always_council(client):
    s = client.get("/settings").json()
    assert "always_council" in s and isinstance(s["always_council"], bool)


def test_stream_route_registered(client):
    # SSE is an infinite stream; Starlette's TestClient buffers responses, so we don't
    # consume it here. Verify the route is wired; live streaming is verified via curl.
    paths = {getattr(r, "path", None) for r in client.app.routes}
    assert "/stream" in paths


def test_dashboard_has_live_eventsource(client):
    html = client.get("/").text
    assert "EventSource" in html
    assert 'id="live"' in html


def test_dashboard_has_live_feed(client):
    html = client.get("/").text
    assert 'id="livefeed"' in html
    assert "Live activity" in html


def test_submit_job_returns_immediately(client):
    r = client.post("/v1/jobs", json={"model": "omniswarm",
        "messages": [{"role": "user", "content": "do a thing"}],
        "omniswarm": {"task_type": "general"}})
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "running"


def test_schedules_api(client):
    r = client.post("/schedules", json={"prompt": "summarize logs", "task_type": "summarize", "interval_minutes": 30})
    assert r.status_code == 200
    sid = r.json()["id"]
    lst = client.get("/schedules").json()
    assert any(s["id"] == sid for s in lst)
    assert client.delete(f"/schedules/{sid}").status_code == 200
    assert all(s["id"] != sid for s in client.get("/schedules").json())


def test_schedule_interval_floor(client):
    r = client.post("/schedules", json={"prompt": "x", "task_type": "general", "interval_minutes": 0})
    assert r.status_code == 200
    assert r.json()["interval_seconds"] >= 60   # floored, never sub-minute


def test_control_panel_has_schedules_section(client):
    html = client.get("/control-panel").text
    assert "Scheduled jobs" in html
    assert 'id="sch-add"' in html


def test_models_available(client, monkeypatch):
    from omniswarm import catalog
    async def fake(app):
        return [{"id": "nvidia/x", "provider": "nvidia", "name": "x",
                 "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 128000, "max_output_tokens": 8192, "chat_capable": True}]
    monkeypatch.setattr(catalog, "get_cached_catalog", fake)
    r = client.get("/models/available")
    assert r.status_code == 200
    body = r.json()
    assert body["models"][0]["id"] == "nvidia/x"
    assert "reliability" in body["models"][0]  # merged reliability field present (may be null)


def test_models_recommend(client, monkeypatch):
    from omniswarm import catalog
    async def fake(app):
        return [{"id": "mistral/devstral-latest", "provider": "mistral", "name": "Devstral",
                 "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 256000, "max_output_tokens": 8192, "chat_capable": True}]
    monkeypatch.setattr(catalog, "get_cached_catalog", fake)
    r = client.get("/models/recommend")
    assert r.status_code == 200
    body = r.json()
    assert "code" in body and "judge" in body and "synth" in body
    assert body["code"]["recommended"] == "mistral/devstral-latest"


def test_settings_accepts_and_applies_model_overrides(client, monkeypatch):
    from omniswarm import catalog, registry
    async def fake(app):
        return [{"id": "mistral/devstral-latest", "provider": "mistral", "name": "Devstral",
                 "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 256000, "max_output_tokens": 8192, "chat_capable": True},
                {"id": "nvidia/meta/llama-4-maverick-17b-128e-instruct", "provider": "nvidia",
                 "name": "Maverick", "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 128000, "max_output_tokens": 8192, "chat_capable": True}]
    monkeypatch.setattr(catalog, "get_cached_catalog", fake)
    try:
        r = client.post("/settings", json={"models": {"code": "mistral/devstral-latest"},
                                           "judge": "nvidia/meta/llama-4-maverick-17b-128e-instruct"})
        assert r.status_code == 200
        assert registry.get_task_type("code").model == "mistral/devstral-latest"
        assert registry.JUDGE_MODEL == "nvidia/meta/llama-4-maverick-17b-128e-instruct"
    finally:
        registry.apply_runtime({})


def test_settings_rejects_unknown_model_id(client, monkeypatch):
    from omniswarm import catalog, registry
    async def fake(app):
        return [{"id": "mistral/devstral-latest", "provider": "mistral", "name": "Devstral",
                 "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 256000, "max_output_tokens": 8192, "chat_capable": True}]
    monkeypatch.setattr(catalog, "get_cached_catalog", fake)
    try:
        r = client.post("/settings", json={"models": {"code": "auto/best-coding"}})
        assert r.status_code == 200
        assert registry.get_task_type("code").model != "auto/best-coding"  # rejected
    finally:
        registry.apply_runtime({})


def test_probe_rejects_unknown_id(client, monkeypatch):
    from omniswarm import catalog
    async def fake(app):
        return [{"id": "nvidia/x", "provider": "nvidia", "name": "x",
                 "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 0, "max_output_tokens": 0, "chat_capable": True}]
    monkeypatch.setattr(catalog, "get_cached_catalog", fake)
    r = client.get("/models/probe?id=bogus/model")
    assert r.status_code == 400


def test_control_panel_has_model_picker(client):
    html = client.get("/control-panel").text
    assert "Model Picker" in html
    assert 'id="mp-list"' in html
    assert 'id="mp-save"' in html


def test_settings_returns_judge_and_synth(client):
    s = client.get("/settings").json()
    assert "judge" in s and "synth" in s
    assert s["judge"] == "nvidia/meta/llama-4-maverick-17b-128e-instruct"
    assert s["synth"] == "mistral/mistral-medium-3-5"


def test_settings_does_not_wipe_models_when_catalog_empty(client, monkeypatch):
    from omniswarm import catalog, registry
    async def empty_cat(app):
        return []
    monkeypatch.setattr(catalog, "get_cached_catalog", empty_cat)
    try:
        before = client.get("/settings").json()["models"]["code"]
        r = client.post("/settings", json={"models": {"code": "mistral/devstral-latest"},
                                           "store_mode": "redact"})
        assert r.status_code == 200
        after = client.get("/settings").json()["models"]["code"]
        assert after == before  # model map untouched when catalog unavailable
        assert client.get("/settings").json()["store_mode"] == "redact"  # non-model setting still applied
    finally:
        registry.apply_runtime({})
        client.post("/settings", json={"store_mode": "full"})


def test_benchmark_endpoint_validates_and_launches(client, monkeypatch):
    from omniswarm import catalog, benchmarks
    async def fake_cat(app):
        return [{"id": "mistral/devstral-latest", "provider": "mistral", "name": "D",
                 "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 256000, "max_output_tokens": 8192, "chat_capable": True}]
    called = {}
    async def fake_run(client_, base, task_type, candidates, on_progress=None):
        called["candidates"] = candidates
        return [{"model": candidates[0], "task_type": task_type, "quality": 1.0,
                 "pass_rate": 1.0, "avg_latency_ms": 10.0, "samples": 8, "failures": 0}]
    monkeypatch.setattr(catalog, "get_cached_catalog", fake_cat)
    monkeypatch.setattr(benchmarks, "run_benchmark", fake_run)
    r = client.post("/benchmark", json={"task_type": "code", "candidates": ["mistral/devstral-latest"]})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "running" and body["calls"] == benchmarks.bank_size("code")


def test_benchmark_rejects_unknown_candidate(client, monkeypatch):
    from omniswarm import catalog
    async def fake_cat(app):
        return [{"id": "mistral/devstral-latest", "provider": "mistral", "name": "D",
                 "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 256000, "max_output_tokens": 8192, "chat_capable": True}]
    monkeypatch.setattr(catalog, "get_cached_catalog", fake_cat)
    r = client.post("/benchmark", json={"task_type": "code", "candidates": ["auto/best-coding"]})
    assert r.status_code == 400


def test_benchmarks_get_shape(client):
    r = client.get("/benchmarks")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_control_panel_has_benchmark(client):
    html = client.get("/control-panel").text
    assert "Benchmark" in html
    assert 'id="bench-run"' in html
    assert 'id="bench-results"' in html


def test_benchmark_rejects_unknown_task_type(client, monkeypatch):
    from omniswarm import catalog
    async def fake_cat(app):
        return [{"id": "mistral/devstral-latest", "provider": "mistral", "name": "D",
                 "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 256000, "max_output_tokens": 8192, "chat_capable": True}]
    monkeypatch.setattr(catalog, "get_cached_catalog", fake_cat)
    r = client.post("/benchmark", json={"task_type": "not_a_type", "candidates": ["mistral/devstral-latest"]})
    assert r.status_code == 400


def test_benchmark_rejects_unknown_task_type(client, monkeypatch):
    from omniswarm import catalog
    async def fake_cat(app):
        return [{"id": "mistral/devstral-latest", "provider": "mistral", "name": "D",
                 "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False},
                 "context_length": 256000, "max_output_tokens": 8192, "chat_capable": True}]
    monkeypatch.setattr(catalog, "get_cached_catalog", fake_cat)
    r = client.post("/benchmark", json={"task_type": "not_a_type", "candidates": ["mistral/devstral-latest"]})
    assert r.status_code == 400


def test_failover_wired_into_app(client):
    from omniswarm import adapters
    assert client.app.state.failover is not None            # tracker attached at startup
    assert client.app.state.settings_lock is not None       # shared save lock exists
    assert adapters._SINK is not None                       # sink installed


def test_dashboard_has_failover_banner(client):
    html = client.get("/").text
    assert 'id="failover-banner"' in html
    assert "showFailover" in html


def test_feedback_and_calibration_endpoints(client):
    # create a job via a completion, then rate it
    client.post("/v1/chat/completions", json={
        "model": "omniswarm", "messages": [{"role": "user", "content": "hi"}]})
    jid = client.get("/jobs").json()[0]["id"]
    r = client.post(f"/jobs/{jid}/feedback", json={"correct": True})
    assert r.status_code == 200 and r.json()["feedback"] == "up"
    # unknown job -> 404
    assert client.post("/jobs/nope/feedback", json={"correct": False}).status_code == 404
    cal = client.get("/calibration").json()
    assert cal["total_rated"] == 1
    assert cal["by_confidence"]["high"]["pct_correct"] == 100.0


def test_stats_includes_cache_block(client):
    s = client.get("/stats").json()
    assert "cache" in s and set(s["cache"]) == {"entries", "hits"}


def test_dashboard_has_calibration_and_feedback_markup(client):
    html = client.get("/").text
    assert "Calibration" in html
    assert "sendFeedback" in html   # the 👍/👎 handler


def test_tightening_privacy_mode_purges_cached_plaintext(client):
    """README promises cached text never outlives your privacy setting."""
    from omniswarm import store
    db = client.app.state.settings.db_path
    store.cache_put(db, "k1", "general", "secret answer", "pass", "high", '["m"]')
    assert store.cache_stats(db)["entries"] == 1
    assert client.post("/settings", json={"store_mode": "redact"}).status_code == 200
    assert store.cache_stats(db)["entries"] == 0        # purged
    # switching back to full does not resurrect anything
    client.post("/settings", json={"store_mode": "full"})
    assert store.cache_stats(db)["entries"] == 0


def test_csv_export_neutralizes_formula_injection(client):
    from omniswarm import store
    db = client.app.state.settings.db_path
    store.create_job(db, "evil", "general", "done")
    store.update_job(db, "evil", status="done", input='=cmd|\'/c calc\'!A0', result="+1+1",
                     note="@SUM(1)", tokens_saved=-5)
    csv_text = client.get("/export?format=csv").text
    assert "'=cmd" in csv_text and "'+1+1" in csv_text and "'@SUM" in csv_text
    assert "\n=cmd" not in csv_text and ",=cmd" not in csv_text


def test_auth_uses_constant_time_compare(monkeypatch, tmp_path):
    import omniswarm.app as app_module
    monkeypatch.setenv("OMNISWARM_DB_PATH", str(tmp_path / "d.db"))
    monkeypatch.setenv("OMNISWARM_RUNTIME", str(tmp_path / "rt.json"))
    from fastapi.testclient import TestClient
    with TestClient(app_module.create_app()) as c:
        c.post("/settings", json={"api_token": "sk-secret"})
        assert c.get("/jobs").status_code == 401                       # missing
        assert c.get("/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/jobs", headers={"Authorization": "Bearer sk-secret"}).status_code == 200


def test_auth_rejects_non_ascii_token_without_500(monkeypatch, tmp_path):
    """A non-ASCII token must be a clean 401, not a 500. compare_digest raises
    TypeError on non-ASCII str; HTTP headers are ASCII-only, but the ?token=
    query param carries arbitrary unicode, so that is the reachable path."""
    import omniswarm.app as app_module
    monkeypatch.setenv("OMNISWARM_DB_PATH", str(tmp_path / "d.db"))
    monkeypatch.setenv("OMNISWARM_RUNTIME", str(tmp_path / "rt.json"))
    from fastapi.testclient import TestClient
    with TestClient(app_module.create_app(), raise_server_exceptions=False) as c:
        c.post("/settings", json={"api_token": "sk-secret"})
        r = c.get("/jobs", params={"token": "tokén-ünicode-💥"})
        assert r.status_code == 401
