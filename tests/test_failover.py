import asyncio

import pytest

from omniswarm import failover, registry


def test_tracker_trips_once_at_threshold():
    t = failover.FailoverTracker(threshold=3)
    assert t.record("m/x", ok=False) is False
    assert t.record("m/x", ok=False) is False
    assert t.record("m/x", ok=False) is True       # trips exactly at threshold
    assert t.record("m/x", ok=False) is False      # in-flight: no duplicate trigger
    t.done("m/x")
    assert t.record("m/x", ok=False) is True       # can trip again after done()


def test_success_resets_counter():
    t = failover.FailoverTracker(threshold=2)
    assert t.record("m/x", ok=False) is False
    assert t.record("m/x", ok=True) is False       # reset
    assert t.record("m/x", ok=False) is False      # back to 1, no trip


def test_threshold_zero_disables():
    t = failover.FailoverTracker(threshold=0)
    for _ in range(10):
        assert t.record("m/x", ok=False) is False


def test_unhealthy_includes_failing_and_cooldown():
    t = failover.FailoverTracker(threshold=5, cooldown=999)
    t.record("m/flaky", ok=False)                  # 1 failure -> unhealthy
    assert "m/flaky" in t.unhealthy()
    t2 = failover.FailoverTracker(threshold=1, cooldown=999)
    t2.record("m/tripped", ok=False)               # trips -> cooldown set
    t2.done("m/tripped")
    assert "m/tripped" in t2.unhealthy()           # still cooling down


CATALOG = [
    {"id": "prov/failing", "provider": "prov", "name": "Failing", "context_length": 128000,
     "max_output_tokens": 8192,
     "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
    {"id": "prov/healthy", "provider": "prov", "name": "Healthy", "context_length": 128000,
     "max_output_tokens": 8192,
     "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
]


def test_choose_replacement_skips_excluded():
    rep = failover.choose_replacement("general", CATALOG, {}, {}, exclude={"prov/failing"})
    assert rep == "prov/healthy"
    rep_none = failover.choose_replacement("general", CATALOG, {}, {},
                                           exclude={"prov/failing", "prov/healthy"})
    assert rep_none is None


def test_affected_slots_detects_pins():
    try:
        registry.apply_runtime({"judge": "test/failing-model",
                                "models": {"code": "test/failing-model"}})
        slots = failover.affected_slots("test/failing-model")
        assert slots["judge"] is True
        assert "code" in slots["models"]
        assert slots["synth"] is False
    finally:
        registry.apply_runtime({})


@pytest.mark.asyncio
async def test_execute_swaps_affected_slots(monkeypatch, tmp_path):
    from omniswarm import catalog as cat_mod, events, runtime as runtime_mod, store as store_mod

    async def fake_cat(app):
        return CATALOG
    monkeypatch.setattr(cat_mod, "get_cached_catalog", fake_cat)
    monkeypatch.setattr(store_mod, "model_reliability", lambda p: {})
    monkeypatch.setattr(store_mod, "latest_benchmarks", lambda p: {})
    saved = {}
    monkeypatch.setattr(runtime_mod, "save_runtime", lambda path, data: saved.update(data))
    published = []
    monkeypatch.setattr(events, "publish", lambda ev: published.append(ev))

    class _S:
        db_path = str(tmp_path / "x.db")

    class _State:
        pass

    class _App:
        state = _State()

    app = _App()
    app.state.settings = _S()
    app.state.runtime = {"models": {}, "judge": "", "synth": "", "members": []}
    app.state.failover = failover.FailoverTracker(threshold=1)
    app.state.settings_lock = asyncio.Lock()

    try:
        registry.apply_runtime({"judge": "prov/failing"})
        swaps = await failover.execute(app, "prov/failing")
        assert swaps == {"judge": "prov/healthy"}
        assert registry.JUDGE_MODEL == "prov/healthy"          # applied live
        assert saved.get("judge") == "prov/healthy"            # persisted
        assert published and published[-1]["type"] == "failover"
        assert published[-1]["from"] == "prov/failing"
    finally:
        registry.apply_runtime({})


@pytest.mark.asyncio
async def test_execute_noop_when_model_not_pinned(monkeypatch, tmp_path):
    from omniswarm import catalog as cat_mod

    async def fake_cat(app):
        return CATALOG
    monkeypatch.setattr(cat_mod, "get_cached_catalog", fake_cat)

    class _S:
        db_path = str(tmp_path / "x.db")

    class _State:
        pass

    class _App:
        state = _State()

    app = _App()
    app.state.settings = _S()
    app.state.runtime = {}
    app.state.failover = failover.FailoverTracker(threshold=1)
    app.state.settings_lock = asyncio.Lock()
    assert await failover.execute(app, "prov/not-used-anywhere") is None


def test_choose_replacement_prefers_proven_over_shiny_new():
    # A capable ZERO-history model that RAW-OUTRANKS a borderline-but-proven one.
    # Failover must still pick the proven one: a shiny model with no track record
    # could be dead (the gpt-5-mini/o3-mini → 404 quota cascade).
    from omniswarm import recommend
    catalog = [
        {"id": "new/shiny", "provider": "new", "name": "Shiny Coder", "context_length": 256000,
         "max_output_tokens": 16384,
         "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
        {"id": "old/borderline", "provider": "old", "name": "Borderline", "context_length": 4000,
         "max_output_tokens": 2048,
         "capabilities": {"tool_calling": False, "reasoning": False, "thinking": False}, "chat_capable": True},
    ]
    # 72% success → empirical PENALTY ranks it below the shiny zero-history model,
    # but it's still proven-healthy (>=70% over real calls).
    reliability = {"old/borderline": {"calls": 200, "success_pct": 72.0, "avg_latency_ms": 3000, "http_429": 0}}
    ranked = [c["id"] for c in recommend.recommend("code", catalog, reliability, {})["ranked"]]
    assert ranked[0] == "new/shiny"          # raw ranking puts the shiny one on top...
    rep = failover.choose_replacement("code", catalog, reliability, {}, exclude=set())
    assert rep == "old/borderline"           # ...but failover picks the proven one


def test_choose_replacement_falls_back_when_none_proven():
    # No model has a track record → fall back to the capability ranking (best effort).
    catalog = [
        {"id": "a/one", "provider": "a", "name": "One", "context_length": 8000, "max_output_tokens": 4096,
         "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
    ]
    rep = failover.choose_replacement("general", catalog, {}, {}, exclude=set())
    assert rep == "a/one"


def test_provider_breaker_trips_on_quota_errors():
    b = failover.ProviderBreaker(threshold=3, cooldown=600)
    # 3 permanent (quota/auth) errors on gemini/* trips the whole provider
    for i in range(3):
        b.record("gemini/model-a" if i % 2 else "gemini/model-b", ok=False, status="HTTP 429")
    assert "gemini" in b.exhausted()
    # a different provider is unaffected
    assert "mistral" not in b.exhausted()


def test_provider_breaker_ignores_transient_and_resets_on_success():
    b = failover.ProviderBreaker(threshold=2, cooldown=600)
    b.record("prov/x", ok=False, status="HTTP 503")   # transient — doesn't count
    b.record("prov/x", ok=False, status="Timeout")    # transient — doesn't count
    assert "prov" not in b.exhausted()
    b.record("prov/x", ok=False, status="HTTP 403")   # permanent
    b.record("prov/x", ok=True, status="ok")          # success resets the provider
    b.record("prov/x", ok=False, status="HTTP 403")
    assert "prov" not in b.exhausted()                # count was reset, only 1 since


def test_provider_breaker_exhausted_models_and_state():
    b = failover.ProviderBreaker(threshold=2, cooldown=600)
    b.record("gemini/a", ok=False, status="HTTP 410")
    b.record("gemini/a", ok=False, status="HTTP 410")
    catalog = [{"id": "gemini/a"}, {"id": "gemini/b"}, {"id": "mistral/c"}]
    assert b.exhausted_models(catalog) == {"gemini/a", "gemini/b"}
    st = b.state()
    assert st["gemini"]["exhausted"] is True and st["gemini"]["perm_fails"] == 2


def test_provider_breaker_cooldown_expiry():
    b = failover.ProviderBreaker(threshold=1, cooldown=60)
    b.record("x/m", ok=False, status="HTTP 429")
    assert "x" in b.exhausted()
    b._last_fail_at["x"] -= 120               # simulate cooldown elapsed (deterministic)
    assert "x" not in b.exhausted()           # breaker resets after cooldown


def test_provider_breaker_can_retrip_after_recovery():
    # QC-found bug: a one-shot latch (n == threshold) could never re-trip after cooldown.
    # Deterministic: age the last-fail timestamp instead of sleeping (no timing flake).
    b = failover.ProviderBreaker(threshold=2, cooldown=60)
    b.record("g/a", ok=False, status="HTTP 429")
    b.record("g/a", ok=False, status="HTTP 429")
    assert "g" in b.exhausted()                 # tripped
    b._last_fail_at["g"] -= 120                 # simulate cooldown fully elapsed
    assert "g" not in b.exhausted()             # cooled down
    # it keeps failing after cooldown → must trip AGAIN (n stays >= threshold, fresh fail)
    b.record("g/a", ok=False, status="HTTP 429")
    assert "g" in b.exhausted()                 # re-tripped (the bug: this used to stay clear)


def test_provider_breaker_state_shape_after_recovery():
    b = failover.ProviderBreaker(threshold=1, cooldown=600)
    b.record("x/m", ok=False, status="HTTP 403")
    b.record("x/m", ok=True, status="ok")       # success clears count
    st = b.state()
    assert st["x"]["exhausted"] is False and st["x"]["perm_fails"] == 0


def test_provider_breaker_threshold_zero_disables():
    b = failover.ProviderBreaker(threshold=0, cooldown=600)
    for _ in range(20):
        b.record("g/a", ok=False, status="HTTP 429")
    assert b.exhausted() == set() and b.state() == {}


def test_provider_breaker_fast_trips_on_explicit_signal():
    # ONE explicit "no active credentials" response trips the breaker immediately,
    # without waiting for `threshold` blind failures.
    b = failover.ProviderBreaker(threshold=4, cooldown=600)
    b.record("mistral/codestral", ok=False, status="HTTP 404 EXHAUSTED")
    assert "mistral" in b.exhausted()               # tripped on the first explicit signal
    assert b.state()["mistral"]["perm_fails"] >= 4


def test_provider_breaker_ambiguous_code_still_counts():
    # A bare 403 (no explicit signal) still needs `threshold` occurrences.
    b = failover.ProviderBreaker(threshold=3, cooldown=600)
    b.record("x/m", ok=False, status="HTTP 403")
    b.record("x/m", ok=False, status="HTTP 403")
    assert "x" not in b.exhausted()                 # only 2 < 3
    b.record("x/m", ok=False, status="HTTP 403")
    assert "x" in b.exhausted()
