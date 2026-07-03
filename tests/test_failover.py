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
