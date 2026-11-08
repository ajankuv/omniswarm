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


def _m(cid, tool_calling=True, ctx=128000):
    return {"id": cid, "provider": cid.split("/", 1)[0], "name": cid, "context_length": ctx,
            "max_output_tokens": 8192, "chat_capable": True,
            "capabilities": {"tool_calling": tool_calling, "reasoning": False, "thinking": False}}


def _proven(*ids):
    return {i: {"calls": 50, "ok": 45, "success_pct": 90.0, "avg_latency_ms": 1000, "http_429": 0} for i in ids}


CATALOG = [
    {"id": "prov/failing", "provider": "prov", "name": "Failing", "context_length": 128000,
     "max_output_tokens": 8192,
     "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
    {"id": "prov/healthy", "provider": "prov", "name": "Healthy", "context_length": 128000,
     "max_output_tokens": 8192,
     "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
]


async def _passes(cid):
    return True


async def test_choose_replacement_skips_excluded():
    rep = await failover.choose_replacement("general", CATALOG, {}, {}, exclude={"prov/failing"},
                                            probe=_passes)
    assert rep == "prov/healthy"
    rep_none = await failover.choose_replacement("general", CATALOG, {}, {},
                                                 exclude={"prov/failing", "prov/healthy"}, probe=_passes)
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
        swaps = await failover.execute(app, "prov/failing", probe=_passes)
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


async def test_choose_replacement_prefers_proven_over_shiny_new():
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
    rep = await failover.choose_replacement("code", catalog, reliability, {}, exclude=set(), probe=_passes)
    assert rep == "old/borderline"           # ...but failover picks the proven one, no probe needed


async def test_choose_replacement_uses_unproven_only_when_probe_passes():
    # No model has a track record. The old code fell back to the raw capability
    # ranking here, which is how the 2026-11-08 incident swapped every slot onto
    # providers that can't run at all. An unproven candidate is only used once a
    # real probe call answers; with no probe (or a failing one) there is no swap.
    catalog = [_m("a/one")]
    assert await failover.choose_replacement("general", catalog, {}, {}, exclude=set()) is None

    async def fails(cid):
        return False
    assert await failover.choose_replacement("general", catalog, {}, {}, exclude=set(),
                                             probe=fails) is None
    assert await failover.choose_replacement("general", catalog, {}, {}, exclude=set(),
                                             probe=_passes) == "a/one"


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
    b.record("gemini/a", ok=False, status="HTTP 403")
    b.record("gemini/a", ok=False, status="HTTP 403")
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


# --- v1.3.1: failover into dead providers (2026-11-08 incident) ---------------------

def test_breaker_trips_provider_on_unavailable_body():
    # "can't run here" (missing CLI/browser, unconfigured transport) benches the whole
    # provider on the FIRST response, for the long unavailable cooldown, not forever.
    b = failover.ProviderBreaker(threshold=4, cooldown=600, unavailable_cooldown=12 * 3600)
    b.record("cfp/kimi", ok=False, status="HTTP 503 UNAVAILABLE")
    assert "cfp" in b.exhausted()
    assert b.exhausted_models([{"id": "cfp/kimi"}, {"id": "cfp/other"}, {"id": "groq/x"}]) == \
        {"cfp/kimi", "cfp/other"}
    assert b.state()["cfp"]["unavailable"] is True
    b._unavailable_at["cfp"] -= 3600          # past the short cooldown: still benched
    b._last_fail_at["cfp"] -= 3600
    assert "cfp" in b.exhausted()
    b._unavailable_at["cfp"] -= 12 * 3600     # past the long cooldown: retried again
    assert "cfp" not in b.exhausted()


def test_breaker_unavailable_cleared_by_success():
    b = failover.ProviderBreaker(threshold=4, cooldown=600, unavailable_cooldown=12 * 3600)
    b.record("aug/gpt5", ok=False, status="HTTP 502 UNAVAILABLE")
    assert "aug" in b.exhausted()
    b.record("aug/kimi", ok=True, status="ok")   # it was fixed: any success clears the provider
    assert "aug" not in b.exhausted()


def test_breaker_404_single_model_does_not_trip_provider():
    # Real-world shape: a provider retires several models (404 "model does not exist")
    # while its other models keep serving. That must bench the gone models, not the
    # whole provider.
    b = failover.ProviderBreaker(threshold=2, cooldown=600)
    gone = ["groq/qwen/qwen3-32b", "groq/meta-llama/llama-4-scout-17b-16e-instruct",
            "groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant"]
    for mid in gone:
        for _ in range(3):
            b.record(mid, ok=False, status="HTTP 404")
    b.record("groq/old-model", ok=False, status="HTTP 410")
    b.record("groq/old-model", ok=False, status="HTTP 410")
    assert "groq" not in b.exhausted()               # the provider still serves its other models
    cat = [{"id": m} for m in gone + ["groq/old-model", "groq/openai/gpt-oss-120b",
                                      "groq/openai/gpt-oss-20b"]]
    assert b.exhausted_models(cat) == set(gone) | {"groq/old-model"}
    b.record("groq/llama-3.1-8b-instant", ok=True, status="ok")   # it came back
    assert "groq/llama-3.1-8b-instant" not in b.exhausted_models(cat)
    # quota/auth codes and explicit signals stay provider-wide
    b.record("groq/openai/gpt-oss-20b", ok=False, status="HTTP 404 EXHAUSTED")
    assert "groq" in b.exhausted()


async def test_choose_replacement_skips_zero_success_history():
    # aug/gpt5 ranks first on capability, but every call it ever made failed. It must
    # not be picked over a never-tried model, and is only usable if a probe passes.
    catalog = [_m("aug/gpt5", ctx=400000), _m("groq/x", ctx=8000)]
    rel = {"aug/gpt5": {"calls": 2, "ok": 0, "success_pct": 0.0, "avg_latency_ms": 100, "http_429": 0}}
    probed = []

    async def probe(cid):
        probed.append(cid)
        return True
    assert await failover.choose_replacement("reasoning", catalog, rel, {}, exclude=set(),
                                             probe=probe) == "groq/x"
    assert probed == ["groq/x"]

    async def only_aug(cid):
        return cid == "aug/gpt5"
    # ...unless a probe actually passes on it
    assert await failover.choose_replacement("reasoning", catalog, rel, {}, exclude=set(),
                                             probe=only_aug) == "aug/gpt5"


async def test_probes_unproven_in_rank_order_max_3():
    catalog = [_m(f"{p}/m") for p in ("a", "b", "c", "d", "e")]   # equal score → id order
    probed = []

    async def fails(cid):
        probed.append(cid)
        return False
    cache = {}
    assert await failover.choose_replacement("general", catalog, {}, {}, exclude=set(),
                                             probe=fails, probe_results=cache) is None
    assert probed == ["a/m", "b/m", "c/m"]               # rank order, never more than 3
    assert cache == {"a/m": False, "b/m": False, "c/m": False}

    probed.clear()

    async def second_passes(cid):
        probed.append(cid)
        return cid == "e/m"
    # cached failures are not re-probed within the same failover
    assert await failover.choose_replacement("general", catalog, {}, {}, exclude=set(),
                                             probe=second_passes, probe_results=cache) == "e/m"
    assert probed == ["d/m", "e/m"]


async def test_choose_replacement_prefers_proven_without_probing():
    catalog = [_m("new/shiny", ctx=256000), _m("old/proven", ctx=4000)]
    probed = []

    async def probe(cid):
        probed.append(cid)
        return True
    rep = await failover.choose_replacement("general", catalog, _proven("old/proven"), {},
                                            exclude=set(), probe=probe)
    assert rep == "old/proven" and probed == []


def _fake_app(tmp_path, runtime=None, catalog=None):
    class _S:
        db_path = str(tmp_path / "x.db")
        omniroute_base_url = "http://gateway.test/v1"
        allowed_providers = frozenset()

    class _State:
        pass

    class _App:
        state = _State()

    app = _App()
    app.state.settings = _S()
    app.state.runtime = runtime or {"models": {}, "judge": "", "synth": "", "members": []}
    app.state.failover = failover.FailoverTracker(threshold=1)
    app.state.provider_breaker = failover.ProviderBreaker(threshold=4, cooldown=600)
    app.state.settings_lock = asyncio.Lock()
    return app


@pytest.fixture()
def wired(monkeypatch):
    """Patch catalog/store/runtime I/O for failover.execute; yields a dict to tune."""
    from omniswarm import catalog as cat_mod
    from omniswarm import events
    from omniswarm import runtime as runtime_mod
    from omniswarm import store as store_mod
    env = {"catalog": CATALOG, "rel": {}, "saved": [], "events": []}

    async def fake_cat(app):
        return env["catalog"]
    monkeypatch.setattr(cat_mod, "get_cached_catalog", fake_cat)
    monkeypatch.setattr(store_mod, "model_reliability", lambda p: env["rel"])
    monkeypatch.setattr(store_mod, "latest_benchmarks", lambda p: {})
    monkeypatch.setattr(runtime_mod, "save_runtime", lambda path, data: env["saved"].append(dict(data)))
    monkeypatch.setattr(events, "publish", lambda ev: env["events"].append(ev))
    yield env
    registry.apply_runtime({})


async def test_execute_probes_outside_settings_lock(wired, tmp_path):
    app = _fake_app(tmp_path)
    probed = []

    async def probe(cid):
        assert not app.state.settings_lock.locked(), "probe ran while holding settings_lock"
        probed.append(cid)
        return True
    registry.apply_runtime({"judge": "prov/failing"})
    swaps = await failover.execute(app, "prov/failing", probe=probe)
    assert probed == ["prov/healthy"]
    assert swaps == {"judge": "prov/healthy"}


async def test_execute_no_swap_when_no_probe_passes(wired, tmp_path):
    app = _fake_app(tmp_path)

    async def fails(cid):
        return False
    registry.apply_runtime({"judge": "prov/failing"})
    assert await failover.execute(app, "prov/failing", probe=fails) is None
    assert registry.JUDGE_MODEL == "prov/failing"          # left alone, not swapped onto a dud
    assert wired["saved"] == []
    assert wired["events"][-1]["note"] == "no healthy replacement available"


async def test_execute_skips_slot_changed_by_operator_mid_failover(wired, tmp_path):
    from omniswarm import runtime as runtime_mod
    wired["catalog"] = CATALOG + [_m("op/choice")]
    app = _fake_app(tmp_path)
    start = {"models": {"code": "prov/failing"}, "judge": "prov/failing", "synth": "", "members": []}
    app.state.runtime = dict(start)
    registry.apply_runtime(start)

    async def operator_post_settings():
        # what POST /settings does: take the lock, write, apply
        async with app.state.settings_lock:
            rt = dict(app.state.runtime)
            rt["models"] = {"code": "op/choice"}
            runtime_mod.save_runtime(None, rt)
            app.state.runtime = rt
            registry.apply_runtime(rt)

    posted = []

    async def probe(cid):
        if not posted:
            # the operator saves while failover is still probing: must not block
            await asyncio.wait_for(operator_post_settings(), timeout=1)
            posted.append(cid)
        return True
    swaps = await failover.execute(app, "prov/failing", probe=probe)
    assert posted                                            # the save went through mid-failover
    assert "code" not in swaps                               # operator's change wins
    assert swaps.get("judge") in {"prov/healthy", "op/choice"}
    assert registry.get_task_type("code").model == "op/choice"
    assert app.state.runtime["models"]["code"] == "op/choice"
    assert wired["saved"][-1]["models"]["code"] == "op/choice"
    assert "code" not in app.state.runtime.get("failover_home", {})


async def test_failover_spreads_slots_across_providers(wired, tmp_path):
    # four proven candidates, one of them on the failing model's own provider
    wired["catalog"] = [_m("a/failing"), _m("a/sibling"), _m("b/one"), _m("c/one"), _m("d/one")]
    wired["rel"] = _proven("a/sibling", "b/one", "c/one", "d/one")
    app = _fake_app(tmp_path)
    pins = {"models": {"general": "a/failing", "summarize": "a/failing", "classify": "a/failing",
                       "draft": "a/failing", "code": "a/failing"},
            "judge": "a/failing", "synth": "", "members": []}
    app.state.runtime = dict(pins)
    registry.apply_runtime(pins)
    swaps = await failover.execute(app, "a/failing", probe=_passes)
    assert set(swaps) == {"general", "summarize", "classify", "draft", "code", "judge"}
    picks = list(swaps.values())
    assert len(set(picks)) == 4                            # every candidate used before any reuse
    first_three = [swaps[k] for k in ("general", "summarize", "classify")]
    assert {p.split("/")[0] for p in first_three} == {"b", "c", "d"}   # other providers first
    from collections import Counter
    assert max(Counter(picks).values()) == 2               # 6 slots over 4 models: reused, spread


async def test_failover_members_get_their_own_replacements(wired, tmp_path):
    wired["catalog"] = [_m("a/failing"), _m("b/one"), _m("c/one"), _m("d/one")]
    wired["rel"] = _proven("b/one", "c/one", "d/one")
    app = _fake_app(tmp_path)
    roles = [m["role"] for m in registry.COUNCIL_MEMBERS][:3]
    pins = {"models": {}, "judge": "", "synth": "",
            "members": [{"role": r, "model": "a/failing"} for r in roles]}
    app.state.runtime = dict(pins)
    registry.apply_runtime(pins)
    swaps = await failover.execute(app, "a/failing", probe=_passes)
    assert {swaps[f"member:{r}"] for r in roles} == {"b/one", "c/one", "d/one"}


async def test_home_model_kept_across_chained_failovers(wired, tmp_path):
    wired["catalog"] = [_m("a/home"), _m("b/second"), _m("c/third")]
    wired["rel"] = _proven("b/second")
    app = _fake_app(tmp_path)
    app.state.runtime = {"models": {}, "judge": "a/home", "synth": "", "members": []}
    registry.apply_runtime(app.state.runtime)
    assert await failover.execute(app, "a/home", probe=_passes) == {"judge": "b/second"}
    assert app.state.runtime["failover_home"] == {"judge": "a/home"}
    assert wired["saved"][-1]["failover_home"] == {"judge": "a/home"}    # persisted
    app.state.failover = failover.FailoverTracker(threshold=1)          # fresh cooldowns
    wired["rel"] = _proven("c/third")
    assert await failover.execute(app, "b/second", probe=_passes) == {"judge": "c/third"}
    assert app.state.runtime["failover_home"] == {"judge": "a/home"}    # the TRUE original


async def test_home_cleared_when_failover_lands_back_home(wired, tmp_path):
    wired["catalog"] = [_m("a/home"), _m("b/second")]
    wired["rel"] = _proven("a/home")
    app = _fake_app(tmp_path)
    app.state.runtime = {"models": {}, "judge": "b/second", "synth": "", "members": [],
                         "failover_home": {"judge": "a/home"}}
    registry.apply_runtime(app.state.runtime)
    assert await failover.execute(app, "b/second", probe=_passes) == {"judge": "a/home"}
    assert app.state.runtime["failover_home"] == {}


async def test_execute_honours_allowed_providers(wired, tmp_path):
    wired["catalog"] = [_m("a/failing"), _m("aug/top", ctx=400000), _m("b/ok")]
    app = _fake_app(tmp_path)
    app.state.settings.allowed_providers = frozenset({"a", "b"})
    app.state.runtime = {"models": {}, "judge": "a/failing", "synth": "", "members": []}
    registry.apply_runtime(app.state.runtime)
    probed = []

    async def probe(cid):
        probed.append(cid)
        return True
    assert await failover.execute(app, "a/failing", probe=probe) == {"judge": "b/ok"}
    assert "aug/top" not in probed


def test_degraded_reports_reasons():
    try:
        registry.apply_runtime({"models": {"code": "groq/now"}, "judge": "gone/model",
                                "synth": "cerebras/x",
                                "members": [{"role": registry.COUNCIL_MEMBERS[0]["role"],
                                             "model": "aug/x"}]})
        pins = dict(failover.current_pins())
        cat = [{"id": mid} for mid in pins.values() if mid != "gone/model"]
        rt = {"failover_home": {"code": "mistral/home"}}
        out = failover.degraded(rt, cat, exhausted_providers={"cerebras"},
                                allowed=frozenset({"groq", "mistral", "nvidia", "cerebras", "gone"}))
        by_slot = {d["slot"]: d for d in out}
        member = f"member:{registry.COUNCIL_MEMBERS[0]['role']}"
        assert set(by_slot) == {"code", "judge", "synth", member}
        assert "home" in by_slot["code"]["reason"] and by_slot["code"]["model"] == "groq/now"
        assert "catalog" in by_slot["judge"]["reason"]
        assert "exhausted" in by_slot["synth"]["reason"]
        assert "not allowed" in by_slot[member]["reason"]
    finally:
        registry.apply_runtime({})


# --- incident replay: 2026-11-08 ------------------------------------------------------
# Free Mistral + Gemini quotas ran out under load. With nothing proven-healthy left, the
# old failover took the raw capability ranking, whose top was gateway providers that
# can never run in the container (missing CLI, missing browser, bad install path,
# unconfigured transport). Their 502/503s were "transient", so they were retried and
# never tripped the breaker, and every slot got the SAME replacement.

_GATEWAY = "http://gateway.test/v1"
_INCIDENT_ERRORS = {
    "aug": (502, "Auggie CLI not found"),
    "cfp": (503, "Cloudflare Playground needs Playwright and its Chromium browser"),
    "dva": (500, "DEVIN_AGENTIC_HOME must be an absolute path"),
    "cxa": (503, "Codex app-server transport is not configured"),
    "mistral": (429, "No active credentials for provider: mistral"),
    "gemini": (429, "Quota exceeded for this project"),
}
_INCIDENT_CATALOG = [
    _m("aug/gpt5", ctx=400000), _m("aug/kimi", ctx=256000), _m("cfp/kimi", ctx=256000),
    _m("dva/swe-1", ctx=200000), _m("cxa/gpt-5-codex", ctx=272000),
    _m("mistral/codestral-latest"), _m("mistral/mistral-medium-3-5"), _m("gemini/flash", ctx=1000000),
    _m("groq/openai/gpt-oss-120b", ctx=131000), _m("groq/llama-3.3-70b"),
    _m("nvidia/meta/llama-4-maverick-17b-128e-instruct"), _m("nvidia/minimaxai/minimax-m3"),
]
_INCIDENT_REL = {
    "aug/gpt5": {"calls": 2, "ok": 0, "success_pct": 0.0, "avg_latency_ms": 100, "http_429": 0},
    "aug/kimi": {"calls": 2, "ok": 0, "success_pct": 0.0, "avg_latency_ms": 100, "http_429": 0},
    "mistral/codestral-latest": {"calls": 500, "ok": 450, "success_pct": 90.0,
                                 "avg_latency_ms": 2000, "http_429": 10},
    "mistral/mistral-medium-3-5": {"calls": 200, "ok": 190, "success_pct": 95.0,
                                   "avg_latency_ms": 2000, "http_429": 2},
    "gemini/flash": {"calls": 300, "ok": 280, "success_pct": 93.3, "avg_latency_ms": 2000, "http_429": 5},
}
_DEAD = ("aug", "cfp", "dva", "cxa")


def _gateway_reply(request):
    import json

    import httpx
    model = json.loads(request.content)["model"]
    err = _INCIDENT_ERRORS.get(model.split("/", 1)[0])
    if err:
        return httpx.Response(err[0], json={"error": {"message": err[1]}})
    return httpx.Response(200, json={"choices": [{"message": {"content": "pong"}}]})


@pytest.mark.parametrize("allowed", [frozenset(), frozenset({"mistral", "gemini", "groq", "nvidia"})],
                         ids=["no-allowlist", "allowlist"])
async def test_incident_replay_2026_11_08(wired, tmp_path, allowed):
    import json
    from collections import Counter

    import httpx
    import respx

    from omniswarm import adapters
    from omniswarm import catalog as cat_mod

    failing = "mistral/codestral-latest"
    wired["catalog"] = cat_mod.filter_allowed(_INCIDENT_CATALOG, allowed)
    wired["rel"] = _INCIDENT_REL
    app = _fake_app(tmp_path)
    app.state.settings.omniroute_base_url = _GATEWAY
    app.state.settings.allowed_providers = allowed
    app.state.failover = failover.FailoverTracker(threshold=6)
    breaker = app.state.provider_breaker
    # an earlier failover had already piled every slot onto one model
    roles = [m["role"] for m in registry.COUNCIL_MEMBERS]
    pins = {"models": {n: failing for n in registry.REGISTRY}, "judge": failing, "synth": failing,
            "members": [{"role": r, "model": failing} for r in roles]}
    app.state.runtime = dict(pins)
    registry.apply_runtime(pins)
    n_slots = len(registry.REGISTRY) + 2 + len(roles)

    def sink(model, ok, lat, status):      # what the app wires up: breaker + failover counter
        breaker.record(model, ok, status)
        app.state.failover.record(model, ok)
    adapters.set_sink(sink)
    try:
        with respx.mock(assert_all_called=False) as gw:
            route = gw.post(f"{_GATEWAY}/chat/completions").mock(side_effect=_gateway_reply)
            async with httpx.AsyncClient() as client:
                app.state.client = client
                # the quota burst: both proven providers say they're out of credits
                for mid in (failing, "gemini/flash"):
                    with pytest.raises(adapters.OmniRouteError):
                        await adapters.generate(client, _GATEWAY, mid, "s", "u", max_retries=2)
                assert {"mistral", "gemini"} <= breaker.exhausted()
                assert route.call_count == 2                     # fail-fast, no retries

                swaps = await failover.execute(app, failing)     # real gateway probe path
                sent = Counter(json.loads(c.request.content)["model"] for c in route.calls)
                probes = sum(sent.values()) - 2
                assert probes <= failover.MAX_PROBES_PER_FAILOVER   # bounded, however many slots
                if not allowed:
                    # the dead providers got probed, failed ONCE each, and are benched now
                    dead = {m: n for m, n in sent.items() if failover._provider_of(m) in _DEAD}
                    assert dead and all(n == 1 for n in dead.values())       # no retries
                    assert {failover._provider_of(m) for m in dead} <= breaker.exhausted()
    finally:
        adapters.set_sink(None)

    assert swaps and len(swaps) == n_slots                         # every slot moved, in ONE pass
    picked = Counter(swaps.values())
    assert {m.split("/", 1)[0] for m in picked} == {"groq", "nvidia"}   # only (both) live providers
    assert len(picked) == (4 if allowed else 3)    # each probe that passed became a distinct pick
    assert max(picked.values()) <= -(-n_slots // len(picked)) + 1  # spread, not piled on one
    assert set(app.state.runtime["failover_home"].values()) == {failing}
    assert not any(failover._provider_of(m) in _DEAD for _, m in failover.current_pins())


# --- gate review fixes -----------------------------------------------------------------

def test_breaker_probe_success_does_not_clear_bench():
    b = failover.ProviderBreaker(threshold=4, cooldown=600, unavailable_cooldown=12 * 3600)
    b.record("aug/x", ok=False, status="probe:HTTP 502 UNAVAILABLE")   # probes still trip it
    assert "aug" in b.exhausted()
    b.record("aug/y", ok=True, status="probe:ok")       # a 5-token probe isn't proof of recovery
    assert "aug" in b.exhausted()
    b.record("aug/y", ok=True, status="ok")             # a real answer is
    assert "aug" not in b.exhausted()
    for _ in range(4):
        b.record("groq/gone", ok=False, status="probe:HTTP 404")
    assert "groq" not in b.exhausted() and "groq/gone" in b.gone_models()


async def test_probe_budget_caps_total_across_slots(wired, tmp_path):
    # 6 slots x 3 probes each would be 18 calls; the whole failover gets a fixed budget
    wired["catalog"] = [_m("a/failing")] + [_m(f"p{i:02d}/m") for i in range(30)]
    app = _fake_app(tmp_path)
    pins = {"models": {n: "a/failing" for n in registry.REGISTRY}, "judge": "", "synth": "",
            "members": []}
    app.state.runtime = dict(pins)
    registry.apply_runtime(pins)
    probed = []

    async def fails(cid):
        probed.append(cid)
        return False
    assert await failover.execute(app, "a/failing", probe=fails) is None
    assert len(probed) == failover.MAX_PROBES_PER_FAILOVER
    assert len(set(probed)) == len(probed)                # never the same model twice


async def test_unpicked_slot_gets_reuse_in_second_pass(wired, tmp_path):
    # the first slot spends its probes on three duds; the second finds a live model.
    # A no-probe second pass then gives the first slot that model instead of no swap.
    wired["catalog"] = [_m("a/failing"), _m("b1/dud"), _m("b2/dud"), _m("b3/dud"), _m("z/live")]
    app = _fake_app(tmp_path)
    pins = {"models": {"general": "a/failing", "summarize": "a/failing"}, "judge": "",
            "synth": "", "members": []}
    app.state.runtime = dict(pins)
    registry.apply_runtime(pins)
    probed = []

    async def only_live(cid):
        probed.append(cid)
        return cid == "z/live"
    swaps = await failover.execute(app, "a/failing", probe=only_live)
    assert swaps == {"general": "z/live", "summarize": "z/live"}
    assert probed == ["b1/dud", "b2/dud", "b3/dud", "z/live"]   # the second pass probed nothing
