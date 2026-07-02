from omniswarm.registry import get_task_type, REGISTRY, COUNCIL_GENERATORS
from omniswarm import registry


def test_known_task_returns_pinned_model():
    assert get_task_type("code").model == "mistral/devstral-latest"
    assert get_task_type("reasoning").model == "mistral/mistral-medium-3-5"


def test_unknown_falls_back_to_general():
    assert get_task_type("nonsense").name == "general"
    assert get_task_type(None).name == "general"


def test_no_auto_or_tllm_models_anywhere():
    for t in REGISTRY.values():
        assert not t.model.startswith("auto/")
        assert not t.model.startswith("tllm/")
    for m in COUNCIL_GENERATORS:
        assert not m.startswith(("auto/", "tllm/"))


def test_council_generators_are_cross_provider():
    providers = {m.split("/")[0] for m in COUNCIL_GENERATORS}
    assert len(providers) >= 2  # diversity is the point


def test_build_registry_honors_config_models():
    cfg = {
        "models": {"general": "x/gen", "code": "x/coder", "reasoning": "x/reason"},
        "judge": "x/judge", "synth": "x/synth", "council_generators": ["x/a", "x/b"],
    }
    reg = registry.build_registry(cfg)
    assert reg["code"].model == "x/coder"
    assert reg["reasoning"].model == "x/reason"
    assert reg["reasoning"].high_stakes is True  # metadata still applied from code
    assert reg["general"].model == "x/gen"


def test_resolve_roster_default_override_and_active():
    from omniswarm import registry
    # default roster (config) — 3 members, all from the library
    default = [m["role"] for m in registry.resolve_roster("general")]
    assert len(default) == 3
    # per-task override: code gets the Security Engineer
    code_roles = [m["role"] for m in registry.resolve_roster("code")]
    assert "Security Engineer" in code_roles
    # active_roster (runtime) wins when provided
    active = registry.resolve_roster("general", ["Fact-Checker", "Safety Sentinel"])
    assert [m["role"] for m in active] == ["Fact-Checker", "Safety Sentinel"]
    # unknown roles ignored; empty resolution falls back to config
    fb = registry.resolve_roster("general", ["Nonexistent"])
    assert len(fb) >= 1
    # never auto/tllm
    assert all(not m["model"].startswith(("auto/", "tllm/")) for m in registry.COUNCIL_MEMBERS)


def test_effective_config_overlays_overrides():
    cfg = registry.effective_config({
        "models": {"code": "x/coder"}, "judge": "x/judge", "synth": "x/synth",
        "members": [{"role": "Fact-Checker", "model": "x/fc"}],
    })
    assert cfg["models"]["code"] == "x/coder"
    assert cfg["models"]["general"] != "x/coder"   # other tasks untouched
    assert cfg["judge"] == "x/judge" and cfg["synth"] == "x/synth"
    fc = next(m for m in cfg["members"] if m["role"] == "Fact-Checker")
    assert fc["model"] == "x/fc"


def test_empty_overrides_are_noop():
    base = registry.effective_config({})
    assert base["models"]["general"] == "nvidia/meta/llama-4-maverick-17b-128e-instruct"


def test_apply_runtime_rebuilds_globals():
    try:
        registry.apply_runtime({"models": {"code": "x/coder"}, "judge": "x/judge",
                                "synth": "x/synth"})
        assert registry.get_task_type("code").model == "x/coder"
        assert registry.JUDGE_MODEL == "x/judge"
        assert registry.COUNCIL_SYNTH_MODEL == "x/synth"
    finally:
        registry.apply_runtime({})   # restore defaults for other tests
        assert registry.JUDGE_MODEL == "nvidia/meta/llama-4-maverick-17b-128e-instruct"


def test_effective_config_rejects_auto_and_tllm_overrides():
    cfg = registry.effective_config({
        "models": {"code": "auto/best-coding", "general": "mistral/mistral-medium-3-5"},
        "judge": "tllm/foo", "synth": "auto/best-fast",
        "members": [{"role": "Fact-Checker", "model": "tllm/bar"}],
    })
    assert cfg["models"]["code"] != "auto/best-coding"       # banned -> default kept
    assert cfg["models"]["general"] == "mistral/mistral-medium-3-5"  # valid -> applied
    assert not cfg["judge"].startswith(("auto/", "tllm/"))
    assert not cfg["synth"].startswith(("auto/", "tllm/"))
    fc = next(m for m in cfg["members"] if m["role"] == "Fact-Checker")
    assert not fc["model"].startswith(("auto/", "tllm/"))
