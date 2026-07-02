from omniswarm import runtime


def test_defaults_when_no_file(tmp_path):
    r = runtime.load_runtime(str(tmp_path / "none.json"))
    assert r["api_token"] == ""
    assert r["rate_limit_per_min"] == 0
    assert r["store_mode"] == "full"
    assert r["active_roster"] == []


def test_save_then_load_roundtrip(tmp_path):
    p = str(tmp_path / "rt.json")
    runtime.save_runtime(p, {"api_token": "secret", "rate_limit_per_min": 5, "store_mode": "none"})
    r = runtime.load_runtime(p)
    assert r["api_token"] == "secret"
    assert r["rate_limit_per_min"] == 5
    assert r["store_mode"] == "none"


def test_partial_file_merges_defaults(tmp_path):
    p = tmp_path / "rt.json"
    p.write_text('{"store_mode": "redact"}')
    r = runtime.load_runtime(str(p))
    assert r["store_mode"] == "redact"
    assert r["api_token"] == ""        # default
    assert r["rate_limit_per_min"] == 0


def test_active_roster_default_and_persist(tmp_path):
    r = runtime.load_runtime(str(tmp_path / "none.json"))
    assert r["active_roster"] == []
    p = str(tmp_path / "rt.json")
    runtime.save_runtime(p, {"active_roster": ["Fact-Checker", "User Advocate"]})
    assert runtime.load_runtime(p)["active_roster"] == ["Fact-Checker", "User Advocate"]


def test_always_council_default_true(tmp_path):
    r = runtime.load_runtime(str(tmp_path / "none.json"))
    assert r["always_council"] is True
    p = str(tmp_path / "rt.json")
    runtime.save_runtime(p, {"always_council": False})
    assert runtime.load_runtime(p)["always_council"] is False


def test_model_override_keys_default_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNISWARM_RUNTIME", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = runtime.load_runtime()
    assert cfg["models"] == {} and cfg["judge"] == "" and cfg["synth"] == ""
    assert cfg["members"] == []


def test_model_overrides_roundtrip(tmp_path):
    p = tmp_path / "rt.json"
    runtime.save_runtime(str(p), {
        "api_token": "", "rate_limit_per_min": 0, "store_mode": "full",
        "active_roster": [], "always_council": True,
        "models": {"code": "mistral/devstral-latest"}, "judge": "nvidia/x",
        "synth": "mistral/y", "members": [{"role": "Fact-Checker", "model": "mistral/z"}],
    })
    cfg = runtime.load_runtime(str(p))
    assert cfg["models"]["code"] == "mistral/devstral-latest"
    assert cfg["judge"] == "nvidia/x" and cfg["synth"] == "mistral/y"
    assert cfg["members"] == [{"role": "Fact-Checker", "model": "mistral/z"}]


def test_malformed_model_overrides_ignored(tmp_path):
    p = tmp_path / "rt.json"
    p.write_text('{"models": "not-a-dict", "members": [{"bad": 1}, {"role":"R","model":"m/x"}]}')
    cfg = runtime.load_runtime(str(p))
    assert cfg["models"] == {}                       # non-dict ignored
    assert cfg["members"] == [{"role": "R", "model": "m/x"}]  # only well-formed kept
