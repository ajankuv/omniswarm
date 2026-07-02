from omniswarm import modelconfig


def test_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNISWARM_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # no omniswarm.toml here
    cfg = modelconfig.load_model_config()
    assert cfg["models"]["code"] == "mistral/devstral-latest"
    assert cfg["judge"] == "nvidia/meta/llama-4-maverick-17b-128e-instruct"
    assert len(cfg["council_generators"]) >= 2


def test_toml_overrides_merge_over_defaults(tmp_path):
    p = tmp_path / "omniswarm.toml"
    p.write_text(
        '[models]\n'
        'code = "myprovider/my-coder"\n'
        '[judge]\n'
        'model = "myprovider/my-judge"\n'
        '[council]\n'
        'synth_model = "myprovider/my-synth"\n'
        'generators = ["a/one", "b/two"]\n'
    )
    cfg = modelconfig.load_model_config(str(p))
    assert cfg["models"]["code"] == "myprovider/my-coder"     # overridden
    assert cfg["models"]["general"] == "nvidia/meta/llama-4-maverick-17b-128e-instruct"  # default preserved
    assert cfg["judge"] == "myprovider/my-judge"
    assert cfg["synth"] == "myprovider/my-synth"
    assert cfg["council_generators"] == ["a/one", "b/two"]


def test_missing_path_falls_back_to_defaults(tmp_path):
    cfg = modelconfig.load_model_config(str(tmp_path / "nope.toml"))
    assert cfg["models"]["general"] == "nvidia/meta/llama-4-maverick-17b-128e-instruct"


def test_default_members_present():
    cfg = modelconfig.load_model_config(None)
    roles = [m["role"] for m in cfg["members"]]
    assert "Security Engineer" in roles
    assert "Fact-Checker" in roles
    assert "User Advocate" in roles
    assert len(roles) >= 7
    # default roster is a small curated subset of the library
    assert set(cfg["default_roster"]).issubset(set(roles))
    assert len(cfg["default_roster"]) == 3
    # per-task overrides seeded for code + reasoning
    assert "code" in cfg["roster_overrides"]


def test_toml_members_and_overrides(tmp_path):
    p = tmp_path / "omniswarm.toml"
    p.write_text(
        '[[council.members]]\n'
        'role = "Editor"\n'
        'model = "x/editor"\n'
        'focus = "grammar and tone"\n\n'
        '[[council.members]]\n'
        'role = "Fact Checker"\n'
        'model = "y/facts"\n'
        'focus = "accuracy"\n\n'
        '[council.roster_overrides]\n'
        'draft = ["Editor"]\n'
    )
    cfg = modelconfig.load_model_config(str(p))
    roles = [m["role"] for m in cfg["members"]]
    assert roles == ["Editor", "Fact Checker"]            # members replaced
    assert cfg["default_roster"] == ["Editor", "Fact Checker"]  # defaults to all when not set
    assert cfg["roster_overrides"]["draft"] == ["Editor"]
