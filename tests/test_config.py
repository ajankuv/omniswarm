import os
from omniswarm.config import get_settings


def test_defaults_point_at_omniroute_lan():
    s = get_settings()
    assert s.omniroute_base_url == "http://localhost:20128/v1"
    assert s.db_path == "omniswarm.db"
    assert s.read_timeout >= 30  # LLM calls are slow


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OMNISWARM_DB_PATH", "/tmp/custom.db")
    monkeypatch.setenv("OMNISWARM_OMNIROUTE_URL", "http://example/v1")
    s = get_settings()
    assert s.db_path == "/tmp/custom.db"
    assert s.omniroute_base_url == "http://example/v1"
