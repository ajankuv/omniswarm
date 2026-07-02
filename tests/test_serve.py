from omniswarm import serve


def test_serve_main_is_callable():
    assert callable(serve.main)


def test_serve_reads_host_port_env(monkeypatch):
    captured = {}

    def fake_run(app, host, port):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    import types
    fake_uvicorn = types.SimpleNamespace(run=fake_run)
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", fake_uvicorn)
    monkeypatch.setenv("OMNISWARM_HOST", "127.0.0.1")
    monkeypatch.setenv("OMNISWARM_PORT", "9123")
    serve.main()
    assert captured["app"] == "omniswarm.app:app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9123
