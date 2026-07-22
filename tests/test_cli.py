import json
import pytest
from omniswarm import cli
from omniswarm.config import Settings


def test_summarize_counts():
    results = [
        {"verdict": "pass", "status": "done", "tokens_saved": 10},
        {"verdict": "escalated", "status": "escalated", "tokens_saved": 5},
        {"verdict": "failed", "status": "failed", "tokens_saved": 0},
        {"verdict": "pass", "status": "done", "tokens_saved": 7},
    ]
    s = cli.summarize(results)
    assert s == {"processed": 4, "passed": 2, "escalated": 1, "failed": 1, "tokens_saved": 22}


@pytest.mark.asyncio
async def test_run_batch_writes_jsonl(monkeypatch, tmp_path):
    async def fake_process(client, settings, req):
        return {"job_id": "j", "text": f"out:{req.user}", "verdict": "pass",
                "confidence": "high", "models_used": ["m"], "tokens_saved": 3, "status": "done"}
    monkeypatch.setattr(cli.store, "init_db", lambda p: None)
    monkeypatch.setattr(cli.engine, "process_job", fake_process)

    out = tmp_path / "results.jsonl"
    settings = Settings("http://x/v1", str(tmp_path / "db.sqlite"), 3, 60, 20, False, 0.0, "high", 5000, 5.0)
    results = await cli.run_batch(["a", "b", "c"], "summarize", str(out), 2, settings)

    assert len(results) == 3
    assert all("task" in r for r in results)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(l) for l in lines]
    assert {p["task"] for p in parsed} == {"a", "b", "c"}


def test_main_submit_end_to_end(monkeypatch, tmp_path, capsys):
    async def fake_process(client, settings, req):
        return {"job_id": "j", "text": "ok", "verdict": "pass",
                "confidence": "high", "models_used": ["m"], "tokens_saved": 4, "status": "done"}
    monkeypatch.setattr(cli.store, "init_db", lambda p: None)
    monkeypatch.setattr(cli.engine, "process_job", fake_process)

    inp = tmp_path / "tasks.txt"
    inp.write_text("first task\nsecond task\n\n")  # blank line ignored
    out = tmp_path / "out.jsonl"

    rc = cli.main(["submit", str(inp), "--out", str(out), "--concurrency", "2"])
    assert rc == 0
    assert len(out.read_text().strip().splitlines()) == 2
    printed = capsys.readouterr().out
    assert "processed 2" in printed
    assert "tokens saved" in printed
