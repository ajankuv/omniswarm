import os
import tempfile
import pytest
from omniswarm import store


@pytest.fixture()
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store.init_db(path)
    yield path
    os.remove(path)


def test_create_get_update_roundtrip(db_path):
    store.create_job(db_path, "j1", "general", "running")
    assert store.get_job(db_path, "j1")["status"] == "running"
    store.update_job(db_path, "j1", status="done", verdict="pass",
                     confidence="high", result="hi", tokens_saved=12)
    row = store.get_job(db_path, "j1")
    assert row["status"] == "done"
    assert row["result"] == "hi"
    assert row["tokens_saved"] == 12


def test_events_and_savings(db_path):
    store.create_job(db_path, "j1", "general", "running")
    store.create_job(db_path, "j2", "general", "running")
    store.update_job(db_path, "j1", tokens_saved=10)
    store.update_job(db_path, "j2", tokens_saved=5)
    store.add_event(db_path, "j1", "generate", "mistral/mistral-large-latest", "abc")
    assert store.total_tokens_saved(db_path) == 15
    assert len(store.list_jobs(db_path)) == 2


def test_update_rejects_unknown_column(db_path):
    store.create_job(db_path, "j1", "general", "running")
    with pytest.raises(KeyError):
        store.update_job(db_path, "j1", evil="DROP")


def test_stats_aggregates(db_path):
    store.create_job(db_path, "j1", "general", "done")
    store.update_job(db_path, "j1", status="done", verdict="pass", tokens_saved=10)
    store.create_job(db_path, "j2", "summarize", "escalated")
    store.update_job(db_path, "j2", status="escalated", verdict="escalated", tokens_saved=4)
    s = store.stats(db_path)
    assert s["total_jobs"] == 2
    assert s["tokens_saved"] == 14
    assert s["by_status"]["done"] == 1
    assert s["by_status"]["escalated"] == 1
    assert s["by_verdict"]["pass"] == 1


import sqlite3


def test_new_columns_exist_and_updatable(db_path):
    store.create_job(db_path, "j1", "general", "running")
    store.update_job(db_path, "j1", input="do the thing", models_used='["a","b"]')
    row = store.get_job(db_path, "j1")
    assert row["input"] == "do the thing"
    assert row["models_used"] == '["a","b"]'


def test_init_db_migrates_old_schema(tmp_path):
    p = str(tmp_path / "old.db")
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, task_type TEXT, status TEXT, "
        "created_at REAL, updated_at REAL)"
    )
    con.commit()
    con.close()
    store.init_db(p)  # must add the missing columns, not fail
    cols = {r[1] for r in sqlite3.connect(p).execute("PRAGMA table_info(jobs)")}
    assert "input" in cols and "models_used" in cols


def test_provenance_column_and_enriched_stats(db_path):
    import json as _j
    store.create_job(db_path, "j1", "code", "done")
    store.update_job(db_path, "j1", status="done", verdict="pass", confidence="high",
                     tokens_saved=10, models_used='["a/m1","b/m2"]',
                     provenance=_j.dumps([{"stage": "draft"}, {"stage": "council"}]))
    store.create_job(db_path, "j2", "code", "done")
    store.update_job(db_path, "j2", status="done", verdict="pass", confidence="medium",
                     tokens_saved=20, models_used='["a/m1"]', provenance="[]")

    row = store.get_job(db_path, "j1")
    assert "provenance" in row
    s = store.stats(db_path)
    assert s["total_jobs"] == 2
    assert s["by_task_type"]["code"] == 2
    assert s["by_confidence"]["high"] == 1 and s["by_confidence"]["medium"] == 1
    assert s["by_model"]["a/m1"] == 2 and s["by_model"]["b/m2"] == 1
    assert s["most_used_model"] == "a/m1"
    assert s["total_model_calls"] == 3
    assert s["council_engaged"] == 1  # j1 has a "council" stage
    assert s["avg_tokens_saved"] == 15.0


def test_list_jobs_filters(db_path):
    store.create_job(db_path, "a", "code", "done")
    store.update_job(db_path, "a", status="done", verdict="pass", input="alpha widget")
    store.create_job(db_path, "b", "general", "escalated")
    store.update_job(db_path, "b", status="escalated", verdict="escalated", input="beta gadget")
    assert {j["id"] for j in store.list_jobs(db_path, status="escalated")} == {"b"}
    assert {j["id"] for j in store.list_jobs(db_path, verdict="pass")} == {"a"}
    assert {j["id"] for j in store.list_jobs(db_path, task_type="code")} == {"a"}
    assert {j["id"] for j in store.list_jobs(db_path, q="gadget")} == {"b"}
    assert len(store.list_jobs(db_path)) == 2  # no filters = all


def test_model_calls_and_reliability(db_path):
    store.record_model_call(db_path, "a/m1", True, 120.0, "ok")
    store.record_model_call(db_path, "a/m1", False, 50.0, "HTTP 429")
    store.record_model_call(db_path, "b/m2", True, 80.0, "ok")
    rel = store.model_reliability(db_path)
    assert rel["a/m1"]["calls"] == 2
    assert rel["a/m1"]["ok"] == 1
    assert rel["a/m1"]["http_429"] == 1
    assert 0 <= rel["a/m1"]["success_pct"] <= 100
    assert rel["b/m2"]["success_pct"] == 100.0
    assert rel["a/m1"]["avg_latency_ms"] > 0


def test_schedules_crud_and_due(db_path):
    sid = store.create_schedule(db_path, "summarize the logs", "summarize", 3600)
    rows = store.list_schedules(db_path)
    assert len(rows) == 1 and rows[0]["id"] == sid
    assert rows[0]["interval_seconds"] == 3600 and rows[0]["enabled"] == 1
    assert rows[0]["next_run"] > 0
    # not due yet (next_run ~ now+3600)
    assert store.due_schedules(db_path, __import__("time").time()) == []
    # force due: pretend far future
    due = store.due_schedules(db_path, __import__("time").time() + 4000)
    assert len(due) == 1 and due[0]["id"] == sid
    # due_schedules advanced next_run, so immediately re-checking returns nothing
    assert store.due_schedules(db_path, __import__("time").time() + 4000) == [] or \
        store.list_schedules(db_path)[0]["next_run"] > __import__("time").time() + 4000
    # disable + delete
    store.set_schedule_enabled(db_path, sid, False)
    assert store.list_schedules(db_path)[0]["enabled"] == 0
    store.delete_schedule(db_path, sid)
    assert store.list_schedules(db_path) == []


def test_record_and_latest_benchmarks(tmp_path):
    db = str(tmp_path / "b.db")
    store.init_db(db)
    store.record_benchmark(db, {"model": "m/a", "task_type": "code", "quality": 0.5,
                                "pass_rate": 0.5, "avg_latency_ms": 900.0, "samples": 8, "failures": 0})
    store.record_benchmark(db, {"model": "m/a", "task_type": "code", "quality": 0.9,
                                "pass_rate": 0.9, "avg_latency_ms": 800.0, "samples": 8, "failures": 0})
    latest = store.latest_benchmarks(db)
    row = latest[("m/a", "code")]
    assert row["quality"] == 0.9              # most recent wins
    assert row["samples"] == 8 and "ts" in row
