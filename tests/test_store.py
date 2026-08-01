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
    # no rate passed -> zero dollars, but the keys are always present
    assert s["dollars_saved"] == 0.0
    assert s["usd_per_mtok"] == 0.0


def test_stats_dollars_saved(db_path):
    store.create_job(db_path, "j1", "general", "done")
    store.update_job(db_path, "j1", tokens_saved=2_000_000)
    store.create_job(db_path, "j2", "general", "done")
    store.update_job(db_path, "j2", tokens_saved=500_000)
    # 2,500,000 tokens @ $5/Mtok = $12.50
    s = store.stats(db_path, usd_per_mtok=5.0)
    assert s["tokens_saved"] == 2_500_000
    assert s["dollars_saved"] == 12.50
    assert s["usd_per_mtok"] == 5.0


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


# --- Feature A: feedback + calibration -------------------------------------

def test_set_feedback_and_calibration(db_path):
    # two high-confidence jobs, one right one wrong; one medium right
    for jid, conf in [("j1", "high"), ("j2", "high"), ("j3", "medium")]:
        store.create_job(db_path, jid, "general", "done")
        store.update_job(db_path, jid, status="done", verdict="pass", confidence=conf)
    assert store.set_feedback(db_path, "j1", "up") is True
    assert store.set_feedback(db_path, "j2", "down") is True
    assert store.set_feedback(db_path, "j3", "up") is True
    assert store.set_feedback(db_path, "missing", "up") is False

    cal = store.calibration(db_path)
    assert cal["by_confidence"]["high"] == {"correct": 1, "wrong": 1, "total": 2, "pct_correct": 50.0}
    assert cal["by_confidence"]["medium"]["pct_correct"] == 100.0
    assert cal["total_rated"] == 3
    assert cal["overall_pct"] == round(100 * 2 / 3, 1)


def test_set_feedback_rejects_bad_value(db_path):
    store.create_job(db_path, "j1", "general", "done")
    with pytest.raises(ValueError):
        store.set_feedback(db_path, "j1", "meh")


# --- Feature B: verified-answer cache --------------------------------------

def test_cache_put_get_and_ttl(db_path):
    store.cache_put(db_path, "k1", "general", "Paris", "pass", "high", '["m1"]')
    hit = store.cache_get(db_path, "k1", ttl_seconds=3600)
    assert hit["result"] == "Paris" and hit["verdict"] == "pass"
    # a second read bumps hits
    store.cache_get(db_path, "k1", ttl_seconds=3600)
    assert store.cache_stats(db_path) == {"entries": 1, "hits": 2}
    # expired by a tiny ttl -> miss
    assert store.cache_get(db_path, "k1", ttl_seconds=-1) is None
    assert store.cache_get(db_path, "nope", ttl_seconds=3600) is None


def test_down_feedback_evicts_cached_answer(db_path):
    # a job whose answer was cached under cache_key "k1"
    store.cache_put(db_path, "k1", "general", "Paris", "pass", "high", '["m1"]')
    store.create_job(db_path, "j1", "general", "done")
    store.update_job(db_path, "j1", status="done", verdict="pass", confidence="high", cache_key="k1")
    assert store.cache_get(db_path, "k1", ttl_seconds=3600) is not None
    store.set_feedback(db_path, "j1", "down")   # thumbs-down evicts the cache entry
    assert store.cache_get(db_path, "k1", ttl_seconds=3600) is None
    # thumbs-up must NOT evict
    store.cache_put(db_path, "k1", "general", "Paris", "pass", "high", '["m1"]')
    store.set_feedback(db_path, "j1", "up")
    assert store.cache_get(db_path, "k1", ttl_seconds=3600) is not None


def test_calibration_excludes_cache_hits(db_path):
    """A cache hit replays an already-counted verdict — it must not multiply one
    council decision into many calibration data points."""
    # one real council job, rated correct
    store.create_job(db_path, "real", "general", "done")
    store.update_job(db_path, "real", status="done", verdict="pass", confidence="high")
    store.set_feedback(db_path, "real", "up")
    # three cache-hit replays of that same answer, all rated wrong
    for i in range(3):
        jid = f"hit{i}"
        store.create_job(db_path, jid, "general", "done")
        store.update_job(db_path, jid, status="done", verdict="pass", confidence="high",
                         note=store.CACHE_HIT_NOTE)
        store.set_feedback(db_path, jid, "down")
    cal = store.calibration(db_path)
    # only the genuine council verdict counts
    assert cal["total_rated"] == 1
    assert cal["by_confidence"]["high"] == {"correct": 1, "wrong": 0, "total": 1, "pct_correct": 100.0}


def test_cache_clear_drops_everything(db_path):
    store.cache_put(db_path, "k1", "general", "a", "pass", "high", '["m"]')
    store.cache_put(db_path, "k2", "general", "b", "pass", "high", '["m"]')
    assert store.cache_stats(db_path)["entries"] == 2
    assert store.cache_clear(db_path) == 2
    assert store.cache_stats(db_path)["entries"] == 0
    assert store.cache_get(db_path, "k1", 3600) is None


def test_cache_prune_drops_expired_and_caps_size(db_path):
    import time as _t
    # an entry older than the TTL is purged
    store.cache_put(db_path, "old", "general", "a", "pass", "high", '["m"]')
    con = store._connect(db_path)
    con.execute("UPDATE answer_cache SET created_at=? WHERE key='old'", (_t.time() - 10_000,))
    con.commit(); con.close()
    assert store.cache_prune(db_path, ttl_seconds=100) == 1
    assert store.cache_stats(db_path)["entries"] == 0

    # cap keeps the HOTTEST entries, not merely the newest
    for i in range(5):
        store.cache_put(db_path, f"k{i}", "general", "x", "pass", "high", '["m"]')
    con = store._connect(db_path)
    con.execute("UPDATE answer_cache SET hits=99 WHERE key='k0'")   # oldest but hottest
    con.commit(); con.close()
    store.cache_prune(db_path, max_entries=2)
    keys = {r["key"] for r in store._connect(db_path).execute("SELECT key FROM answer_cache")}
    assert "k0" in keys and len(keys) == 2      # hot entry survives despite being oldest


def test_cache_put_self_prunes_to_cap(db_path):
    for i in range(10):
        store.cache_put(db_path, f"k{i}", "general", "x", "pass", "high", '["m"]',
                        ttl_seconds=0, max_entries=3)
    assert store.cache_stats(db_path)["entries"] == 3     # never grows past the cap


def test_list_jobs_tolerates_bad_limit(db_path):
    store.create_job(db_path, "j1", "general", "done")
    # non-int limits used to crash SQLite with a datatype mismatch
    for bad in ("abc", 1.7, None, -1):
        rows = store.list_jobs(db_path, limit=bad)   # must not raise
        assert isinstance(rows, list)
    assert len(store.list_jobs(db_path, limit=0)) == 0
