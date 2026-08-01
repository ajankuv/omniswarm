import json
import sqlite3
import time
from omniswarm.util import new_id

_ALLOWED = {"status", "verdict", "confidence", "result", "tokens_saved", "note",
            "input", "models_used", "provenance", "feedback", "cache_key"}

# Marker written on jobs answered from the verified cache. Single-sourced because
# calibration must EXCLUDE these: a cache hit replays a prior council verdict
# without re-running the council, so counting it would multiply one decision into
# many data points and corrupt the track-record.
CACHE_HIT_NOTE = "served from verified cache"


def _connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=5)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=5000")
    return con


def init_db(path: str) -> None:
    con = _connect(path)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                verdict TEXT,
                confidence TEXT,
                result TEXT,
                tokens_saved INTEGER DEFAULT 0,
                note TEXT,
                input TEXT,
                models_used TEXT,
                provenance TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                model TEXT,
                snippet TEXT,
                ts REAL NOT NULL
            )"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS model_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                ok INTEGER NOT NULL,
                status TEXT,
                latency_ms REAL,
                ts REAL NOT NULL
            )"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS schedules (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                task_type TEXT NOT NULL,
                interval_seconds INTEGER NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                next_run REAL NOT NULL,
                last_run REAL,
                created_at REAL NOT NULL
            )"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS benchmarks (
                id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                quality REAL,
                pass_rate REAL,
                avg_latency_ms REAL,
                samples INTEGER,
                failures INTEGER,
                ts REAL NOT NULL
            )"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS answer_cache (
                key TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                result TEXT,
                verdict TEXT,
                confidence TEXT,
                models_used TEXT,
                hits INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            )"""
        )
        existing = {r["name"] for r in con.execute("PRAGMA table_info(jobs)")}
        for col in ("input", "models_used", "provenance", "feedback", "cache_key"):
            if col not in existing:
                con.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
        con.commit()
    finally:
        con.close()


def create_job(path: str, job_id: str, task_type: str, status: str) -> None:
    now = time.time()
    con = _connect(path)
    try:
        con.execute(
            "INSERT INTO jobs (id, task_type, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, task_type, status, now, now),
        )
        con.commit()
    finally:
        con.close()


def update_job(path: str, job_id: str, **fields) -> None:
    bad = set(fields) - _ALLOWED
    if bad:
        raise KeyError(f"unknown job column(s): {bad}")
    cols = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [time.time(), job_id]
    con = _connect(path)
    try:
        con.execute(f"UPDATE jobs SET {cols}, updated_at=? WHERE id=?", values)
        con.commit()
    finally:
        con.close()


def get_job(path: str, job_id: str) -> dict | None:
    con = _connect(path)
    try:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def list_jobs(path: str, limit: int = 50, status: str | None = None,
              verdict: str | None = None, task_type: str | None = None,
              q: str | None = None) -> list[dict]:
    # coerce+clamp: a non-int limit crashes SQLite (datatype mismatch), and a
    # negative limit means "unbounded" in SQLite, which could return the whole table.
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    limit = max(0, min(limit, 10000))
    clauses: list[str] = []
    params: list = []
    if status:
        clauses.append("status = ?"); params.append(status)
    if verdict:
        clauses.append("verdict = ?"); params.append(verdict)
    if task_type:
        clauses.append("task_type = ?"); params.append(task_type)
    if q:
        clauses.append("(COALESCE(input,'') LIKE ? OR COALESCE(result,'') LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    con = _connect(path)
    try:
        rows = con.execute(
            f"SELECT * FROM jobs{where} ORDER BY created_at DESC LIMIT ?", params
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def add_event(path: str, job_id: str, stage: str, model: str, snippet: str) -> None:
    con = _connect(path)
    try:
        con.execute(
            "INSERT INTO events (job_id, stage, model, snippet, ts) VALUES (?, ?, ?, ?, ?)",
            (job_id, stage, model, snippet[:500], time.time()),
        )
        con.commit()
    finally:
        con.close()


def total_tokens_saved(path: str) -> int:
    con = _connect(path)
    try:
        row = con.execute("SELECT COALESCE(SUM(tokens_saved), 0) AS t FROM jobs").fetchone()
        return int(row["t"])
    finally:
        con.close()


# --- Feature A: verdict feedback + calibration -----------------------------

def set_feedback(path: str, job_id: str, value: str) -> bool:
    """Record human/agent feedback on a finished job: "up" (correct) or "down"
    (wrong). A "down" also evicts the job's cached answer, if any, so a bad
    verified-cache entry cannot keep being served. Returns False if no such job."""
    if value not in ("up", "down"):
        raise ValueError(f"feedback must be 'up' or 'down', got {value!r}")
    con = _connect(path)
    try:
        row = con.execute("SELECT cache_key FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return False
        con.execute("UPDATE jobs SET feedback=?, updated_at=? WHERE id=?",
                    (value, time.time(), job_id))
        if value == "down" and row["cache_key"]:
            con.execute("DELETE FROM answer_cache WHERE key=?", (row["cache_key"],))
        con.commit()
        return True
    finally:
        con.close()


def calibration(path: str) -> dict:
    """For jobs that carry feedback, bucket by the confidence claimed at
    issue-time and report how often that confidence was actually correct.
    This is the trust track-record: does "high" really mean high?

    Cache-hit jobs are excluded: they replay a council verdict that was already
    counted when it was first issued, so including them would let one decision
    contribute arbitrarily many data points. Feedback on a cache hit still
    matters — it evicts the entry (see set_feedback)."""
    buckets: dict[str, dict] = {}
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT COALESCE(confidence,'none') AS c, feedback, COUNT(*) AS n "
            "FROM jobs WHERE feedback IS NOT NULL AND COALESCE(note,'') != ? "
            "GROUP BY c, feedback", (CACHE_HIT_NOTE,)
        ).fetchall()
        for r in rows:
            b = buckets.setdefault(r["c"], {"up": 0, "down": 0})
            b[r["feedback"]] = r["n"]
        out = {}
        total_up = total = 0
        for c, b in buckets.items():
            n = b["up"] + b["down"]
            out[c] = {"correct": b["up"], "wrong": b["down"], "total": n,
                      "pct_correct": round(100 * b["up"] / n, 1) if n else 0.0}
            total_up += b["up"]; total += n
        return {"by_confidence": out, "total_rated": total,
                "overall_pct": round(100 * total_up / total, 1) if total else 0.0}
    finally:
        con.close()


# --- Feature B: verified-answer cache --------------------------------------

def cache_get(path: str, key: str, ttl_seconds: float) -> dict | None:
    """Return a fresh cached answer for this key (and bump its hit count), or
    None if absent or older than ttl_seconds."""
    con = _connect(path)
    try:
        row = con.execute("SELECT * FROM answer_cache WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if ttl_seconds and (time.time() - row["created_at"]) > ttl_seconds:
            return None
        con.execute("UPDATE answer_cache SET hits=hits+1 WHERE key=?", (key,))
        con.commit()
        return dict(row)
    finally:
        con.close()


def cache_prune(path: str, ttl_seconds: float = 0.0, max_entries: int = 0) -> int:
    """Keep the cache bounded. Drops entries past their TTL, then — if still over
    max_entries — evicts the least-useful ones. Eviction ranks by hit count first
    so a hot entry is not thrown away just for being old. Returns rows deleted.

    Without this the table only ever grows: expired rows are skipped on read but
    never removed, so every unique pass+high prompt costs disk forever."""
    deleted = 0
    con = _connect(path)
    try:
        if ttl_seconds and ttl_seconds > 0:
            cur = con.execute("DELETE FROM answer_cache WHERE created_at < ?",
                              (time.time() - ttl_seconds,))
            deleted += cur.rowcount or 0
        if max_entries and max_entries > 0:
            n = con.execute("SELECT COUNT(*) AS c FROM answer_cache").fetchone()["c"]
            if n > max_entries:
                # SQLite has no DELETE ... LIMIT, so select the keepers by subquery
                cur = con.execute(
                    "DELETE FROM answer_cache WHERE key NOT IN "
                    "(SELECT key FROM answer_cache ORDER BY hits DESC, created_at DESC LIMIT ?)",
                    (max_entries,),
                )
                deleted += cur.rowcount or 0
        con.commit()
        return deleted
    finally:
        con.close()


def cache_put(path: str, key: str, task_type: str, result: str,
              verdict: str, confidence: str, models_used: str,
              ttl_seconds: float = 0.0, max_entries: int = 0) -> None:
    con = _connect(path)
    try:
        con.execute(
            "INSERT INTO answer_cache (key, task_type, result, verdict, confidence, "
            "models_used, hits, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?) "
            "ON CONFLICT(key) DO UPDATE SET result=excluded.result, verdict=excluded.verdict, "
            "confidence=excluded.confidence, models_used=excluded.models_used, "
            "created_at=excluded.created_at",
            (key, task_type, result, verdict, confidence, models_used, time.time()),
        )
        con.commit()
    finally:
        con.close()
    if ttl_seconds or max_entries:
        cache_prune(path, ttl_seconds, max_entries)


def cache_clear(path: str) -> int:
    """Drop every cached answer. Called when the privacy mode tightens — cached
    plaintext must not outlive the setting that allowed it to be stored."""
    con = _connect(path)
    try:
        n = con.execute("SELECT COUNT(*) AS c FROM answer_cache").fetchone()["c"]
        con.execute("DELETE FROM answer_cache")
        con.commit()
        return int(n)
    finally:
        con.close()


def cache_stats(path: str) -> dict:
    con = _connect(path)
    try:
        row = con.execute(
            "SELECT COUNT(*) AS entries, COALESCE(SUM(hits),0) AS hits FROM answer_cache"
        ).fetchone()
        return {"entries": int(row["entries"]), "hits": int(row["hits"])}
    finally:
        con.close()


def record_model_call(path: str, model: str, ok: bool, latency_ms: float, status: str) -> None:
    con = _connect(path)
    try:
        con.execute(
            "INSERT INTO model_calls (model, ok, status, latency_ms, ts) VALUES (?, ?, ?, ?, ?)",
            (model, 1 if ok else 0, status, float(latency_ms), time.time()),
        )
        con.commit()
    finally:
        con.close()


def model_reliability(path: str) -> dict:
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT model, COUNT(*) AS calls, SUM(ok) AS oks, AVG(latency_ms) AS lat, "
            "SUM(CASE WHEN status='HTTP 429' THEN 1 ELSE 0 END) AS r429 "
            "FROM model_calls GROUP BY model"
        ).fetchall()
        out = {}
        for r in rows:
            calls = int(r["calls"]); oks = int(r["oks"] or 0)
            out[r["model"]] = {
                "calls": calls,
                "ok": oks,
                "fail": calls - oks,
                "success_pct": round(100.0 * oks / calls, 1) if calls else 0.0,
                "avg_latency_ms": round(float(r["lat"] or 0.0), 1),
                "http_429": int(r["r429"] or 0),
            }
        return out
    finally:
        con.close()


def record_benchmark(path: str, result: dict) -> None:
    con = _connect(path)
    try:
        con.execute(
            "INSERT INTO benchmarks (id, model, task_type, quality, pass_rate, avg_latency_ms, "
            "samples, failures, ts) VALUES (?,?,?,?,?,?,?,?,?)",
            (new_id(), result["model"], result["task_type"], result.get("quality"),
             result.get("pass_rate"), result.get("avg_latency_ms"), result.get("samples"),
             result.get("failures"), time.time()),
        )
        con.commit()
    finally:
        con.close()


def latest_benchmarks(path: str) -> dict:
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT model, task_type, quality, pass_rate, avg_latency_ms, samples, failures, ts "
            "FROM benchmarks ORDER BY ts ASC"
        ).fetchall()
        out = {}
        for r in rows:  # ascending ts -> later rows overwrite -> most recent wins
            out[(r["model"], r["task_type"])] = {
                "quality": r["quality"], "pass_rate": r["pass_rate"],
                "avg_latency_ms": r["avg_latency_ms"], "samples": r["samples"],
                "failures": r["failures"], "ts": r["ts"],
            }
        return out
    finally:
        con.close()


def stats(path: str, usd_per_mtok: float = 0.0) -> dict:
    con = _connect(path)
    try:
        total = con.execute(
            "SELECT COUNT(*) AS c, COALESCE(SUM(tokens_saved), 0) AS t FROM jobs"
        ).fetchone()
        n = int(total["c"])
        saved = int(total["t"])

        def group(col):
            return {
                (r[col] if r[col] is not None else "none"): r["c"]
                for r in con.execute(f"SELECT {col}, COUNT(*) AS c FROM jobs GROUP BY {col}")
            }

        by_status = group("status")
        by_verdict = group("verdict")
        by_task_type = group("task_type")
        by_confidence = group("confidence")

        by_model: dict[str, int] = {}
        total_calls = 0
        for r in con.execute("SELECT models_used FROM jobs WHERE models_used IS NOT NULL"):
            try:
                for m in json.loads(r["models_used"]):
                    by_model[m] = by_model.get(m, 0) + 1
                    total_calls += 1
            except (TypeError, json.JSONDecodeError):
                continue

        council_engaged = 0
        for r in con.execute("SELECT provenance FROM jobs WHERE provenance IS NOT NULL"):
            try:
                if any(s.get("stage") in ("council", "synthesis") for s in json.loads(r["provenance"])):
                    council_engaged += 1
            except (TypeError, json.JSONDecodeError, AttributeError):
                continue

        most_used_model = max(by_model, key=by_model.get) if by_model else None
        avg_tokens_saved = round(saved / n, 1) if n else 0.0

        return {
            "total_jobs": n,
            "tokens_saved": saved,
            "dollars_saved": round(saved / 1_000_000 * usd_per_mtok, 2),
            "usd_per_mtok": usd_per_mtok,
            "avg_tokens_saved": avg_tokens_saved,
            "total_model_calls": total_calls,
            "most_used_model": most_used_model,
            "council_engaged": council_engaged,
            "by_status": by_status,
            "by_verdict": by_verdict,
            "by_task_type": by_task_type,
            "by_confidence": by_confidence,
            "by_model": by_model,
        }
    finally:
        con.close()


def create_schedule(path: str, prompt: str, task_type: str, interval_seconds: int) -> str:
    sid = new_id()
    now = time.time()
    con = _connect(path)
    try:
        con.execute(
            "INSERT INTO schedules (id, prompt, task_type, interval_seconds, enabled, next_run, "
            "last_run, created_at) VALUES (?, ?, ?, ?, 1, ?, NULL, ?)",
            (sid, prompt, task_type, int(interval_seconds), now + int(interval_seconds), now),
        )
        con.commit()
    finally:
        con.close()
    return sid


def list_schedules(path: str) -> list[dict]:
    con = _connect(path)
    try:
        rows = con.execute("SELECT * FROM schedules ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def delete_schedule(path: str, sid: str) -> None:
    con = _connect(path)
    try:
        con.execute("DELETE FROM schedules WHERE id=?", (sid,))
        con.commit()
    finally:
        con.close()


def set_schedule_enabled(path: str, sid: str, enabled: bool) -> None:
    con = _connect(path)
    try:
        con.execute("UPDATE schedules SET enabled=? WHERE id=?", (1 if enabled else 0, sid))
        con.commit()
    finally:
        con.close()


def due_schedules(path: str, now: float) -> list[dict]:
    """Return enabled schedules whose next_run <= now, and advance their next_run."""
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT * FROM schedules WHERE enabled=1 AND next_run<=?", (now,)
        ).fetchall()
        due = [dict(r) for r in rows]
        for r in due:
            con.execute(
                "UPDATE schedules SET next_run=?, last_run=? WHERE id=?",
                (now + int(r["interval_seconds"]), now, r["id"]),
            )
        con.commit()
        return due
    finally:
        con.close()
