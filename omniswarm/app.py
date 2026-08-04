import asyncio
import csv
import io
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from omniswarm import benchmarks, catalog, engine, events, failover, recommend, registry, runtime, store
from omniswarm.adapters import OmniRouteError, generate, health
from omniswarm.config import get_settings
from omniswarm.util import new_id


class ChatMessage(BaseModel):
    role: str
    content: str


class OmniSwarmOptions(BaseModel):
    task_type: str = "general"


class ScheduleIn(BaseModel):
    prompt: str
    task_type: str = "general"
    interval_minutes: int = 60


class FeedbackIn(BaseModel):
    correct: bool


class ChatCompletionRequest(BaseModel):
    model: str = "omniswarm"
    messages: list[ChatMessage]
    max_tokens: int = 512
    omniswarm: OmniSwarmOptions = Field(default_factory=OmniSwarmOptions)


_CSV_RISKY = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(v) -> str:
    """Neutralize spreadsheet formula injection (CWE-1236) in exported cells.

    Job input is user-supplied and results are LLM-generated, so a cell can start
    with a formula trigger and execute when the export is opened in Excel/Sheets.
    Checked after stripping leading whitespace, since importers trim it. Genuine
    numbers are left alone so "-5" stays numeric rather than becoming text."""
    s = "" if v is None else str(v)
    if s.lstrip()[:1] not in _CSV_RISKY:
        return s
    try:
        float(s)
        return s          # a plain number, not a formula
    except ValueError:
        return "'" + s


def _last(messages: list[ChatMessage], role: str, default: str = "") -> str:
    for m in reversed(messages):
        if m.role == role:
            return m.content
    return default


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.init_db(settings.db_path)
        # jobs left "running" from a previous process are orphaned — mark them failed
        for _j in store.list_jobs(settings.db_path, 10000, status="running"):
            store.update_job(settings.db_path, _j["id"], status="failed",
                             note="interrupted by restart")
        timeout = httpx.Timeout(
            connect=settings.connect_timeout, read=settings.read_timeout,
            write=settings.read_timeout, pool=settings.connect_timeout,
        )
        limits = httpx.Limits(max_connections=settings.max_connections)
        app.state.client = httpx.AsyncClient(timeout=timeout, limits=limits)
        app.state.settings = settings
        app.state.runtime = runtime.load_runtime()
        registry.apply_runtime(app.state.runtime)     # apply persisted model overrides live
        app.state.catalog = None
        app.state.catalog_at = 0.0
        from omniswarm import adapters as _adapters
        _db = settings.db_path
        app.state.failover = failover.FailoverTracker()
        app.state.provider_breaker = failover.ProviderBreaker()
        app.state.settings_lock = asyncio.Lock()

        def _sink(model, ok, lat, status):
            store.record_model_call(_db, model, ok, lat, status)
            app.state.provider_breaker.record(model, ok, status)   # T1.2 provider circuit breaker
            if app.state.failover.record(model, ok):
                # sink runs inside the event loop (called from async generate)
                task = asyncio.get_running_loop().create_task(failover.execute(app, model))
                app.state.bg_tasks.add(task)
                task.add_done_callback(app.state.bg_tasks.discard)

        _adapters.set_sink(_sink)
        try:
            app.state.omniroute_ok = await health(app.state.client, settings.omniroute_base_url)
        except Exception:
            app.state.omniroute_ok = False
        app.state.job_sem = asyncio.Semaphore(int(os.environ.get("OMNISWARM_MAX_CONCURRENT_JOBS", "3")))
        app.state.bg_tasks = set()
        app.state.sched_interval = int(os.environ.get("OMNISWARM_SCHED_INTERVAL", "20"))
        app.state.scheduler_task = asyncio.create_task(_scheduler())
        try:
            yield
        finally:
            app.state.scheduler_task.cancel()
            await app.state.client.aclose()

    app = FastAPI(title="OmniSwarm", version="0.1.0", lifespan=lifespan)

    import time as _time
    _rl: dict[str, list[float]] = {}

    def _auth(request: Request):
        token = app.state.runtime.get("api_token") or ""
        if not token:
            return
        supplied = None
        header = request.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            supplied = header[7:]
        supplied = supplied or request.query_params.get("token")
        # constant-time: a plain != leaks the matching prefix length via timing.
        # compared as bytes — compare_digest raises TypeError on non-ASCII str,
        # which would turn a junk header into a 500 instead of a clean 401.
        if not secrets.compare_digest((supplied or "").encode("utf-8"),
                                      token.encode("utf-8")):
            raise HTTPException(status_code=401, detail="invalid or missing API token")

    def _rate_limit(request: Request):
        limit = app.state.runtime.get("rate_limit_per_min") or 0
        if not limit:
            return
        key = request.headers.get("Authorization") or (request.client.host if request.client else "anon")
        now = _time.time()
        times = [t for t in _rl.get(key, []) if now - t < 60]
        if len(times) >= limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        times.append(now)
        _rl[key] = times

    @app.post("/v1/chat/completions", dependencies=[Depends(_auth), Depends(_rate_limit)])
    async def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        job_req = engine.JobRequest(
            task_type=req.omniswarm.task_type,
            system=_last(req.messages, "system", "You are a helpful assistant."),
            user=_last(req.messages, "user"),
            max_tokens=req.max_tokens,
        )
        result = await engine.process_job(app.state.client, app.state.settings, job_req,
                                          store_mode=app.state.runtime.get("store_mode", "full"),
                                          active_roster=app.state.runtime.get("active_roster"),
                                          always_council=app.state.runtime.get("always_council", True))
        return {
            "id": f"omniswarm-{result['job_id']}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "omniswarm",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
            "omniswarm": {
                "job_id": result["job_id"],
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "models_used": result["models_used"],
                "tokens_saved": result["tokens_saved"],
            },
        }

    def _launch_bg(job_req):
        job_id = new_id()

        async def _run():
            async with app.state.job_sem:
                try:
                    await engine.process_job(
                        app.state.client, app.state.settings, job_req,
                        store_mode=app.state.runtime.get("store_mode", "full"),
                        active_roster=app.state.runtime.get("active_roster"),
                        always_council=app.state.runtime.get("always_council", True),
                        job_id=job_id,
                    )
                except Exception:
                    pass

        task = asyncio.create_task(_run())
        app.state.bg_tasks.add(task)
        task.add_done_callback(app.state.bg_tasks.discard)
        return job_id

    async def _scheduler():
        while True:
            try:
                await asyncio.sleep(app.state.sched_interval)
                due = await asyncio.to_thread(store.due_schedules, app.state.settings.db_path, time.time())
                for s in due:
                    _launch_bg(engine.JobRequest(
                        task_type=s["task_type"],
                        system="You are a precise, helpful assistant. Do exactly what is asked.",
                        user=s["prompt"],
                    ))
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    @app.post("/v1/jobs", dependencies=[Depends(_auth), Depends(_rate_limit)])
    async def submit_job(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        job_req = engine.JobRequest(
            task_type=req.omniswarm.task_type,
            system=_last(req.messages, "system", "You are a helpful assistant."),
            user=_last(req.messages, "user"),
            max_tokens=req.max_tokens,
        )
        job_id = _launch_bg(job_req)
        return {"job_id": job_id, "status": "running", "stream": "/stream"}

    @app.get("/schedules", dependencies=[Depends(_auth)])
    async def get_schedules():
        return store.list_schedules(app.state.settings.db_path)

    @app.post("/schedules", dependencies=[Depends(_auth), Depends(_rate_limit)])
    async def add_schedule(body: ScheduleIn):
        if not body.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt required")
        interval = max(60, int(body.interval_minutes) * 60)
        sid = store.create_schedule(app.state.settings.db_path, body.prompt, body.task_type, interval)
        rows = store.list_schedules(app.state.settings.db_path)
        return next((s for s in rows if s["id"] == sid), {"id": sid, "interval_seconds": interval})

    @app.delete("/schedules/{sid}", dependencies=[Depends(_auth)])
    async def remove_schedule(sid: str):
        store.delete_schedule(app.state.settings.db_path, sid)
        return {"ok": True}

    @app.post("/schedules/{sid}/toggle", dependencies=[Depends(_auth)])
    async def toggle_schedule(sid: str, enabled: bool = True):
        store.set_schedule_enabled(app.state.settings.db_path, sid, enabled)
        return {"ok": True, "enabled": enabled}

    @app.get("/jobs", dependencies=[Depends(_auth)])
    async def jobs(limit: int = 50, status: str | None = None, verdict: str | None = None,
                   task_type: str | None = None, q: str | None = None):
        return store.list_jobs(app.state.settings.db_path, limit, status, verdict, task_type, q)

    @app.get("/export", dependencies=[Depends(_auth)])
    async def export(format: str = "json", status: str | None = None, verdict: str | None = None,
                     task_type: str | None = None, q: str | None = None):
        rows = store.list_jobs(app.state.settings.db_path, 10000, status, verdict, task_type, q)
        if format == "csv":
            cols = ["id", "task_type", "status", "verdict", "confidence", "tokens_saved",
                    "created_at", "input", "result", "note", "models_used"]
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(cols)
            for r in rows:
                w.writerow([_csv_safe(r.get(c, "")) for c in cols])
            return Response(content=buf.getvalue(), media_type="text/csv",
                            headers={"Content-Disposition": "attachment; filename=omniswarm-jobs.csv"})
        return JSONResponse(content=rows,
                            headers={"Content-Disposition": "attachment; filename=omniswarm-jobs.json"})

    @app.get("/jobs/{job_id}", dependencies=[Depends(_auth)])
    async def job(job_id: str):
        row = store.get_job(app.state.settings.db_path, job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="job not found")
        return row

    @app.post("/jobs/{job_id}/feedback", dependencies=[Depends(_auth)])
    async def feedback(job_id: str, body: FeedbackIn):
        value = "up" if body.correct else "down"
        ok = store.set_feedback(app.state.settings.db_path, job_id, value)
        if not ok:
            raise HTTPException(status_code=404, detail="job not found")
        return {"ok": True, "feedback": value}

    @app.get("/calibration", dependencies=[Depends(_auth)])
    async def calibration():
        return store.calibration(app.state.settings.db_path)

    @app.get("/healthz")
    async def healthz():
        ok = await health(app.state.client, app.state.settings.omniroute_base_url)
        app.state.omniroute_ok = ok
        return {"ok": ok, "omniroute": ok}

    @app.get("/stream", dependencies=[Depends(_auth)])
    async def stream(request: Request):
        from omniswarm import events
        q = events.subscribe()

        async def gen():
            try:
                yield ": connected\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"data: {json.dumps(ev)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                events.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/stats", dependencies=[Depends(_auth)])
    async def stats():
        s = store.stats(app.state.settings.db_path,
                        app.state.settings.savings_usd_per_mtok)
        s["cache"] = store.cache_stats(app.state.settings.db_path)
        # T1.2/T1.4 — provider breaker state for dashboard visibility + alerts
        br = getattr(app.state, "provider_breaker", None)
        s["providers"] = br.state() if br is not None else {}
        s["exhausted_providers"] = sorted(br.exhausted()) if br is not None else []
        return s

    @app.get("/reliability", dependencies=[Depends(_auth)])
    async def reliability():
        return store.model_reliability(app.state.settings.db_path)

    @app.get("/settings")
    async def get_settings_route():
        rt = app.state.runtime
        from omniswarm import registry
        return {
            "protected": bool(rt.get("api_token")),
            "rate_limit_per_min": rt.get("rate_limit_per_min", 0),
            "store_mode": rt.get("store_mode", "full"),
            "gateway_url": app.state.settings.omniroute_base_url,
            "models": {n: t.model for n, t in registry.REGISTRY.items()},
            "council": [m["role"] for m in registry.COUNCIL_MEMBERS],
            "members": registry.COUNCIL_MEMBERS,
            "active_roster": rt.get("active_roster", []),
            "always_council": rt.get("always_council", True),
            "judge": registry.JUDGE_MODEL,
            "synth": registry.COUNCIL_SYNTH_MODEL,
        }

    @app.post("/settings", dependencies=[Depends(_auth)])
    async def post_settings_route(request: Request):
        body = await request.json()
        rt = dict(app.state.runtime)
        if "api_token" in body and isinstance(body["api_token"], str):
            rt["api_token"] = body["api_token"]
        if "rate_limit_per_min" in body and isinstance(body["rate_limit_per_min"], int):
            rt["rate_limit_per_min"] = max(0, body["rate_limit_per_min"])
        if body.get("store_mode") in ("full", "redact", "none"):
            rt["store_mode"] = body["store_mode"]
            # Cached answers hold plaintext that only "full" mode permits. Tightening
            # the privacy mode must not leave it readable — or servable — afterwards.
            if rt["store_mode"] != "full":
                store.cache_clear(app.state.settings.db_path)
        if isinstance(body.get("active_roster"), list):
            valid = {m["role"] for m in registry.COUNCIL_MEMBERS}
            rt["active_roster"] = [r for r in body["active_roster"] if r in valid]
        if isinstance(body.get("always_council"), bool):
            rt["always_council"] = body["always_council"]
        cat_ids = {m["id"] for m in await catalog.get_cached_catalog(app)}

        def _valid(mid):
            return isinstance(mid, str) and mid in cat_ids and not mid.startswith(("auto/", "tllm/"))

        # Only touch model overrides when we have a catalog to validate against;
        # otherwise an unavailable gateway would silently wipe the model map to defaults.
        if cat_ids:
            if isinstance(body.get("models"), dict):
                # MERGE validated entries into the existing overrides — don't replace.
                # A replace meant a partial update dropped other slots, and an
                # all-invalid dict wiped every override to {} (silent config loss).
                merged = dict(rt.get("models") or {})
                for k, v in body["models"].items():
                    if _valid(v):
                        merged[str(k)] = v
                rt["models"] = merged
            if _valid(body.get("judge")):
                rt["judge"] = body["judge"]
            if _valid(body.get("synth")):
                rt["synth"] = body["synth"]
            if isinstance(body.get("members"), list):
                rt["members"] = [{"role": str(m["role"]), "model": m["model"]}
                                 for m in body["members"]
                                 if isinstance(m, dict) and "role" in m and _valid(m.get("model"))]
        # serialize with auto-failover so concurrent saves can't clobber each other
        async with app.state.settings_lock:
            runtime.save_runtime(None, rt)
            app.state.runtime = rt
            registry.apply_runtime(rt)
        return {"ok": True, "protected": bool(rt.get("api_token")),
                "rate_limit_per_min": rt["rate_limit_per_min"], "store_mode": rt["store_mode"],
                "active_roster": rt.get("active_roster", []),
                "always_council": rt.get("always_council", True),
                "models": {n: t.model for n, t in registry.REGISTRY.items()},
                "judge": registry.JUDGE_MODEL, "synth": registry.COUNCIL_SYNTH_MODEL,
                "members": registry.COUNCIL_MEMBERS}

    @app.get("/models/available")
    async def models_available():
        cat = await catalog.get_cached_catalog(app)
        rel = store.model_reliability(app.state.settings.db_path)
        return {"models": [{**m, "reliability": rel.get(m["id"])} for m in cat]}

    @app.get("/models/recommend")
    async def models_recommend():
        cat = await catalog.get_cached_catalog(app)
        rel = store.model_reliability(app.state.settings.db_path)
        q = store.latest_benchmarks(app.state.settings.db_path)
        out = {slot: recommend.recommend(slot, cat, rel, q) for slot in recommend.SLOT_TASK_TYPES}
        out["judge"] = recommend.recommend("judge", cat, rel, q)
        out["synth"] = recommend.recommend("synth", cat, rel, q)
        out["members"] = {mem["role"]: recommend.recommend("synth", cat, rel, q)
                          for mem in registry.COUNCIL_MEMBERS}
        return out

    @app.get("/models/probe", dependencies=[Depends(_auth)])
    async def models_probe(id: str):
        cat = await catalog.get_cached_catalog(app)
        if id not in {m["id"] for m in cat}:
            raise HTTPException(status_code=400, detail="unknown model id")
        t0 = _time.time()
        try:
            await generate(app.state.client, app.state.settings.omniroute_base_url, id,
                           "Reply with one word.", "ping", max_tokens=5, max_retries=0)
            return {"ok": True, "latency": round(_time.time() - t0, 2), "err": ""}
        except OmniRouteError as e:
            return {"ok": False, "latency": round(_time.time() - t0, 2), "err": str(e)[:120]}

    class BenchmarkIn(BaseModel):
        task_type: str = "general"
        candidates: list[str] = []

    @app.post("/benchmark", dependencies=[Depends(_auth)])
    async def start_benchmark(body: BenchmarkIn):
        if body.task_type not in benchmarks.OBJECTIVE_BANKS and body.task_type not in benchmarks.RUBRIC_BANKS:
            raise HTTPException(status_code=400, detail="unknown task_type")
        cat_ids = {m["id"] for m in await catalog.get_cached_catalog(app)}
        cands = [c for c in body.candidates
                 if isinstance(c, str) and c in cat_ids and not c.startswith(("auto/", "tllm/"))]
        if not cands:
            raise HTTPException(status_code=400, detail="no valid candidate models")
        bid = new_id()

        async def _run():
            async with app.state.job_sem:
                def _progress(ev):
                    events.publish({"type": "benchmark", "benchmark_id": bid,
                                    "task_type": body.task_type, **ev})
                try:
                    results = await benchmarks.run_benchmark(
                        app.state.client, app.state.settings.omniroute_base_url,
                        body.task_type, cands, on_progress=_progress)
                    for res in results:
                        await asyncio.to_thread(store.record_benchmark, app.state.settings.db_path, res)
                    events.publish({"type": "benchmark", "benchmark_id": bid,
                                    "task_type": body.task_type, "done": len(cands),
                                    "total": len(cands), "finished": True})
                except Exception:
                    # publish a terminal error so the UI progress bar never hangs
                    events.publish({"type": "benchmark", "benchmark_id": bid,
                                    "task_type": body.task_type, "error": True, "finished": True})

        task = asyncio.create_task(_run())
        app.state.bg_tasks.add(task)
        task.add_done_callback(app.state.bg_tasks.discard)
        return {"benchmark_id": bid, "status": "running",
                "calls": len(cands) * benchmarks.bank_size(body.task_type)}

    @app.get("/benchmarks")
    async def get_benchmarks():
        latest = store.latest_benchmarks(app.state.settings.db_path)
        return [{"model": k[0], "task_type": k[1], **v} for k, v in latest.items()]

    _static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")

    @app.get("/")
    async def dashboard():
        return FileResponse(_static / "dashboard.html", media_type="text/html")

    @app.get("/control-panel")
    async def control_panel():
        return FileResponse(_static / "control-panel.html", media_type="text/html")

    return app


app = create_app()
