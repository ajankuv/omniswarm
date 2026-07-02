"""OmniSwarm MCP server — exposes the engine as native Claude Code tools.

Runs in-process over stdio. It does NOT need the OmniSwarm HTTP service to be
running; it calls the engine directly, reusing all council/QC logic.
"""
import asyncio
import httpx
import logging
import os
import time
from mcp.server.fastmcp import FastMCP

from omniswarm import catalog, engine, recommend, registry, runtime, store
from omniswarm.config import get_settings
from omniswarm.engine import JobRequest
from omniswarm.util import new_id

log = logging.getLogger("omniswarm.mcp")

_mcp = FastMCP("omniswarm")
_settings = get_settings()
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        timeout = httpx.Timeout(
            connect=_settings.connect_timeout, read=_settings.read_timeout,
            write=_settings.read_timeout, pool=_settings.connect_timeout,
        )
        _client = httpx.AsyncClient(
            timeout=timeout, limits=httpx.Limits(max_connections=_settings.max_connections)
        )
    return _client


class _RemoteError(Exception):
    """The remote OmniSwarm app could not be reached or returned an error."""


def _remote_base() -> str | None:
    return os.environ.get("OMNISWARM_REMOTE") or None


def _remote_headers() -> dict:
    tok = os.environ.get("OMNISWARM_REMOTE_TOKEN") or ""
    return {"Authorization": f"Bearer {tok}"} if tok else {}


async def _remote_post(path: str, body: dict) -> dict:
    try:
        base = _remote_base()
        if not base:
            raise _RemoteError("OMNISWARM_REMOTE not set")
        resp = await _get_client().post(base + path, json=body, headers=_remote_headers())
        if resp.status_code == 401:
            raise _RemoteError("unauthorized (set OMNISWARM_REMOTE_TOKEN)")
        resp.raise_for_status()
        return resp.json()
    except _RemoteError:
        raise
    except Exception as e:  # transport / decode errors
        raise _RemoteError(str(e)[:160])


async def _remote_get(path: str) -> tuple[int, dict | list]:
    try:
        base = _remote_base()
        if not base:
            raise _RemoteError("OMNISWARM_REMOTE not set")
        resp = await _get_client().get(base + path, headers=_remote_headers())
        if resp.status_code == 404:
            return 404, {}
        if resp.status_code == 401:
            raise _RemoteError("unauthorized (set OMNISWARM_REMOTE_TOKEN)")
        resp.raise_for_status()
        return resp.status_code, resp.json()
    except _RemoteError:
        raise
    except Exception as e:
        raise _RemoteError(str(e)[:160])


_DEFAULT_SYS = "You are a precise, helpful assistant. Do exactly what is asked."
_bg_tasks: set[asyncio.Task] = set()
_bg_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _bg_sem
    if _bg_sem is None:
        _bg_sem = asyncio.Semaphore(int(os.environ.get("OMNISWARM_MAX_CONCURRENT_JOBS", "3")))
    return _bg_sem


def _startup() -> None:
    """Ensure the DB exists and MCP jobs use the same models the Control Panel picker set.
    Called from main() at server start — not at import (keeps tests side-effect free)."""
    store.init_db(_settings.db_path)
    registry.apply_runtime(runtime.load_runtime())


@_mcp.tool()
async def omniswarm_delegate(task: str, task_type: str = "general") -> dict:
    """Delegate bulk or templated work to free models, QC'd by a council before return.

    Use this to offload work that does not need your own reasoning — summaries, drafts,
    classification, first-pass code, extraction — so it costs $0 instead of your tokens.
    Returns a vetted result plus a confidence verdict:
      - verdict "pass" + confidence "high"  -> trust it
      - verdict "escalated"                  -> the council was unsure; read the result and decide
      - status "failed"                      -> the free-model gateway was unreachable

    Args:
        task: the full instruction or content to process.
        task_type: one of general, summarize, classify, draft, code, reasoning (default general).
    """
    if _remote_base():
        try:
            resp = await _remote_post("/v1/jobs", {
                "messages": [{"role": "user", "content": task}],
                "omniswarm": {"task_type": task_type},
            })
        except _RemoteError as e:
            log.warning("omniswarm_delegate remote error: %s", e)
            return {"status": "error", "error": str(e)}
        job_id = resp.get("job_id")
        if not job_id:
            return {"status": "error", "error": "remote did not return a job_id"}
        deadline = time.monotonic() + float(os.environ.get("OMNISWARM_REMOTE_POLL_TIMEOUT", "180"))
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            try:
                code, row = await _remote_get(f"/jobs/{job_id}")
            except _RemoteError as e:
                return {"status": "error", "error": str(e)}
            if code == 404:
                continue
            if row.get("status") in ("done", "escalated", "failed"):
                return {k: row.get(k) for k in
                        ("status", "verdict", "confidence", "result", "models_used", "note")}
        return {"status": "running", "job_id": job_id,
                "note": "still running after timeout; poll omniswarm_result"}

    store.init_db(_settings.db_path)
    req = engine.JobRequest(
        task_type=task_type,
        system="You are a precise, helpful assistant. Do exactly what is asked.",
        user=task,
    )
    return await engine.process_job(_get_client(), _settings, req)


@_mcp.tool()
async def omniswarm_stats() -> dict:
    """Report what OmniSwarm has worked on: job counts by status and verdict, and the
    approximate number of Claude tokens saved by offloading work to free models."""
    store.init_db(_settings.db_path)
    return store.stats(_settings.db_path)


@_mcp.tool()
async def omniswarm_submit(task: str, task_type: str = "general") -> dict:
    """Submit work to free models WITHOUT waiting — returns a job_id immediately.

    Use this to fan out bulk/templated work (summaries, drafts, classification, first-pass
    code, extraction) and keep working; collect the vetted result later with omniswarm_result.
    The job is council-QC'd just like omniswarm_delegate, but runs in the background.

    Args:
        task: the full instruction or content to process.
        task_type: one of general, summarize, classify, draft, code, reasoning (default general).
    Returns: {"job_id": <id>, "status": "running"}.
    """
    if _remote_base():
        try:
            resp = await _remote_post("/v1/jobs", {
                "messages": [{"role": "user", "content": task}],
                "omniswarm": {"task_type": task_type},
            })
        except _RemoteError as e:
            log.warning("omniswarm_submit remote error: %s", e)
            return {"job_id": None, "status": "error", "error": str(e)}
        job_id = resp.get("job_id")
        if not job_id:
            return {"job_id": None, "status": "error", "error": "remote did not return a job_id"}
        return {"job_id": job_id, "status": "running"}

    job_id = new_id()

    async def _run():
        async with _get_sem():
            try:
                await engine.process_job(
                    _get_client(), _settings,
                    JobRequest(task_type=task_type, system=_DEFAULT_SYS, user=task),
                    job_id=job_id,
                )
            except Exception as exc:
                # Non-gateway error: process_job only records OmniRouteError failures,
                # so mark the row failed here to avoid a job stuck "running" forever.
                try:
                    await asyncio.to_thread(
                        store.update_job, _settings.db_path, job_id,
                        status="failed", note=f"unexpected error: {exc}",
                    )
                except Exception:
                    pass

    t = asyncio.create_task(_run())
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return {"job_id": job_id, "status": "running"}


@_mcp.tool()
async def omniswarm_result(job_id: str) -> dict:
    """Fetch the outcome of a job submitted with omniswarm_submit.

    Returns {"status": "running"} if not finished yet, {"status": "not_found"} for an
    unknown id, or the finished job with verdict/confidence/result/models_used:
      - verdict "pass" + confidence "high" -> trust it
      - verdict "escalated" -> the council was unsure; read result and decide
      - status "failed" -> the free-model gateway was unreachable
    """
    if _remote_base():
        try:
            code, data = await _remote_get(f"/jobs/{job_id}")
        except _RemoteError as e:
            return {"status": "error", "error": str(e)}
        if code == 404:
            return {"status": "not_found"}
        return {k: data.get(k) for k in
                ("status", "verdict", "confidence", "result", "models_used", "note")}

    row = await asyncio.to_thread(store.get_job, _settings.db_path, job_id)
    if row is None:
        return {"status": "not_found"}
    return {
        "status": row.get("status"),
        "verdict": row.get("verdict"),
        "confidence": row.get("confidence"),
        "result": row.get("result"),
        "models_used": row.get("models_used"),
        "note": row.get("note"),
    }


@_mcp.tool()
async def omniswarm_list_jobs(limit: int = 20, status: str | None = None) -> list[dict]:
    """List recent OmniSwarm jobs (most recent first). Optional status filter
    (running|done|escalated|failed). Returns compact rows without the heavy result text."""
    if _remote_base():
        q = f"/jobs?limit={int(limit)}" + (f"&status={status}" if status else "")
        try:
            _code, rows = await _remote_get(q)
        except _RemoteError as e:
            log.warning("omniswarm_list_jobs remote error: %s", e)
            return []
        return [
            {"id": r.get("id"), "task_type": r.get("task_type"), "status": r.get("status"),
             "verdict": r.get("verdict"), "confidence": r.get("confidence"),
             "created_at": r.get("created_at")}
            for r in rows
        ]

    rows = await asyncio.to_thread(
        store.list_jobs, _settings.db_path, limit, status, None, None, None
    )
    return [
        {"id": r["id"], "task_type": r.get("task_type"), "status": r.get("status"),
         "verdict": r.get("verdict"), "confidence": r.get("confidence"),
         "created_at": r.get("created_at")}
        for r in rows
    ]


@_mcp.tool()
async def omniswarm_recommend(task_type: str = "general") -> dict:
    """Recommend the best FREE model for a task type, from the live gateway catalog +
    OmniSwarm's own success/latency history. Use this to self-select a model before
    delegating. Returns {"recommended": <id or null>, "ranked": [...], "why": <str>}."""
    try:
        cat = await catalog.fetch_catalog(_get_client(), _settings.omniroute_base_url)
    except Exception:
        return {"recommended": None, "ranked": [], "why": "catalog unavailable"}
    rel = await asyncio.to_thread(store.model_reliability, _settings.db_path)
    return recommend.recommend(task_type, cat, rel)


@_mcp.tool()
async def omniswarm_list_task_types() -> list[dict]:
    """List the task types OmniSwarm supports, the model each currently uses, and whether
    the type is high-stakes (always reviewed by the full council)."""
    return [
        {"name": name, "model": t.model, "high_stakes": t.high_stakes}
        for name, t in registry.REGISTRY.items()
    ]


def main() -> None:
    """Console entry point: run the MCP server over stdio."""
    _startup()
    _mcp.run()


if __name__ == "__main__":
    main()
