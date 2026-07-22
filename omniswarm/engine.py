import asyncio
import json
from dataclasses import dataclass

from omniswarm import events, store
from omniswarm.adapters import generate, OmniRouteError
from omniswarm.council import review
from omniswarm.registry import get_task_type
from omniswarm.util import cache_key, estimate_tokens, new_id


@dataclass
class JobRequest:
    task_type: str
    system: str
    user: str
    max_tokens: int = 512


_CONF_RANK = {"high": 3, "medium": 2, "low": 1}


def _meets_confidence(conf: str, minimum: str) -> bool:
    return _CONF_RANK.get(conf, 0) >= _CONF_RANK.get(minimum, 3)


def _mask(text: str, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "redact":
        return f"[redacted: {len(text)} chars]"
    return text


async def process_job(client, settings, req: JobRequest, store_mode: str = "full", active_roster=None, always_council=False, job_id=None) -> dict:
    task = get_task_type(req.task_type)
    job_id = job_id or new_id()
    key = cache_key(req.system, req.user, task.name)
    await asyncio.to_thread(store.create_job, settings.db_path, job_id, task.name, "running")
    events.publish({"type": "job", "job_id": job_id, "status": "running", "task_type": task.name})
    await asyncio.to_thread(store.update_job, settings.db_path, job_id,
                            input=_mask(req.user, store_mode), cache_key=key)
    base = settings.omniroute_base_url

    # Feature B — verified-answer cache: a repeat of a prompt we already QC'd to
    # pass+high returns the vetted answer instantly, $0, no council call.
    if getattr(settings, "cache_enabled", False):
        cached = await asyncio.to_thread(store.cache_get, settings.db_path, key,
                                         settings.cache_ttl_seconds)
        if cached:
            saved = estimate_tokens(req.user) + estimate_tokens(cached["result"] or "")
            prov = [{"stage": "cache", "model": "verified-cache", "detail": "cache hit",
                     "verdict": cached["verdict"], "confidence": cached["confidence"]}]
            events.publish({"type": "step", "job_id": job_id, "stage": "cache",
                            "model": "verified-cache", "detail": "served from verified cache"})
            await asyncio.to_thread(
                store.update_job, settings.db_path, job_id,
                status="done", verdict=cached["verdict"], confidence=cached["confidence"],
                result=_mask(cached["result"] or "", store_mode), tokens_saved=saved,
                note=store.CACHE_HIT_NOTE, models_used=cached["models_used"] or "[]",
                provenance=(json.dumps(prov) if store_mode == "full" else "[]"),
            )
            events.publish({"type": "job", "job_id": job_id, "status": "done",
                            "verdict": cached["verdict"], "task_type": task.name,
                            "cache_hit": True})
            return {
                "job_id": job_id, "text": cached["result"] or "", "verdict": cached["verdict"],
                "confidence": cached["confidence"], "models_used": json.loads(cached["models_used"] or "[]"),
                "tokens_saved": saved, "status": "done", "cache_hit": True,
            }

    try:
        candidate = await generate(client, base, task.model, req.system, req.user, req.max_tokens)
        await asyncio.to_thread(
            store.add_event, settings.db_path, job_id, "generate", task.model, candidate
        )
        rr = await review(client, base, task, req.system, req.user, candidate, active_roster=active_roster, always_council=always_council, on_step=lambda s: events.publish({"type": "step", "job_id": job_id, **s}))
        status = "done" if rr.verdict == "pass" else "escalated"
        saved = estimate_tokens(req.user) + estimate_tokens(rr.text)
        await asyncio.to_thread(
            store.update_job, settings.db_path, job_id,
            status=status, verdict=rr.verdict, confidence=rr.confidence,
            result=_mask(rr.text, store_mode), tokens_saved=saved, note=rr.note,
            models_used=json.dumps(rr.models_used),
            provenance=(json.dumps(rr.steps) if store_mode == "full" else "[]"),
        )
        # cache only vetted, high-confidence answers — and only when we're
        # persisting text anyway (privacy modes must not leak into the cache).
        if (getattr(settings, "cache_enabled", False) and store_mode == "full"
                and rr.verdict == "pass"
                and _meets_confidence(rr.confidence, settings.cache_min_confidence)):
            await asyncio.to_thread(
                store.cache_put, settings.db_path, key, task.name, rr.text,
                rr.verdict, rr.confidence, json.dumps(rr.models_used),
                settings.cache_ttl_seconds, getattr(settings, "cache_max_entries", 0),
            )
        events.publish({"type": "job", "job_id": job_id, "status": status,
                        "verdict": rr.verdict, "task_type": task.name})
        return {
            "job_id": job_id, "text": rr.text, "verdict": rr.verdict,
            "confidence": rr.confidence, "models_used": rr.models_used,
            "tokens_saved": saved, "status": status, "cache_hit": False,
        }
    except OmniRouteError as e:
        await asyncio.to_thread(
            store.update_job, settings.db_path, job_id,
            status="failed", note=str(e), models_used="[]", provenance="[]",
        )
        events.publish({"type": "job", "job_id": job_id, "status": "failed",
                        "verdict": "failed", "task_type": task.name})
        return {
            "job_id": job_id, "text": "", "verdict": "failed", "confidence": "low",
            "models_used": [], "tokens_saved": 0, "status": "failed",
        }
