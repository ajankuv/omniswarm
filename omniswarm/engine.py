import asyncio
import json
from dataclasses import dataclass

from omniswarm import events, store
from omniswarm.adapters import generate, OmniRouteError
from omniswarm.council import review
from omniswarm.registry import get_task_type
from omniswarm.util import estimate_tokens, new_id


@dataclass
class JobRequest:
    task_type: str
    system: str
    user: str
    max_tokens: int = 512


def _mask(text: str, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "redact":
        return f"[redacted: {len(text)} chars]"
    return text


async def process_job(client, settings, req: JobRequest, store_mode: str = "full", active_roster=None, always_council=False, job_id=None) -> dict:
    task = get_task_type(req.task_type)
    job_id = job_id or new_id()
    await asyncio.to_thread(store.create_job, settings.db_path, job_id, task.name, "running")
    events.publish({"type": "job", "job_id": job_id, "status": "running", "task_type": task.name})
    await asyncio.to_thread(store.update_job, settings.db_path, job_id, input=_mask(req.user, store_mode))
    base = settings.omniroute_base_url
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
        events.publish({"type": "job", "job_id": job_id, "status": status,
                        "verdict": rr.verdict, "task_type": task.name})
        return {
            "job_id": job_id, "text": rr.text, "verdict": rr.verdict,
            "confidence": rr.confidence, "models_used": rr.models_used,
            "tokens_saved": saved, "status": status,
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
