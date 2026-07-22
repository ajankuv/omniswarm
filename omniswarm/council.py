import asyncio
from dataclasses import dataclass, field

from omniswarm.adapters import generate
from omniswarm import registry
from omniswarm.registry import TaskType, resolve_roster
from omniswarm.util import extract_json
from omniswarm.validators import run_validators


@dataclass
class JudgeResult:
    score: float
    reason: str
    action: str  # "pass" | "fix" | "unsure"


@dataclass
class ReviewResult:
    verdict: str  # "pass" | "escalated"
    confidence: str  # "high" | "medium" | "low"
    text: str
    models_used: list[str] = field(default_factory=list)
    note: str = ""
    steps: list[dict] = field(default_factory=list)


def confidence_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _snip(s: str, n: int = 240) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


_JUDGE_SYS = (
    "You are a strict quality judge. Apply the rubric to the candidate answer. "
    "Ignore answer length; do not reward verbosity. Respond ONLY with JSON: "
    '{"score": <0..1>, "action": "pass|fix|unsure", "reason": "<short>"}. '
    'Use "pass" only if the answer fully satisfies the rubric.'
)


async def judge(client, base_url, model, rubric, user_input, candidate) -> JudgeResult:
    user = f"RUBRIC:\n{rubric}\n\nTASK:\n{user_input}\n\nCANDIDATE ANSWER:\n{candidate}"
    raw = await generate(client, base_url, model, _JUDGE_SYS, user, max_tokens=300)
    # key-scoped: the prompt embeds the task + draft, so echoed text must not forge a verdict
    data = extract_json(raw, require_keys=("action", "score")) or {}
    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    action = data.get("action", "unsure")
    if action not in {"pass", "fix", "unsure"}:
        action = "unsure"
    return JudgeResult(score=score, reason=str(data.get("reason", "")), action=action)


_MEMBER_SYS = (
    "You are a {role} on a review council. Review the DRAFT answer strictly from your "
    "perspective as a {role}, focusing on: {focus}. Be concise and concrete. Respond ONLY "
    'with JSON: {{"verdict": "approve|revise|reject", "issues": "<short; empty if approve>"}}'
)

_SYNTH_SYS = (
    "You are the council chair. You receive a DRAFT answer and role-based critiques from "
    "several reviewers. Integrate the valid points into one final answer (keep the draft "
    "where critiques are unfounded; fix what is validly raised). Do NOT introduce claims "
    "unsupported by the draft or critiques. Respond ONLY with JSON: "
    '{"answer": "<final>", "confidence": "high|medium|low", "disagreement": "<short; empty>"}'
)


async def _member_review(client, base_url, member, user_input, candidate) -> dict:
    sys = _MEMBER_SYS.format(role=member["role"], focus=member.get("focus", ""))
    user = f"TASK:\n{user_input}\n\nDRAFT ANSWER:\n{candidate}"
    raw = await generate(client, base_url, member["model"], sys, user, max_tokens=300)
    data = extract_json(raw, require_keys=("verdict", "issues")) or {}
    verdict = data.get("verdict", "revise")
    if verdict not in {"approve", "revise", "reject"}:
        verdict = "revise"
    return {"role": member["role"], "model": member["model"],
            "verdict": verdict, "issues": str(data.get("issues", ""))}


async def run_council(client, base_url, members, synth_model, user_input, candidate, on_step=None) -> ReviewResult:
    steps: list[dict] = []

    def add(s):
        steps.append(s)
        if on_step:
            on_step(s)

    if not members:
        rr = ReviewResult("escalated", "low", candidate, [], note="council: no members configured")
        rr.steps = steps
        return rr
    reviews = await asyncio.gather(
        *[_member_review(client, base_url, m, user_input, candidate) for m in members],
        return_exceptions=True,
    )
    ok = [r for r in reviews if isinstance(r, dict)]
    for r in ok:
        detail = f'{r["verdict"]}: {r["issues"]}' if r["issues"] else r["verdict"]
        add({"stage": "review", "role": r["role"], "model": r["model"], "detail": detail})
    if not ok:
        rr = ReviewResult("escalated", "low", candidate, [m["model"] for m in members],
                          note="council: no reviewers responded")
        rr.steps = steps
        return rr
    critiques = "\n\n".join(f'[{r["role"]} — {r["verdict"]}] {r["issues"]}' for r in ok)
    synth_user = f"TASK:\n{user_input}\n\nDRAFT ANSWER:\n{candidate}\n\nCRITIQUES:\n{critiques}"
    raw = await generate(client, base_url, synth_model, _SYNTH_SYS, synth_user, max_tokens=800)
    data = extract_json(raw, require_keys=("answer", "disagreement")) or {}
    answer = str(data.get("answer", candidate))
    disagreement = str(data.get("disagreement", "")).strip()
    any_reject = any(r["verdict"] == "reject" for r in ok)
    any_revise = any(r["verdict"] == "revise" for r in ok)
    models = [r["model"] for r in ok] + [synth_model]
    # Verdict is driven by OBJECTIVE council signals, not the chair's self-reported
    # confidence string (a strong model can solve correctly yet self-report "low").
    # Escalate only on a real problem: a reviewer reject or stated disagreement.
    escalate = any_reject or bool(disagreement)
    verdict = "escalated" if escalate else "pass"
    confidence = "low" if escalate else ("medium" if any_revise else "high")
    note = disagreement or ("a reviewer rejected the draft" if any_reject else "")
    add({"stage": "synthesis", "model": synth_model, "confidence": confidence,
         "detail": disagreement or "council approved"})
    rr = ReviewResult(verdict, confidence, answer, models, note=note)
    rr.steps = steps
    return rr


async def review(client, base_url, task: TaskType, system, user_input, candidate, active_roster=None, always_council=False, on_step=None) -> ReviewResult:
    models_used = [task.model]
    steps: list[dict] = []

    def add(s):
        steps.append(s)
        if on_step:
            on_step(s)

    add({"stage": "draft", "model": task.model, "detail": _snip(candidate)})

    def finish(rr: ReviewResult) -> ReviewResult:
        rr.steps = steps + rr.steps
        return rr

    # Tier 0: deterministic validators, with one fix attempt.
    failed = run_validators(task.validators, candidate, task.params)
    if failed:
        add({"stage": "validate", "detail": f"failed: {', '.join(failed)} — regenerating"})
        candidate = await generate(
            client, base_url, task.model, system,
            f"{user_input}\n\nFix these problems: {', '.join(failed)}",
        )
        models_used.append(task.model)
        add({"stage": "fix", "model": task.model, "detail": _snip(candidate)})
        failed = run_validators(task.validators, candidate, task.params)
        if failed:
            add({"stage": "validate", "detail": f"still failing: {', '.join(failed)}"})
            return finish(ReviewResult("escalated", "low", candidate, models_used,
                                       note=f"validators failed: {', '.join(failed)}"))
    else:
        add({"stage": "validate", "detail": "passed"})

    # Tier 1: judge.
    jr = await judge(client, base_url, registry.JUDGE_MODEL, task.rubric, user_input, candidate)
    models_used.append(registry.JUDGE_MODEL)
    add({"stage": "judge", "model": registry.JUDGE_MODEL, "score": jr.score,
         "action": jr.action, "detail": jr.reason})
    if jr.action == "pass" and jr.score >= 0.8 and not task.high_stakes and not always_council:
        return finish(ReviewResult("pass", confidence_from_score(jr.score), candidate, models_used))
    if jr.action == "pass" and jr.score >= 0.8 and (task.high_stakes or always_council):
        add({"stage": "note", "detail": "council always-on / high-stakes — routing to council"})

    if jr.action == "fix" and len(models_used) < task.budget:
        candidate = await generate(
            client, base_url, task.model, system,
            f"{user_input}\n\nRevise considering this feedback: {jr.reason}",
        )
        models_used.append(task.model)
        add({"stage": "revise", "model": task.model, "detail": _snip(candidate)})
        jr2 = await judge(client, base_url, registry.JUDGE_MODEL, task.rubric, user_input, candidate)
        models_used.append(registry.JUDGE_MODEL)
        add({"stage": "judge", "model": registry.JUDGE_MODEL, "score": jr2.score,
             "action": jr2.action, "detail": jr2.reason})
        if jr2.action == "pass" and jr2.score >= 0.8 and not task.high_stakes and not always_council:
            return finish(ReviewResult("pass", confidence_from_score(jr2.score), candidate, models_used))

    # Budget guard before the expensive tier.
    if len(models_used) >= task.budget:
        add({"stage": "budget", "detail": "budget exhausted before council"})
        return finish(ReviewResult("escalated", "low", candidate, models_used,
                                   note="budget exhausted before council"))

    # Tier 2: role-based council review board.
    add({"stage": "council", "detail": "escalating to the role-based council"})
    cr = await run_council(client, base_url, resolve_roster(task.name, active_roster), registry.COUNCIL_SYNTH_MODEL,
                           user_input, candidate, on_step=on_step)
    cr.models_used = models_used + cr.models_used
    return finish(cr)
