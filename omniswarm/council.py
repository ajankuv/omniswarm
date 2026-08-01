import asyncio
from dataclasses import dataclass, field

from omniswarm.adapters import generate
from omniswarm import registry
from omniswarm.registry import TaskType, resolve_roster
from omniswarm.util import extract_json
from omniswarm.validators import run_validators


@dataclass
class JudgeResult:
    score: float | None   # None = the judge output did not parse (no signal, not 0.0)
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
    # A parse failure must be "no signal" (None), NOT a 0.0 — otherwise a judge whose
    # JSON didn't parse looks identical to a judge that scored the answer terribly, and
    # the council would wrongly escalate answers its reviewers unanimously approved.
    try:
        score = float(data["score"]) if "score" in data else None
    except (TypeError, ValueError):
        score = None
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
    data = extract_json(raw, require_keys=("verdict", "issues"))
    # A reviewer whose output didn't parse has cast NO vote — abstain (return None,
    # filtered out upstream). Defaulting to "revise" would let parse noise masquerade
    # as a real "needs changes" vote and wrongly tip the escalation majority.
    if not data or "verdict" not in data:
        return None
    verdict = data["verdict"]
    if verdict not in {"approve", "revise", "reject"}:
        verdict = "revise"
    return {"role": member["role"], "model": member["model"],
            "verdict": verdict, "issues": str(data.get("issues", ""))}


async def run_council(client, base_url, members, synth_model, user_input, candidate, on_step=None, judge_score=None) -> ReviewResult:
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
    reject_count = sum(1 for r in ok if r["verdict"] == "reject")
    revise_count = sum(1 for r in ok if r["verdict"] == "revise")
    approve_count = sum(1 for r in ok if r["verdict"] == "approve")
    any_reject = reject_count > 0
    any_revise = revise_count > 0
    models = [r["model"] for r in ok] + [synth_model]
    # Verdict is driven by OBJECTIVE council signals, not the chair's self-reported
    # confidence string (a strong model can solve correctly yet self-report "low").
    # Escalate on a real problem:
    #  - any reviewer reject, or the chair stated a disagreement;
    #  - EVERY reviewer wants changes and none approved — the unanimous "needs work"
    #    that junk/agentic-promise answers produce (this was the pass-medium bug).
    #    A split vote (some approve, some revise) is NOT escalated — it passes with
    #    honest "medium" confidence, so ordinary nitpicks don't send everything back.
    #  - the judge already scored the draft poorly — a low Tier-1 score must not be
    #    discarded just because the council ran.
    wants_change = reject_count + revise_count
    unanimous_change = approve_count == 0 and wants_change > 0
    low_judge = judge_score is not None and judge_score < 0.5
    escalate = any_reject or bool(disagreement) or unanimous_change or low_judge
    verdict = "escalated" if escalate else "pass"
    confidence = "low" if escalate else ("medium" if any_revise else "high")
    note = (disagreement
            or ("a reviewer rejected the draft" if any_reject else "")
            or (f"all {len(ok)} reviewers wanted changes" if unanimous_change else "")
            or (f"judge scored the draft {judge_score:.2f}" if low_judge else ""))
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
    last_judge_score = jr.score
    add({"stage": "judge", "model": registry.JUDGE_MODEL, "score": jr.score,
         "action": jr.action, "detail": jr.reason})
    if jr.action == "pass" and jr.score is not None and jr.score >= 0.8 and not task.high_stakes and not always_council:
        return finish(ReviewResult("pass", confidence_from_score(jr.score), candidate, models_used))
    if jr.action == "pass" and jr.score is not None and jr.score >= 0.8 and (task.high_stakes or always_council):
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
        last_judge_score = jr2.score
        add({"stage": "judge", "model": registry.JUDGE_MODEL, "score": jr2.score,
             "action": jr2.action, "detail": jr2.reason})
        if jr2.action == "pass" and jr2.score is not None and jr2.score >= 0.8 and not task.high_stakes and not always_council:
            return finish(ReviewResult("pass", confidence_from_score(jr2.score), candidate, models_used))

    # Budget guard before the expensive tier.
    if len(models_used) >= task.budget:
        add({"stage": "budget", "detail": "budget exhausted before council"})
        return finish(ReviewResult("escalated", "low", candidate, models_used,
                                   note="budget exhausted before council"))

    # Tier 2: role-based council review board.
    add({"stage": "council", "detail": "escalating to the role-based council"})
    cr = await run_council(client, base_url, resolve_roster(task.name, active_roster), registry.COUNCIL_SYNTH_MODEL,
                           user_input, candidate, on_step=on_step, judge_score=last_judge_score)
    cr.models_used = models_used + cr.models_used
    return finish(cr)
