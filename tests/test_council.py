import dataclasses
import pytest
from omniswarm import council
from omniswarm.registry import get_task_type


def test_confidence_thresholds():
    assert council.confidence_from_score(0.95) == "high"
    assert council.confidence_from_score(0.6) == "medium"
    assert council.confidence_from_score(0.2) == "low"


@pytest.mark.asyncio
async def test_review_passes_when_judge_confident(monkeypatch):
    async def fake_judge(*a, **k):
        return council.JudgeResult(score=0.9, reason="good", action="pass")
    monkeypatch.setattr(council, "judge", fake_judge)
    task = get_task_type("general")
    r = await council.review(None, "http://x/v1", task, "sys", "user", "candidate text")
    assert r.verdict == "pass"
    assert r.confidence == "high"
    assert r.text == "candidate text"


@pytest.mark.asyncio
async def test_review_escalates_when_validators_fail_twice(monkeypatch):
    async def fake_generate(*a, **k):
        return "   "  # still blank after fix attempt
    monkeypatch.setattr(council, "generate", fake_generate)
    task = get_task_type("general")  # validators=("non_empty",)
    r = await council.review(None, "http://x/v1", task, "sys", "user", "   ")
    assert r.verdict == "escalated"
    assert "non_empty" in r.note


@pytest.mark.asyncio
async def test_unsure_judge_triggers_council(monkeypatch):
    async def fake_judge(*a, **k):
        return council.JudgeResult(score=0.5, reason="meh", action="unsure")
    async def fake_council(*a, **k):
        return council.ReviewResult("pass", "medium", "synth answer", ["m1", "m2", "synth"])
    monkeypatch.setattr(council, "judge", fake_judge)
    monkeypatch.setattr(council, "run_council", fake_council)
    task = get_task_type("general")
    r = await council.review(None, "http://x/v1", task, "sys", "user", "candidate")
    assert r.text == "synth answer"
    assert r.verdict == "pass"


@pytest.mark.asyncio
async def test_budget_guard_escalates_before_council(monkeypatch):
    """Budget guard fires when models_used reaches budget before council tier."""
    async def fake_judge(*a, **k):
        return council.JudgeResult(score=0.5, reason="meh", action="unsure")
    monkeypatch.setattr(council, "judge", fake_judge)
    # budget=2: models_used=[task.model] after init (1), +JUDGE_MODEL after judge (2)
    # guard fires: len(models_used) >= task.budget -> escalate
    task = dataclasses.replace(get_task_type("general"), budget=2)
    r = await council.review(None, "http://x/v1", task, "sys", "user", "candidate")
    assert r.verdict == "escalated"
    assert "budget" in r.note


@pytest.mark.asyncio
async def test_fix_revise_rejudge_returns_pass(monkeypatch):
    """Fix branch: first judge returns fix, generate produces revised text, second judge passes."""
    call_count = {"n": 0}

    async def fake_judge(*a, **k):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return council.JudgeResult(score=0.6, reason="needs work", action="fix")
        return council.JudgeResult(score=0.9, reason="great", action="pass")

    async def fake_generate(*a, **k):
        return "revised candidate"

    monkeypatch.setattr(council, "judge", fake_judge)
    monkeypatch.setattr(council, "generate", fake_generate)
    task = get_task_type("general")  # budget=8, high_stakes=False
    r = await council.review(None, "http://x/v1", task, "sys", "user", "original candidate")
    assert r.verdict == "pass"
    assert r.confidence == "high"
    assert r.text == "revised candidate"


@pytest.mark.asyncio
async def test_high_stakes_routes_to_council_despite_confident_judge(monkeypatch):
    """High-stakes tasks must not short-circuit at Tier-1; council is always reached."""
    async def fake_judge(*a, **k):
        return council.JudgeResult(score=0.9, reason="looks good", action="pass")

    sentinel = council.ReviewResult("pass", "high", "council answer", ["g1", "g2", "synth"])

    async def fake_council(*a, **k):
        return sentinel

    async def fake_generate(*a, **k):
        return "generate output"

    monkeypatch.setattr(council, "judge", fake_judge)
    monkeypatch.setattr(council, "run_council", fake_council)
    monkeypatch.setattr(council, "generate", fake_generate)
    task = get_task_type("reasoning")  # high_stakes=True
    r = await council.review(None, "http://x/v1", task, "sys", "user", "candidate")
    assert r.text == "council answer"
    assert r.verdict == "pass"


@pytest.mark.asyncio
async def test_review_records_steps_on_pass(monkeypatch):
    async def fake_judge(*a, **k):
        return council.JudgeResult(score=0.9, reason="solid", action="pass")
    monkeypatch.setattr(council, "judge", fake_judge)
    task = get_task_type("general")
    r = await council.review(None, "http://x/v1", task, "sys", "user", "candidate text")
    stages = [s["stage"] for s in r.steps]
    assert "draft" in stages
    assert "judge" in stages
    judge_step = next(s for s in r.steps if s["stage"] == "judge")
    assert judge_step["score"] == 0.9
    assert judge_step["detail"] == "solid"


@pytest.mark.asyncio
async def test_council_members_critique_and_synthesize(monkeypatch):
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        if "council chair" in system.lower():
            return '{"answer": "final answer", "confidence": "high", "disagreement": ""}'
        return '{"verdict": "approve", "issues": ""}'  # each member approves
    monkeypatch.setattr(council, "generate", fake_generate)
    members = [
        {"role": "Developer", "model": "a/dev", "focus": "bugs"},
        {"role": "Project Manager", "model": "b/pm", "focus": "scope"},
    ]
    r = await council.run_council(None, "http://x/v1", members, "c/synth", "task", "the draft")
    stages = [s["stage"] for s in r.steps]
    assert stages.count("review") == 2
    assert "synthesis" in stages
    roles = [s.get("role") for s in r.steps if s["stage"] == "review"]
    assert roles == ["Developer", "Project Manager"]
    assert r.verdict == "pass"
    assert r.text == "final answer"


@pytest.mark.asyncio
async def test_council_passes_on_unanimous_approval_despite_low_self_report(monkeypatch):
    # Regression: a correct answer with all members approving and no disagreement
    # must PASS even when the synth model self-reports "low" confidence (or omits it).
    # The verdict is driven by objective council signals, not the chair's self-doubt.
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        if "council chair" in system.lower():
            return '{"answer": "5 cents", "confidence": "low", "disagreement": ""}'
        return '{"verdict": "approve", "issues": ""}'  # every member approves
    monkeypatch.setattr(council, "generate", fake_generate)
    members = [{"role": "Fact-Checker", "model": "a/fc", "focus": "facts"},
               {"role": "Data Guardian", "model": "b/dg", "focus": "numbers"}]
    r = await council.run_council(None, "http://x/v1", members, "c/synth", "task", "5 cents")
    assert r.verdict == "pass"
    assert r.confidence == "high"  # unanimous approve, no revise -> high


@pytest.mark.asyncio
async def test_council_medium_confidence_when_a_member_asks_revise(monkeypatch):
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        if "council chair" in system.lower():
            return '{"answer": "ok", "confidence": "high", "disagreement": ""}'
        if model == "a/fc":
            return '{"verdict": "revise", "issues": "tighten wording"}'
        return '{"verdict": "approve", "issues": ""}'
    monkeypatch.setattr(council, "generate", fake_generate)
    members = [{"role": "Fact-Checker", "model": "a/fc", "focus": "facts"},
               {"role": "Editor", "model": "b/ed", "focus": "clarity"}]
    r = await council.run_council(None, "http://x/v1", members, "c/synth", "task", "draft")
    assert r.verdict == "pass"
    assert r.confidence == "medium"  # a revise (no reject/disagreement) -> medium


@pytest.mark.asyncio
async def test_council_escalates_when_member_rejects(monkeypatch):
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        if "council chair" in system.lower():
            return '{"answer": "x", "confidence": "high", "disagreement": ""}'
        if model == "a/dev":
            return '{"verdict": "reject", "issues": "logic is broken"}'
        return '{"verdict": "approve", "issues": ""}'
    monkeypatch.setattr(council, "generate", fake_generate)
    members = [{"role": "Developer", "model": "a/dev", "focus": "bugs"},
               {"role": "PM", "model": "b/pm", "focus": "scope"}]
    r = await council.run_council(None, "http://x/v1", members, "c/synth", "task", "draft")
    assert r.verdict == "escalated"
    assert "reject" in r.note.lower() or "broken" in r.note.lower()


@pytest.mark.asyncio
async def test_review_uses_active_roster(monkeypatch):
    captured = {}
    async def fake_run_council(client, base_url, members, synth_model, user_input, candidate, **kwargs):
        captured["roles"] = [m["role"] for m in members]
        return council.ReviewResult("pass", "high", "x", ["m"])
    async def fake_judge(*a, **k):
        return council.JudgeResult(score=0.5, reason="meh", action="unsure")
    monkeypatch.setattr(council, "run_council", fake_run_council)
    monkeypatch.setattr(council, "judge", fake_judge)
    task = get_task_type("general")
    await council.review(None, "http://x/v1", task, "sys", "user", "cand",
                         active_roster=["Fact-Checker", "Safety Sentinel"])
    assert captured["roles"] == ["Fact-Checker", "Safety Sentinel"]


@pytest.mark.asyncio
async def test_always_council_routes_to_council_on_confident_pass(monkeypatch):
    async def fake_judge(*a, **k):
        return council.JudgeResult(score=0.95, reason="great", action="pass")
    async def fake_run_council(*a, **k):
        return council.ReviewResult("pass", "high", "board answer", ["m1", "m2"])
    monkeypatch.setattr(council, "judge", fake_judge)
    monkeypatch.setattr(council, "run_council", fake_run_council)
    task = get_task_type("general")  # not high_stakes
    r = await council.review(None, "http://x/v1", task, "sys", "user", "candidate", always_council=True)
    assert r.text == "board answer"  # did NOT short-circuit on the confident judge pass


@pytest.mark.asyncio
async def test_review_calls_on_step(monkeypatch):
    async def fake_judge(*a, **k):
        return council.JudgeResult(score=0.95, reason="ok", action="pass")
    monkeypatch.setattr(council, "judge", fake_judge)
    seen = []
    task = get_task_type("general")
    await council.review(None, "http://x/v1", task, "sys", "user", "cand",
                         always_council=False, on_step=lambda s: seen.append(s["stage"]))
    assert "draft" in seen and "judge" in seen


@pytest.mark.asyncio
async def test_always_council_routes_fix_branch_to_council(monkeypatch):
    # first judge says "fix", the re-judge passes; with always_council on it must
    # STILL reach the council instead of short-circuiting on the re-judge pass.
    calls = {"n": 0}

    async def fake_judge(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return council.JudgeResult(score=0.6, reason="fix it", action="fix")
        return council.JudgeResult(score=0.95, reason="better", action="pass")

    async def fake_generate(*a, **k):
        return "revised candidate"

    async def fake_run_council(*a, **k):
        return council.ReviewResult("pass", "high", "council answer", ["m1", "m2"])

    monkeypatch.setattr(council, "judge", fake_judge)
    monkeypatch.setattr(council, "generate", fake_generate)
    monkeypatch.setattr(council, "run_council", fake_run_council)
    task = get_task_type("general")
    r = await council.review(None, "http://x/v1", task, "sys", "user", "candidate", always_council=True)
    assert r.text == "council answer"  # did NOT short-circuit on the re-judge pass


@pytest.mark.asyncio
async def test_council_reads_judge_model_live(monkeypatch):
    from omniswarm import registry

    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        return '{"score": 1.0, "action": "pass", "reason": "ok"}'
    monkeypatch.setattr(council, "generate", fake_generate)
    registry.apply_runtime({"judge": "test/live-judge"})
    try:
        task = registry.get_task_type("general")
        r = await council.review(None, "http://x/v1", task, "sys", "q", "candidate",
                                 always_council=False)
        # judge ran with the live-applied model, recorded in models_used
        assert "test/live-judge" in r.models_used
    finally:
        registry.apply_runtime({})


@pytest.mark.asyncio
async def test_council_escalates_on_revise_majority(monkeypatch):
    # The "Build me a game" bug: every reviewer says the draft needs work, but the
    # old logic only escalated on a hard reject, so it returned pass+medium.
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        if "council chair" in system.lower():
            return '{"answer": "I will build it, let me start...", "disagreement": ""}'
        return '{"verdict": "revise", "issues": "no deliverable, lacks implementation"}'
    monkeypatch.setattr(council, "generate", fake_generate)
    members = [{"role": "Code Auditor", "model": "a/ca", "focus": "code"},
               {"role": "User Advocate", "model": "b/ua", "focus": "intent"},
               {"role": "Security Engineer", "model": "c/se", "focus": "sec"}]
    r = await council.run_council(None, "http://x/v1", members, "c/synth", "build me a game", "draft")
    assert r.verdict == "escalated"           # unanimous revise must NOT be a pass
    assert "all 3 reviewers wanted changes" in r.note


@pytest.mark.asyncio
async def test_council_escalates_when_judge_scored_low(monkeypatch):
    # A poor Tier-1 judge score must not be discarded once the council runs.
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        if "council chair" in system.lower():
            return '{"answer": "ok", "disagreement": ""}'
        return '{"verdict": "approve", "issues": ""}'   # council approves...
    monkeypatch.setattr(council, "generate", fake_generate)
    members = [{"role": "Dev", "model": "a/d", "focus": "x"}]
    # ...but the judge already scored it 0.2
    r = await council.run_council(None, "http://x/v1", members, "c/synth", "t", "draft", judge_score=0.2)
    assert r.verdict == "escalated"
    assert "0.2" in r.note


@pytest.mark.asyncio
async def test_council_still_passes_single_revise_among_approvals(monkeypatch):
    # Guard against over-correction: one dissenting "revise" among approvals stays pass.
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        if "council chair" in system.lower():
            return '{"answer": "ok", "disagreement": ""}'
        if model == "a/x":
            return '{"verdict": "revise", "issues": "nit"}'
        return '{"verdict": "approve", "issues": ""}'
    monkeypatch.setattr(council, "generate", fake_generate)
    members = [{"role": "A", "model": "a/x", "focus": ""},
               {"role": "B", "model": "b/y", "focus": ""},
               {"role": "C", "model": "c/z", "focus": ""}]
    r = await council.run_council(None, "http://x/v1", members, "c/synth", "t", "draft", judge_score=0.9)
    assert r.verdict == "pass" and r.confidence == "medium"


@pytest.mark.asyncio
async def test_judge_parse_failure_does_not_escalate_approved_answer(monkeypatch):
    # Regression from live stress test: the judge's JSON failed to parse (score None,
    # NOT 0.0). The council unanimously APPROVED. That must PASS, not escalate —
    # a missing judge signal is not a low score.
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        if "council chair" in system.lower():
            return '{"answer": "ok", "disagreement": ""}'
        return '{"verdict": "approve", "issues": ""}'
    monkeypatch.setattr(council, "generate", fake_generate)
    members = [{"role": "A", "model": "a/x", "focus": ""},
               {"role": "B", "model": "b/y", "focus": ""},
               {"role": "C", "model": "c/z", "focus": ""}]
    r = await council.run_council(None, "http://x/v1", members, "c/synth", "t", "draft", judge_score=None)
    assert r.verdict == "pass"          # unanimous approve + no judge signal -> pass
    assert r.confidence == "high"


@pytest.mark.asyncio
async def test_judge_returns_none_score_on_parse_failure(monkeypatch):
    async def fake_generate(*a, **k):
        return "the judge model rambled with no JSON at all"
    monkeypatch.setattr(council, "generate", fake_generate)
    jr = await council.judge(None, "http://x/v1", "m", "rubric", "task", "answer")
    assert jr.score is None and jr.action == "unsure"


@pytest.mark.asyncio
async def test_member_parse_failure_abstains_not_revise(monkeypatch):
    # Two reviewers' JSON is garbage; one approves. Parse failures must ABSTAIN,
    # not count as "revise" — otherwise noise forces a false escalation.
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        if "council chair" in system.lower():
            return '{"answer": "ok", "disagreement": ""}'
        if model == "good/m":
            return '{"verdict": "approve", "issues": ""}'
        return 'the model rambled, no json here'          # parse failure
    monkeypatch.setattr(council, "generate", fake_generate)
    members = [{"role": "A", "model": "good/m", "focus": ""},
               {"role": "B", "model": "bad/1", "focus": ""},
               {"role": "C", "model": "bad/2", "focus": ""}]
    r = await council.run_council(None, "http://x/v1", members, "c/synth", "t", "draft", judge_score=0.9)
    # only one real vote (approve) -> not a change-majority -> pass
    assert r.verdict == "pass"
    stages = [s for s in r.steps if s["stage"] == "review"]
    assert len(stages) == 1                                # abstainers cast no review


@pytest.mark.asyncio
async def test_council_passes_split_vote_with_medium_confidence(monkeypatch):
    # 2 revise, 1 approve — a split, NOT unanimous. Must PASS (medium), so ordinary
    # nitpicks don't escalate every legitimate answer back to the caller.
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=2):
        if "council chair" in system.lower():
            return '{"answer": "ok", "disagreement": ""}'
        if model == "c/ok":
            return '{"verdict": "approve", "issues": ""}'
        return '{"verdict": "revise", "issues": "could be tighter"}'
    monkeypatch.setattr(council, "generate", fake_generate)
    members = [{"role": "A", "model": "a/x", "focus": ""},
               {"role": "B", "model": "b/y", "focus": ""},
               {"role": "C", "model": "c/ok", "focus": ""}]
    r = await council.run_council(None, "http://x/v1", members, "c/synth", "t", "draft", judge_score=0.9)
    assert r.verdict == "pass" and r.confidence == "medium"
