import pytest
from omniswarm import benchmarks


def test_banks_cover_all_task_types_with_edge_cases():
    for tt in ("classify", "code", "reasoning", "general"):
        assert benchmarks.bank_size(tt) >= 7
    for tt in ("summarize", "draft"):
        assert benchmarks.bank_size(tt) >= 6


@pytest.mark.asyncio
async def test_run_benchmark_objective_scores_pass_rate(monkeypatch):
    # classify bank: fake generate returns the correct label for every prompt
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=0):
        u = user.lower()
        if "love" in u or "best" in u:
            return "positive"
        if "broke" in u or "waste" in u:
            return "negative"
        if "crash" in u:
            return "support"
        if "cancel" in u:
            return "cancel"
        if "tuesday" in u or "arrived" in u:
            return "neutral"
        if "bonjour" in u:
            return "french"
        if "fed" in u or "rate" in u:
            return "finance"
        return "spam"
    monkeypatch.setattr(benchmarks, "generate", fake_generate)
    out = await benchmarks.run_benchmark(None, "http://x/v1", "classify", ["m/perfect"])
    r = out[0]
    assert r["model"] == "m/perfect" and r["task_type"] == "classify"
    assert r["pass_rate"] == 1.0 and r["quality"] == 1.0 and r["failures"] == 0
    assert r["samples"] == benchmarks.bank_size("classify")


@pytest.mark.asyncio
async def test_run_benchmark_graceful_candidate_failure(monkeypatch):
    async def boom(client, base_url, model, system, user, max_tokens=512, max_retries=0):
        raise RuntimeError("gateway down")
    monkeypatch.setattr(benchmarks, "generate", boom)
    out = await benchmarks.run_benchmark(None, "http://x/v1", "general", ["m/dead"])
    r = out[0]
    assert r["quality"] == 0.0 and r["failures"] == r["samples"]  # all failed, no stall


@pytest.mark.asyncio
async def test_run_benchmark_rubric_uses_judge(monkeypatch):
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=0):
        return "a concise summary."
    async def fake_judge(client, base_url, model, rubric, user_input, candidate):
        from omniswarm.council import JudgeResult
        return JudgeResult(score=0.9, reason="good", action="pass")
    monkeypatch.setattr(benchmarks, "generate", fake_generate)
    monkeypatch.setattr(benchmarks, "judge", fake_judge)
    out = await benchmarks.run_benchmark(None, "http://x/v1", "summarize", ["m/writer"])
    assert out[0]["quality"] == pytest.approx(0.9, abs=0.01)


@pytest.mark.asyncio
async def test_run_benchmark_reports_progress(monkeypatch):
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=0):
        return "42"
    monkeypatch.setattr(benchmarks, "generate", fake_generate)
    seen = []
    await benchmarks.run_benchmark(None, "http://x/v1", "general", ["m/a", "m/b"],
                                   on_progress=lambda ev: seen.append(ev))
    assert len(seen) == 2 and seen[-1]["done"] == 2 and seen[-1]["total"] == 2


def test_num_checker_rejects_superstring():
    chk = benchmarks._num(3)
    assert chk("ANSWER: 3") is True
    assert chk("I count 13") is False
    assert chk("35 things") is False


def test_clock_edge_case_rejects_common_wrong_answer():
    # find the clock item's checker in the reasoning bank
    items = benchmarks.OBJECTIVE_BANKS["reasoning"][1]
    clock = [chk for (user, chk) in items if "clock" in user.lower()][0]
    assert clock("The angle is 7.5 degrees") is True
    assert clock("The angle is 75 degrees") is False


def test_gold_symbol_rejects_australia():
    items = benchmarks.OBJECTIVE_BANKS["general"][1]
    gold = [chk for (user, chk) in items if "gold" in user.lower()][0]
    assert gold("Au") is True
    assert gold("The capital of Australia is Canberra") is False


@pytest.mark.asyncio
async def test_run_benchmark_rubric_survives_judge_failure(monkeypatch):
    async def fake_generate(client, base_url, model, system, user, max_tokens=512, max_retries=0):
        return "a summary"
    async def boom_judge(client, base_url, model, rubric, user_input, candidate):
        raise RuntimeError("judge down")
    monkeypatch.setattr(benchmarks, "generate", fake_generate)
    monkeypatch.setattr(benchmarks, "judge", boom_judge)
    out = await benchmarks.run_benchmark(None, "http://x/v1", "summarize", ["m/x"])
    r = out[0]
    assert r["failures"] == r["samples"] and r["quality"] == 0.0  # judge failures don't stall/crash
