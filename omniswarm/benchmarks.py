"""Curated eval banks + a benchmark runner. Measures free models' QUALITY per task type
so recommendations can be evidence-based. Objective banks are auto-scored by checkers;
rubric banks are scored by the configured judge model. No new dependencies.
"""
import asyncio
import re
import time

from omniswarm.adapters import generate
from omniswarm.council import judge
from omniswarm import registry


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _has(ans: str):
    return lambda out: _norm(ans) in _norm(out)


def _num(n):
    # match the number as a standalone value, not inside a larger number (e.g. 3 must not match "13")
    return lambda out: bool(re.search(rf"(?<![\d.]){re.escape(str(n))}(?![\d])", out or ""))


def _anyof(*a):
    return lambda out: any(_norm(x) in _norm(out) for x in a)


# --- objective banks: (system, [(user, checker), ...]) — includes >=1 edge/ambiguous case ---
OBJECTIVE_BANKS = {
    "classify": ("Classify. Reply with ONLY the single label, nothing else.", [
        ("Sentiment (positive/negative/neutral): 'Absolutely love it, best purchase ever!'", _has("positive")),
        ("Sentiment (positive/negative/neutral): 'It broke after one day, total waste.'", _has("negative")),
        ("Sentiment (positive/negative/neutral): 'The package arrived on Tuesday.'", _has("neutral")),
        ("Intent (cancel/refund/support/sales): 'I want to cancel my subscription now.'", _has("cancel")),
        ("Intent (cancel/refund/support/sales): 'My app keeps crashing on launch.'", _has("support")),
        ("Language (english/spanish/french/german): 'Bonjour, comment allez-vous?'", _has("french")),
        ("Topic (sports/finance/health/tech): 'The Fed raised interest rates by 25bps.'", _has("finance")),
        # edge: sarcasm reads negative despite positive words
        ("Sentiment (positive/negative/neutral): 'Oh great, it broke again. Just wonderful.'", _has("negative")),
    ]),
    "code": ("You are a careful programmer. Write correct code, then end with 'ANSWER: <result>'.", [
        ("Write a Python function to reverse a string, then give reverse('OmniSwarm').", _has("mrawSinmO")),
        ("Write is_prime(n); state is_prime(97).", _has("true")),
        ("Sum even numbers in [1..10]; give the sum.", _num(30)),
        ("Count vowels in 'benchmarking'; give the count.", _num(3)),
        ("FizzBuzz: what is printed for n=15?", _anyof("fizzbuzz")),
        ("Write factorial(n); give factorial(6).", _num(720)),
        ("Find the max of [3,1,4,1,5,9,2,6]; give it.", _num(9)),
        # edge: off-by-one boundary
        ("How many integers are in the inclusive range 3..17?", _num(15)),
    ]),
    "reasoning": ("Solve. End with a final line 'ANSWER: <value>'.", [
        ("A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How many cents is the ball?", _num(5)),
        ("If 5 machines take 5 min to make 5 widgets, how many minutes for 100 machines to make 100 widgets?", _num(5)),
        ("How many times does 'r' appear in 'strawberry'?", _num(3)),
        ("I have 3 apples, eat 1, buy 5 more, give away 2. How many now?", _num(5)),
        ("A farmer has 17 sheep; all but 9 die. How many are left?", _num(9)),
        ("Next number: 2, 6, 12, 20, 30, ?", _num(42)),
        ("If today is Monday, what day is it 100 days from now?", _has("wednesday")),
        # edge: classic clock-angle; decimal-preserving checker prevents false positive from "75 degrees"
        ("A clock shows 3:15. What is the angle in degrees between the hands?", lambda out: bool(re.search(r"7\.5", out))),
    ]),
    "general": ("Answer concisely and correctly. End with 'ANSWER: <value>' for factual answers.", [
        ("Capital of Australia?", _has("canberra")),
        ("Chemical symbol for gold?", lambda out: bool(re.search(r"\bau\b", out, re.I))),
        ("How many continents are there?", _num(7)),
        ("Who wrote 'Romeo and Juliet'?", _has("shakespeare")),
        ("What year did WWII end?", _num(1945)),
        ("Square root of 144?", _num(12)),
        ("What planet is the Red Planet?", _has("mars")),
        # edge: common misconception
        ("Is the Great Wall of China visible from the Moon with the naked eye? yes/no", _has("no")),
    ]),
}

# --- rubric banks: (system, [prompt, ...]) scored 0..1 by the judge ---
_RUBRIC_RUBRIC = ("Numbered evaluation steps:\n1. Directly responsive to the request?\n"
                  "2. Accurate and coherent?\n3. Concise and well-formed?\n4. Do not reward verbosity.")
RUBRIC_BANKS = {
    "summarize": ("Summarize the text in ONE clear sentence.", [
        "The mitochondria is the powerhouse of the cell, generating most of the cell's ATP via cellular respiration.",
        "Photosynthesis lets green plants use sunlight to turn carbon dioxide and water into glucose and oxygen.",
        "The 2008 crisis began with the US housing bubble collapse, causing major bank failures and a global recession.",
        "Machine learning is a subset of AI where systems learn patterns from data to predict without explicit programming.",
        "The water cycle: water evaporates, condenses into clouds, precipitates, and collects in rivers and oceans.",
        "Regular exercise improves cardiovascular health, strengthens muscles and bones, and lifts mood via endorphins.",
    ]),
    "draft": ("Write a single polished, compelling sentence for the request.", [
        "A tagline for a premium stainless-steel water bottle.",
        "A one-line product description for noise-cancelling headphones.",
        "An opening line for a blog post about sustainable gardening.",
        "A meta description for a page about cast-iron skillet care.",
        "A call-to-action for a newsletter signup on a cooking site.",
        "A one-sentence elevator pitch for a habit-tracking app.",
    ]),
}

_PER_CALL_TIMEOUT = 45  # seconds; a slow candidate fails this sample, not the run


def bank_size(task_type: str) -> int:
    if task_type in OBJECTIVE_BANKS:
        return len(OBJECTIVE_BANKS[task_type][1])
    if task_type in RUBRIC_BANKS:
        return len(RUBRIC_BANKS[task_type][1])
    return len(OBJECTIVE_BANKS["general"][1])


async def _gen(client, base_url, model, system, user):
    t0 = time.monotonic()
    try:
        out = await asyncio.wait_for(
            generate(client, base_url, model, system, user, max_tokens=512, max_retries=0),
            timeout=_PER_CALL_TIMEOUT,
        )
        return out, (time.monotonic() - t0) * 1000.0, None
    except Exception as e:  # noqa: BLE001 — a bad candidate must not stall the run
        return None, None, str(e)[:120]


async def _score_objective(client, base_url, model, task_type):
    system, items = OBJECTIVE_BANKS[task_type]
    correct = 0
    lats = []
    failures = 0
    for user, check in items:
        out, lat, err = await _gen(client, base_url, model, system, user)
        if err is not None:
            failures += 1
            continue
        lats.append(lat)
        if check(out):
            correct += 1
    total = len(items)
    pass_rate = correct / total if total else 0.0
    return pass_rate, pass_rate, (sum(lats) / len(lats) if lats else 0.0), total, failures


async def _score_rubric(client, base_url, model, task_type):
    system, prompts = RUBRIC_BANKS[task_type]
    scores = []
    lats = []
    failures = 0
    for prompt in prompts:
        out, lat, err = await _gen(client, base_url, model, system, prompt)
        if err is not None:
            failures += 1
            continue
        # The judge call needs the same timeout/graceful-failure guard as the candidate
        # call — a slow or erroring judge model must fail this sample, not the whole run.
        try:
            jr = await asyncio.wait_for(
                judge(client, base_url, registry.JUDGE_MODEL, _RUBRIC_RUBRIC, prompt, out),
                timeout=_PER_CALL_TIMEOUT,
            )
        except Exception:  # noqa: BLE001 — a bad judge response fails this sample, not the run
            failures += 1
            continue
        lats.append(lat)
        scores.append(max(0.0, min(1.0, jr.score)))
    total = len(prompts)
    quality = sum(scores) / len(scores) if scores else 0.0
    # For rubric banks, quality equals pass_rate (average judge score; no binary hit-rate for open-ended tasks)
    return quality, quality, (sum(lats) / len(lats) if lats else 0.0), total, failures


async def run_benchmark(client, base_url, task_type, candidates, on_progress=None):
    """Benchmark each candidate model on `task_type`; return one result dict per model.
    A failing candidate yields a row with failures>0 (never stalls the run)."""
    results = []
    total = len(candidates)
    for i, model in enumerate(candidates, start=1):
        if task_type in RUBRIC_BANKS:
            quality, pass_rate, lat, samples, failures = await _score_rubric(client, base_url, model, task_type)
        else:
            tt = task_type if task_type in OBJECTIVE_BANKS else "general"
            quality, pass_rate, lat, samples, failures = await _score_objective(client, base_url, model, tt)
        results.append({
            "model": model, "task_type": task_type,
            "quality": round(quality, 4), "pass_rate": round(pass_rate, 4),
            "avg_latency_ms": round(lat, 1), "samples": samples, "failures": failures,
        })
        if on_progress:
            on_progress({"model": model, "done": i, "total": total})
    return results
