import time as _time

from omniswarm import recommend

CATALOG = [
    {"id": "nvidia/meta/llama-4-maverick-17b-128e-instruct", "provider": "nvidia",
     "name": "Llama 4 Maverick", "context_length": 128000, "max_output_tokens": 8192,
     "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
    {"id": "mistral/mistral-medium-3-5", "provider": "mistral", "name": "Mistral Medium",
     "context_length": 128000, "max_output_tokens": 8192,
     "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
    {"id": "mistral/devstral-latest", "provider": "mistral", "name": "Devstral",
     "context_length": 256000, "max_output_tokens": 8192,
     "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
    {"id": "nvidia/deepseek-ai/deepseek-r1", "provider": "nvidia", "name": "DeepSeek R1",
     "context_length": 128000, "max_output_tokens": 8192,
     "capabilities": {"tool_calling": False, "reasoning": True, "thinking": True}, "chat_capable": True},
    {"id": "gemini/gemini-3-flash-preview", "provider": "gemini", "name": "Gemini Flash",
     "context_length": 1000000, "max_output_tokens": 8192,
     "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
]
REL = {
    "nvidia/meta/llama-4-maverick-17b-128e-instruct": {"calls": 42, "ok": 42, "fail": 0, "success_pct": 100.0, "avg_latency_ms": 500.0, "http_429": 0},
    "mistral/mistral-medium-3-5": {"calls": 24, "ok": 24, "fail": 0, "success_pct": 100.0, "avg_latency_ms": 1100.0, "http_429": 0},
    "mistral/devstral-latest": {"calls": 16, "ok": 16, "fail": 0, "success_pct": 100.0, "avg_latency_ms": 2500.0, "http_429": 0},
    "gemini/gemini-3-flash-preview": {"calls": 12, "ok": 0, "fail": 12, "success_pct": 0.0, "avg_latency_ms": 17.0, "http_429": 0},
}


def test_code_prefers_coder_family():
    r = recommend.recommend("code", CATALOG, REL)
    assert r["recommended"] == "mistral/devstral-latest"


def test_reasoning_prefers_thinking_model_but_only_if_reliable():
    # r1 is the only reasoning-capable model; it has no failures recorded -> recommended
    r = recommend.recommend("reasoning", CATALOG, REL)
    assert r["recommended"] == "nvidia/deepseek-ai/deepseek-r1"


def test_classify_penalizes_thinking_and_picks_fast_reliable():
    r = recommend.recommend("classify", CATALOG, REL)
    # fastest 100%-reliable non-thinking model
    assert r["recommended"] == "nvidia/meta/llama-4-maverick-17b-128e-instruct"
    assert "100%" in r["why"]


def test_dead_model_never_recommended():
    r = recommend.recommend("general", CATALOG, REL)
    assert r["recommended"] != "gemini/gemini-3-flash-preview"
    ids = [x["id"] for x in r["ranked"]]
    # dead model may appear ranked low, but not first
    assert ids[0] != "gemini/gemini-3-flash-preview"
    # dead model (0% success over 12 calls) is ranked last
    assert ids[-1] == "gemini/gemini-3-flash-preview"


def test_judge_excludes_reasoning_only():
    r = recommend.recommend("judge", CATALOG, REL)
    assert r["recommended"] != "nvidia/deepseek-ai/deepseek-r1"  # thinking-only excluded from judge


def test_synth_excludes_reasoning_only():
    r = recommend.recommend("synth", CATALOG, REL)
    assert r["recommended"] != "nvidia/deepseek-ai/deepseek-r1"  # thinking-only excluded from synth


def test_no_history_model_ranks_on_capability_not_penalized():
    cat = [{"id": "nvidia/new-coder", "provider": "nvidia", "name": "New Coder",
            "context_length": 64000, "max_output_tokens": 8192,
            "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True}]
    r = recommend.recommend("code", cat, {})
    assert r["recommended"] == "nvidia/new-coder"
    assert "no usage history" in r["why"]


def test_deterministic_output():
    a = recommend.recommend("general", CATALOG, REL)
    b = recommend.recommend("general", CATALOG, REL)
    assert a == b


def test_dead_model_with_fresh_benchmark_still_not_recommended():
    # Safety invariant: a proven-dead model (0% over >=3 calls) must never be
    # recommended even if it has a fresh, perfect benchmark — dead penalty wins.
    quality = {("gemini/gemini-3-flash-preview", "general"): {
        "quality": 1.0, "pass_rate": 1.0, "avg_latency_ms": 10.0, "samples": 8, "ts": _time.time()}}
    r = recommend.recommend("general", CATALOG, REL, quality)
    assert r["recommended"] != "gemini/gemini-3-flash-preview"


def test_fresh_quality_dominates_capability():
    # 'weak' has better capability score for 'general' but 'strong' has a fresh top benchmark
    cat = [
        {"id": "prov/weak", "provider": "prov", "name": "Weak", "context_length": 128000,
         "max_output_tokens": 8192, "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
        {"id": "prov/strong", "provider": "prov", "name": "Strong", "context_length": 8000,
         "max_output_tokens": 8192, "capabilities": {"tool_calling": False, "reasoning": False, "thinking": False}, "chat_capable": True},
    ]
    quality = {("prov/strong", "general"): {"quality": 1.0, "pass_rate": 1.0,
               "avg_latency_ms": 500.0, "samples": 8, "ts": _time.time()}}
    r = recommend.recommend("general", cat, {}, quality)
    assert r["recommended"] == "prov/strong"
    assert "benchmark" in r["why"].lower()


def test_stale_quality_does_not_dominate():
    cat = [
        {"id": "prov/weak", "provider": "prov", "name": "Weak", "context_length": 128000,
         "max_output_tokens": 8192, "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
        {"id": "prov/strong", "provider": "prov", "name": "Strong", "context_length": 8000,
         "max_output_tokens": 8192, "capabilities": {"tool_calling": False, "reasoning": False, "thinking": False}, "chat_capable": True},
    ]
    old = _time.time() - (8 * 86400)  # older than freshness window
    quality = {("prov/strong", "general"): {"quality": 1.0, "pass_rate": 1.0,
               "avg_latency_ms": 500.0, "samples": 8, "ts": old}}
    r = recommend.recommend("general", cat, {}, quality)
    assert r["recommended"] == "prov/weak"  # stale benchmark must not dominate


def test_no_quality_arg_is_backward_compatible():
    r = recommend.recommend("general", CATALOG, REL)  # existing fixtures, no quality
    assert "recommended" in r and "ranked" in r


def test_judge_slot_uses_general_benchmark_as_proxy():
    cat = [
        {"id": "prov/a", "provider": "prov", "name": "A", "context_length": 8000, "max_output_tokens": 8192,
         "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
        {"id": "prov/b", "provider": "prov", "name": "B", "context_length": 8000, "max_output_tokens": 8192,
         "capabilities": {"tool_calling": True, "reasoning": False, "thinking": False}, "chat_capable": True},
    ]
    # a fresh 'general' benchmark for prov/b should sway the JUDGE recommendation (proxy)
    quality = {("prov/b", "general"): {"quality": 1.0, "pass_rate": 1.0,
               "avg_latency_ms": 300.0, "samples": 8, "ts": _time.time()}}
    r = recommend.recommend("judge", cat, {}, quality)
    assert r["recommended"] == "prov/b"
    assert "benchmark" in r["why"].lower()
