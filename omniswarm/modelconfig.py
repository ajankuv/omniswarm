"""Optional config loading for model selection.

Reads an `omniswarm.toml` (stdlib tomllib) and merges it over built-in DEFAULTS,
so anyone can point OmniSwarm at their own gateway's model names without editing code.
No file → defaults (today's pinned models), so behavior is unchanged out of the box.
"""
import os
import tomllib

# Models below are chosen from a benchmark (2026-06) across the OmniRoute pool.
# At benchmark time only NVIDIA + Mistral were serving (Groq 403, Gemini 404,
# Cerebras 403, SambaNova 404, ghm 429), so hot paths are load-balanced 3/3
# across those two live providers. nvidia/mistral-large-3 was dropped from every
# hot path (incl. judge) because it collapses under sustained load (0-4/8, timeouts).
# When Groq/Gemini/SambaNova recover, re-add them to council for wider spread.
DEFAULTS: dict = {
    "models": {
        # NVIDIA (llama-4-maverick): 8/8 classify@0.37s, 8/8 general@0.53s, 9.83/10 summarize@0.96s
        "general": "nvidia/meta/llama-4-maverick-17b-128e-instruct",
        "summarize": "nvidia/meta/llama-4-maverick-17b-128e-instruct",
        "classify": "nvidia/meta/llama-4-maverick-17b-128e-instruct",
        # Mistral (spreads load off NVIDIA): mistral-medium 8/8 reasoning@1.59s, 8.33/10 summarize
        "draft": "mistral/mistral-medium-3-5",
        "reasoning": "mistral/mistral-medium-3-5",
        # devstral: code specialist, 7/8 code@2.06s (maverick 8/8 but 6.06s — devstral is purpose-built + Mistral spread)
        "code": "mistral/devstral-latest",
    },
    # Judge: maverick — only model that never failed and is fast (nvmistral collapses under load).
    "judge": "nvidia/meta/llama-4-maverick-17b-128e-instruct",
    # Chair: mistral-medium — reliable Mistral synthesizer, different provider from the maverick judge.
    "synth": "mistral/mistral-medium-3-5",
    "council_generators": [
        "nvidia/meta/llama-4-maverick-17b-128e-instruct",
        "mistral/mistral-medium-3-5",
        "nvidia/minimaxai/minimax-m3",
    ],
    "members": [
        {"role": "Security Engineer", "model": "mistral/devstral-latest",
         "focus": "known CVEs in named dependencies/versions, injection (SQL/command/path), secret or "
                  "credential leakage, SSRF, unsafe deserialization, and other exploitable or harmful code"},
        {"role": "Code Auditor", "model": "mistral/codestral-latest",
         "focus": "code correctness, edge cases, efficiency, and whether the logic actually works"},
        {"role": "Clarity Editor", "model": "nvidia/meta/llama-4-maverick-17b-128e-instruct",
         "focus": "readability, coherence, structure, and logical flow of the explanation"},
        {"role": "Fact-Checker", "model": "mistral/mistral-medium-3-5",
         "focus": "factual accuracy — cross-check claims against well-known facts and flag unsupported ones"},
        {"role": "Safety Sentinel", "model": "nvidia/meta/llama-4-maverick-17b-128e-instruct",
         "focus": "harmful, biased, unsafe, or ethically questionable content, including edge cases"},
        {"role": "Data Guardian", "model": "nvidia/minimaxai/minimax-m3",
         "focus": "statistical claims, numerical accuracy, data interpretation, and privacy/data-leak risks"},
        {"role": "User Advocate", "model": "mistral/mistral-medium-3-5",
         "focus": "whether the answer meets the user's intent, is practical and actionable, and avoids misuse"},
    ],
    "default_roster": ["Fact-Checker", "Clarity Editor", "User Advocate"],
    "roster_overrides": {
        "code": ["Security Engineer", "Code Auditor", "User Advocate"],
        "reasoning": ["Fact-Checker", "Data Guardian", "Clarity Editor"],
    },
}


def _discover() -> str | None:
    env = os.environ.get("OMNISWARM_CONFIG")
    if env:
        return env if os.path.exists(env) else None
    return "omniswarm.toml" if os.path.exists("omniswarm.toml") else None


def load_model_config(path: str | None = None) -> dict:
    cfg = {
        "models": dict(DEFAULTS["models"]),
        "judge": DEFAULTS["judge"],
        "synth": DEFAULTS["synth"],
        "council_generators": list(DEFAULTS["council_generators"]),
        "members": [dict(m) for m in DEFAULTS["members"]],
        "default_roster": list(DEFAULTS["default_roster"]),
        "roster_overrides": {k: list(v) for k, v in DEFAULTS["roster_overrides"].items()},
    }
    path = path or _discover()
    if not path or not os.path.exists(path):
        return cfg
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if isinstance(data.get("models"), dict):
        cfg["models"].update(data["models"])
    if isinstance(data.get("judge"), dict) and "model" in data["judge"]:
        cfg["judge"] = data["judge"]["model"]
    council = data.get("council")
    if isinstance(council, dict):
        if "synth_model" in council:
            cfg["synth"] = council["synth_model"]
        if "generators" in council:
            cfg["council_generators"] = list(council["generators"])
        members = council.get("members")
        if isinstance(members, list):
            cleaned = [
                {"role": m["role"], "model": m["model"], "focus": m.get("focus", "")}
                for m in members
                if isinstance(m, dict) and "role" in m and "model" in m
            ]
            if cleaned:
                cfg["members"] = cleaned
                cfg["default_roster"] = [m["role"] for m in cleaned]  # default to all members
        if isinstance(council.get("default_roster"), list):
            cfg["default_roster"] = list(council["default_roster"])
        if isinstance(council.get("roster_overrides"), dict):
            cfg["roster_overrides"] = {k: list(v) for k, v in council["roster_overrides"].items()}
    return cfg
