"""Runtime-mutable settings (editable from the Control Panel), stored as JSON."""
import json
import os

DEFAULTS = {"api_token": "", "rate_limit_per_min": 0, "store_mode": "full",
            "active_roster": [], "always_council": True,
            "models": {}, "judge": "", "synth": "", "members": []}
_VALID_MODES = {"full", "redact", "none"}


def _discover(path: str | None) -> str:
    return path or os.environ.get("OMNISWARM_RUNTIME") or "omniswarm.runtime.json"


def load_runtime(path: str | None = None) -> dict:
    cfg = {**DEFAULTS, "active_roster": list(DEFAULTS["active_roster"]),
           "models": {}, "judge": "", "synth": "", "members": []}
    p = _discover(path)
    if p and os.path.exists(p):
        try:
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, dict):
                if isinstance(data.get("api_token"), str):
                    cfg["api_token"] = data["api_token"]
                if isinstance(data.get("rate_limit_per_min"), int):
                    cfg["rate_limit_per_min"] = max(0, data["rate_limit_per_min"])
                if data.get("store_mode") in _VALID_MODES:
                    cfg["store_mode"] = data["store_mode"]
                if isinstance(data.get("active_roster"), list):
                    cfg["active_roster"] = [str(r) for r in data["active_roster"]]
                if isinstance(data.get("always_council"), bool):
                    cfg["always_council"] = data["always_council"]
                if isinstance(data.get("models"), dict):
                    cfg["models"] = {str(k): str(v) for k, v in data["models"].items() if isinstance(v, str)}
                if isinstance(data.get("judge"), str):
                    cfg["judge"] = data["judge"]
                if isinstance(data.get("synth"), str):
                    cfg["synth"] = data["synth"]
                if isinstance(data.get("members"), list):
                    cfg["members"] = [
                        {"role": str(m["role"]), "model": str(m["model"])}
                        for m in data["members"]
                        if isinstance(m, dict) and "role" in m and "model" in m
                    ]
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save_runtime(path: str | None, data: dict) -> None:
    p = _discover(path)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
