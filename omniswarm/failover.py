"""Automatic failover: when a pinned model keeps failing at the gateway, swap every
slot that uses it (task types, judge, council chair, members) to the top-ranked
healthy alternative — through the same runtime-override path the Model Picker uses,
so the swap is persisted, applied live, and visible/undoable in the Control Panel.

Scope (v1, by design): in-memory per-process counters (single-worker deployments);
no silent auto-restore — the user restores via the picker or a fresh benchmark.
The failure counter counts per-attempt gateway calls (adapters retry 3x per request,
so the default threshold of 6 is roughly two fully-failed requests).
"""
import logging
import os
import time

from omniswarm import recommend

log = logging.getLogger("omniswarm.failover")

DEFAULT_THRESHOLD = 6      # per-attempt sink records; ≈2 failed requests
COOLDOWN_SECONDS = 600.0   # a tripped model is not eligible as a replacement for this long


class FailoverTracker:
    """Consecutive-failure counter per model. `record` returns True exactly once
    when a model crosses the threshold (until `done` is called for it)."""

    def __init__(self, threshold: int | None = None, cooldown: float = COOLDOWN_SECONDS):
        if threshold is None:
            threshold = int(os.environ.get("OMNISWARM_FAILOVER_THRESHOLD", str(DEFAULT_THRESHOLD)))
        self.threshold = threshold
        self.cooldown = cooldown
        self._fails: dict[str, int] = {}
        self._tripped_at: dict[str, float] = {}
        self._in_flight: set[str] = set()

    def record(self, model: str, ok: bool) -> bool:
        if self.threshold <= 0:
            return False
        if ok:
            self._fails[model] = 0
            return False
        n = self._fails.get(model, 0) + 1
        self._fails[model] = n
        if n >= self.threshold and model not in self._in_flight:
            self._in_flight.add(model)
            self._tripped_at[model] = time.time()
            return True
        return False

    def unhealthy(self) -> set[str]:
        """Models currently failing or recently tripped — not eligible as replacements."""
        now = time.time()
        bad = {m for m, n in self._fails.items() if n > 0}
        bad |= {m for m, ts in self._tripped_at.items() if now - ts < self.cooldown}
        return bad

    def done(self, model: str) -> None:
        self._in_flight.discard(model)


def affected_slots(model: str) -> dict:
    """Which slots are currently pinned to `model` (reads the live registry)."""
    from omniswarm import registry
    return {
        "models": [name for name, t in registry.REGISTRY.items() if t.model == model],
        "judge": registry.JUDGE_MODEL == model,
        "synth": registry.COUNCIL_SYNTH_MODEL == model,
        "members": [m["role"] for m in registry.COUNCIL_MEMBERS if m["model"] == model],
    }


def choose_replacement(slot: str, catalog: list[dict], reliability: dict,
                       quality: dict, exclude: set[str]) -> str | None:
    """Top-ranked candidate for `slot` that isn't excluded (failing/cooling-down/dead)."""
    ranked = recommend.recommend(slot, catalog, reliability, quality)["ranked"]
    for cand in ranked:
        if cand["id"] not in exclude:
            return cand["id"]
    return None


async def execute(app, model: str) -> dict | None:
    """Perform the failover for `model` on the given app. Returns the swap map, or None."""
    from omniswarm import catalog as _catalog, events, registry, runtime as _runtime, store as _store
    tracker = app.state.failover
    try:
        slots = affected_slots(model)
        if not (slots["models"] or slots["judge"] or slots["synth"] or slots["members"]):
            return None
        cat = await _catalog.get_cached_catalog(app)
        if not cat:
            log.warning("failover for %s skipped: catalog unavailable", model)
            return None
        rel = _store.model_reliability(app.state.settings.db_path)
        quality = _store.latest_benchmarks(app.state.settings.db_path)
        exclude = {model} | tracker.unhealthy()
        swaps: dict[str, str] = {}
        async with app.state.settings_lock:
            rt = dict(app.state.runtime)
            models_ov = dict(rt.get("models") or {})
            for name in slots["models"]:
                rep = choose_replacement(name, cat, rel, quality, exclude)
                if rep:
                    models_ov[name] = rep
                    swaps[name] = rep
            if slots["judge"]:
                rep = choose_replacement("judge", cat, rel, quality, exclude)
                if rep:
                    rt["judge"] = rep
                    swaps["judge"] = rep
            if slots["synth"]:
                rep = choose_replacement("synth", cat, rel, quality, exclude)
                if rep:
                    rt["synth"] = rep
                    swaps["synth"] = rep
            if slots["members"]:
                rep = choose_replacement("synth", cat, rel, quality, exclude)
                if rep:
                    current = {m["role"]: m["model"] for m in registry.COUNCIL_MEMBERS}
                    for role in slots["members"]:
                        current[role] = rep
                        swaps[f"member:{role}"] = rep
                    rt["members"] = [{"role": r, "model": mm} for r, mm in current.items()]
            if not swaps:
                events.publish({"type": "failover", "from": model, "swaps": {},
                                "note": "no healthy replacement available"})
                log.warning("failover for %s: no healthy replacement available", model)
                return None
            rt["models"] = models_ov
            _runtime.save_runtime(None, rt)
            app.state.runtime = rt
            registry.apply_runtime(rt)
        events.publish({"type": "failover", "from": model, "swaps": swaps,
                        "note": f"auto-failover: {model} kept failing at the gateway"})
        log.warning("auto-failover: %s -> %s", model, swaps)
        return swaps
    except Exception:
        log.exception("auto-failover for %s crashed", model)
        return None
    finally:
        tracker.done(model)
