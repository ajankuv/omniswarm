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
from omniswarm.adapters import _is_permanent, is_exhaustion

log = logging.getLogger("omniswarm.failover")

DEFAULT_THRESHOLD = 6      # per-attempt sink records; ≈2 failed requests
COOLDOWN_SECONDS = 600.0   # a tripped model is not eligible as a replacement for this long

# T1.2 — provider circuit breaker
PROVIDER_FAIL_THRESHOLD = int(os.environ.get("OMNISWARM_PROVIDER_BREAKER_THRESHOLD", "4"))
PROVIDER_COOLDOWN = float(os.environ.get("OMNISWARM_PROVIDER_BREAKER_COOLDOWN", str(30 * 60)))


def _provider_of(model: str) -> str:
    return model.split("/", 1)[0]


class ProviderBreaker:
    """Trips a breaker for a whole PROVIDER when its models keep returning quota/auth/
    gone errors (403/410/404/429), so routing stops hammering an exhausted provider
    instead of rediscovering it dead model-by-model. Feeds the failover exclude set
    and the dashboard. Counters are per-provider, in-memory (single-worker)."""

    def __init__(self, threshold: int | None = None, cooldown: float | None = None):
        self.threshold = PROVIDER_FAIL_THRESHOLD if threshold is None else threshold
        self.cooldown = PROVIDER_COOLDOWN if cooldown is None else cooldown
        self._perm_fails: dict[str, int] = {}       # consecutive quota/auth fails
        self._last_fail_at: dict[str, float] = {}   # when the last one landed

    def record(self, model: str, ok: bool, status: str) -> None:
        if self.threshold <= 0:
            return
        p = _provider_of(model)
        if ok:
            self._perm_fails[p] = 0                  # a success clears the provider
            return
        # only quota/auth/gone counts toward tripping a provider (transient 5xx doesn't)
        if _is_permanent(status):
            now = time.time()
            was_exhausted = self._is_exhausted(p, now)
            # an EXPLICIT exhaustion signal from the gateway ("no active credentials",
            # rate_limit…) is definitive — jump straight to the threshold and trip now,
            # don't wait to count blind HTTP codes. Ambiguous codes still accrue.
            bumped = self._perm_fails.get(p, 0) + 1
            self._perm_fails[p] = max(bumped, self.threshold) if is_exhaustion(status) else bumped
            self._last_fail_at[p] = now
            if not was_exhausted and self._is_exhausted(p, now):
                log.warning("provider breaker TRIPPED: %s exhausted (%s)", p,
                            "explicit signal" if is_exhaustion(status) else f"{self._perm_fails[p]} errors")

    def _is_exhausted(self, p: str, now: float) -> bool:
        # exhausted while it has >= threshold recent permanent fails. If it stops
        # failing for `cooldown` (or a call succeeds → count reset), it clears — and
        # it can trip AGAIN later, unlike a one-shot latch.
        return (self._perm_fails.get(p, 0) >= self.threshold
                and (now - self._last_fail_at.get(p, 0)) < self.cooldown)

    def exhausted(self) -> set[str]:
        now = time.time()
        return {p for p in self._perm_fails if self._is_exhausted(p, now)}

    def exhausted_models(self, catalog: list[dict]) -> set[str]:
        ex = self.exhausted()
        return {m["id"] for m in catalog if _provider_of(m["id"]) in ex}

    def state(self) -> dict:
        now = time.time()
        out = {}
        for p in self._perm_fails:
            last = self._last_fail_at.get(p)
            out[p] = {"exhausted": self._is_exhausted(p, now),
                      "perm_fails": self._perm_fails.get(p, 0),
                      "since_s": int(now - last) if last else None}
        return out


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
            # visibility: a success wiping a building failure streak is exactly what a
            # postmortem needs to see (it explains why a failover did NOT fire)
            if self._fails.get(model):
                log.warning("failover counter reset: %s succeeded after %d consecutive failures",
                            model, self._fails[model])
            self._fails[model] = 0
            return False
        n = self._fails.get(model, 0) + 1
        self._fails[model] = n
        if n * 2 >= self.threshold:
            log.warning("failover counter: %s at %d/%d consecutive failures", model, n, self.threshold)
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


def _proven_healthy(cid: str, reliability: dict) -> bool:
    """A model with real, mostly-successful recent history — safe to fail over TO."""
    r = reliability.get(cid)
    return bool(r and r.get("calls", 0) >= 3 and r.get("success_pct", 0.0) >= 70.0)


def choose_replacement(slot: str, catalog: list[dict], reliability: dict,
                       quality: dict, exclude: set[str]) -> str | None:
    """Pick a replacement for `slot`, excluding failing/cooling-down/dead models.

    Auto-failover must land on something KNOWN-GOOD. The recommender ranks a shiny
    zero-history model on capability alone (empirical is neutral, `_is_dead` needs
    3+ calls @0%), so it would happily swap to a brand-new model that's actually
    dead — exactly the quota-burn cascade (gpt-5-mini/o3-mini → 404). So we prefer
    the best-ranked PROVEN-healthy candidate first, and only fall back to the raw
    capability ranking when nothing has a track record yet."""
    ranked = recommend.recommend(slot, catalog, reliability, quality)["ranked"]
    candidates = [c for c in ranked if c["id"] not in exclude]
    for c in candidates:
        if _proven_healthy(c["id"], reliability):
            return c["id"]
    return candidates[0]["id"] if candidates else None


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
        # exclude failing/cooling-down models AND every model of an exhausted provider
        # (breaker), so we never swap onto a provider that's out of quota.
        exclude = {model} | tracker.unhealthy()
        breaker = getattr(app.state, "provider_breaker", None)
        if breaker is not None:
            exclude |= breaker.exhausted_models(cat)
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
