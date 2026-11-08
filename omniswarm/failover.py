"""Automatic failover: when a pinned model keeps failing at the gateway, swap every
slot that uses it (task types, judge, council chair, members) to the top-ranked
healthy alternative — through the same runtime-override path the Model Picker uses,
so the swap is persisted, applied live, and visible/undoable in the Control Panel.

Scope (v1, by design): in-memory per-process counters (single-worker deployments);
no silent auto-restore — the user restores via the picker or a fresh benchmark.
Each moved slot remembers its original ("home") model in the runtime, and /stats
reports slots that are off home or otherwise degraded, so drift is visible.
The failure counter counts per-attempt gateway calls (adapters retry 3x per request,
so the default threshold of 6 is roughly two fully-failed requests).
"""
import logging
import os
import time

from omniswarm import recommend
from omniswarm.adapters import _is_permanent, is_exhaustion, is_unavailable, status_code

log = logging.getLogger("omniswarm.failover")

DEFAULT_THRESHOLD = 6      # per-attempt sink records; ≈2 failed requests
COOLDOWN_SECONDS = 600.0   # a tripped model is not eligible as a replacement for this long

# T1.2 — provider circuit breaker
PROVIDER_FAIL_THRESHOLD = int(os.environ.get("OMNISWARM_PROVIDER_BREAKER_THRESHOLD", "4"))
PROVIDER_COOLDOWN = float(os.environ.get("OMNISWARM_PROVIDER_BREAKER_COOLDOWN", str(30 * 60)))
# a provider that CAN'T RUN here (missing CLI/browser, unconfigured transport) is benched
# much longer: long enough to stop burning calls on it, short enough that a misread or
# a later fix clears itself without a restart.
PROVIDER_UNAVAILABLE_COOLDOWN = float(os.environ.get("OMNISWARM_PROVIDER_UNAVAILABLE_COOLDOWN",
                                                     str(12 * 3600)))
# gone-model codes: they count against the one model, not its whole provider
_MODEL_ONLY_HTTP = frozenset({404, 410})

MAX_PROBES = 3             # unproven candidates probed per slot before giving up


def _provider_of(model: str) -> str:
    return model.split("/", 1)[0]


class ProviderBreaker:
    """Trips a breaker for a whole PROVIDER when its models keep returning quota/auth
    errors (401/402/403/429), so routing stops hammering an exhausted provider instead
    of rediscovering it dead model-by-model. A "can't run here" response (UNAVAILABLE)
    trips it at once, for the longer unavailable cooldown. Gone-model codes (404/410)
    without an explicit exhaustion signal count against that MODEL only, so one retired
    model doesn't bench a provider that still serves the rest. Feeds the failover
    exclude set and the dashboard. Counters are in-memory (single-worker)."""

    def __init__(self, threshold: int | None = None, cooldown: float | None = None,
                 unavailable_cooldown: float | None = None):
        self.threshold = PROVIDER_FAIL_THRESHOLD if threshold is None else threshold
        self.cooldown = PROVIDER_COOLDOWN if cooldown is None else cooldown
        self.unavailable_cooldown = (PROVIDER_UNAVAILABLE_COOLDOWN if unavailable_cooldown is None
                                     else unavailable_cooldown)
        self._perm_fails: dict[str, int] = {}       # consecutive quota/auth fails
        self._last_fail_at: dict[str, float] = {}   # when the last one landed
        self._unavailable_at: dict[str, float] = {}  # provider said it can't run here
        self._model_fails: dict[str, int] = {}      # consecutive 404/410 per model
        self._model_last_fail_at: dict[str, float] = {}

    def record(self, model: str, ok: bool, status: str) -> None:
        if self.threshold <= 0:
            return
        p = _provider_of(model)
        if ok:
            self._perm_fails[p] = 0                  # a success clears the provider
            self._unavailable_at.pop(p, None)
            self._model_fails.pop(model, None)
            return
        # only quota/auth/gone/can't-run counts (transient 5xx doesn't)
        if not _is_permanent(status):
            return
        now = time.time()
        was_exhausted = self._is_exhausted(p, now)
        if is_unavailable(status):
            # the provider can't run at all — trip now, bench it for the long cooldown
            self._unavailable_at[p] = now
            self._perm_fails[p] = max(self._perm_fails.get(p, 0) + 1, self.threshold)
            self._last_fail_at[p] = now
            if not was_exhausted:
                log.warning("provider breaker TRIPPED: %s can't run here (%s), benched for %ds",
                            p, status, int(self.unavailable_cooldown))
            return
        if status_code(status) in _MODEL_ONLY_HTTP and not is_exhaustion(status):
            n = self._model_fails.get(model, 0) + 1
            self._model_fails[model] = n
            self._model_last_fail_at[model] = now
            if n == self.threshold:
                log.warning("model breaker TRIPPED: %s gone (%d x %s)", model, n, status)
            return
        # an EXPLICIT exhaustion signal from the gateway ("no active credentials",
        # rate_limit…) is definitive — jump straight to the threshold and trip now,
        # don't wait to count blind HTTP codes. Ambiguous codes still accrue.
        bumped = self._perm_fails.get(p, 0) + 1
        self._perm_fails[p] = max(bumped, self.threshold) if is_exhaustion(status) else bumped
        self._last_fail_at[p] = now
        if not was_exhausted and self._is_exhausted(p, now):
            log.warning("provider breaker TRIPPED: %s exhausted (%s)", p,
                        "explicit signal" if is_exhaustion(status) else f"{self._perm_fails[p]} errors")

    def _is_unavailable(self, p: str, now: float) -> bool:
        at = self._unavailable_at.get(p)
        return at is not None and (now - at) < self.unavailable_cooldown

    def _is_exhausted(self, p: str, now: float) -> bool:
        # exhausted while it has >= threshold recent permanent fails. If it stops
        # failing for `cooldown` (or a call succeeds → count reset), it clears — and
        # it can trip AGAIN later, unlike a one-shot latch.
        if self._is_unavailable(p, now):
            return True
        return (self._perm_fails.get(p, 0) >= self.threshold
                and (now - self._last_fail_at.get(p, 0)) < self.cooldown)

    def exhausted(self) -> set[str]:
        now = time.time()
        return {p for p in self._perm_fails if self._is_exhausted(p, now)}

    def gone_models(self) -> set[str]:
        """Single models benched by repeated 404/410 (their provider stays usable)."""
        now = time.time()
        return {m for m, n in self._model_fails.items()
                if n >= self.threshold and (now - self._model_last_fail_at.get(m, 0)) < self.cooldown}

    def exhausted_models(self, catalog: list[dict]) -> set[str]:
        ex = self.exhausted()
        return {m["id"] for m in catalog if _provider_of(m["id"]) in ex} | self.gone_models()

    def state(self) -> dict:
        now = time.time()
        out = {}
        for p in self._perm_fails:
            last = self._last_fail_at.get(p)
            out[p] = {"exhausted": self._is_exhausted(p, now),
                      "unavailable": self._is_unavailable(p, now),
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


def current_pins() -> list[tuple[str, str]]:
    """Every slot and the model it's pinned to now, keyed like failover swaps:
    task-type name, "judge", "synth", "member:<role>"."""
    from omniswarm import registry
    pins = [(name, t.model) for name, t in registry.REGISTRY.items()]
    pins += [("judge", registry.JUDGE_MODEL), ("synth", registry.COUNCIL_SYNTH_MODEL)]
    pins += [(f"member:{m['role']}", m["model"]) for m in registry.COUNCIL_MEMBERS]
    return pins


def degraded(runtime: dict, catalog: list[dict], exhausted_providers: set[str],
             allowed: frozenset[str]) -> list[dict]:
    """Slots that aren't where they should be, one entry per slot with every reason:
    moved off their home model by auto-failover, pinned to a model the gateway no
    longer lists, on a breaker-exhausted provider, or on a provider not allowed.
    An empty catalog (gateway unreachable) skips the not-listed check."""
    from omniswarm.catalog import provider_allowed
    home = runtime.get("failover_home") or {}
    cat_ids = {m["id"] for m in catalog}
    out = []
    for slot, mid in current_pins():
        reasons = []
        h = home.get(slot)
        if h and h != mid:
            reasons.append(f"off its home model {h} (auto-failover)")
        if not provider_allowed(mid, allowed):
            reasons.append("provider not allowed (OMNISWARM_ALLOWED_PROVIDERS)")
        elif cat_ids and mid not in cat_ids:
            reasons.append("not in the gateway catalog")
        if _provider_of(mid) in exhausted_providers:
            reasons.append("provider exhausted (circuit breaker)")
        if reasons:
            out.append({"slot": slot, "model": mid, "reason": "; ".join(reasons)})
    return out


def _proven_healthy(cid: str, reliability: dict) -> bool:
    """A model with real, mostly-successful recent history — safe to fail over TO."""
    r = reliability.get(cid)
    return bool(r and r.get("calls", 0) >= 3 and r.get("success_pct", 0.0) >= 70.0)


def _zero_success(cid: str, reliability: dict) -> bool:
    """Tried at least once and never succeeded — only usable if a probe passes."""
    r = reliability.get(cid)
    if not r or r.get("calls", 0) < 1:
        return False
    oks = r.get("ok")
    return oks == 0 if oks is not None else r.get("success_pct", 0.0) == 0.0


async def choose_replacement(slot: str, catalog: list[dict], reliability: dict,
                             quality: dict, exclude: set[str], *, probe=None,
                             probe_results: dict | None = None, picked: dict | None = None,
                             avoid_providers: frozenset[str] | set[str] = frozenset(),
                             max_probes: int = MAX_PROBES) -> str | None:
    """Pick a replacement for `slot`, excluding failing/cooling-down/dead models.

    Auto-failover must land on something KNOWN-GOOD. The recommender ranks a shiny
    zero-history model on capability alone, so falling back to its raw ranking swapped
    every slot onto gateway providers that can't run at all (the 2026-11-08 incident).
    Order:
      1. the best-ranked PROVEN-healthy candidate, preferring a provider not in
         `avoid_providers` and a model not already `picked` for another slot;
      2. unproven candidates in rank order (never-tried first, zero-success last),
         used only once `probe(id)` passes; at most `max_probes` probes, results
         cached in `probe_results` across slots of one failover;
      3. reuse of an already-picked model (the least-picked first) over no swap.
    No probe function means unproven candidates are never used. `exclude` is checked
    again before each probe, so the caller can grow it while probing (a failed probe
    can trip a provider breaker, which benches that provider's other models)."""
    picked = picked or {}
    results = {} if probe_results is None else probe_results
    pool = [m for m in catalog if m["id"] not in exclude]
    ranked = [c["id"] for c in recommend.recommend(slot, pool, reliability, quality, limit=None)["ranked"]]
    rank = {cid: i for i, cid in enumerate(ranked)}
    proven = [c for c in ranked if _proven_healthy(c, reliability)]
    unproven = ([c for c in ranked if c not in proven and not _zero_success(c, reliability)]
                + [c for c in ranked if _zero_success(c, reliability)])

    def _fresh_diverse(ids: list[str]) -> list[str]:
        fresh = [c for c in ids if c not in picked]
        other = [c for c in fresh if _provider_of(c) not in avoid_providers]
        return other + [c for c in fresh if c not in other]

    for c in _fresh_diverse(proven):
        return c
    probes = 0
    for c in _fresh_diverse(unproven):
        if c in exclude:
            continue
        if c not in results:
            if probe is None or probes >= max_probes:
                break
            probes += 1
            try:
                results[c] = bool(await probe(c))
            except Exception:
                log.exception("failover probe of %s crashed", c)
                results[c] = False
        if results[c] and c not in exclude:
            return c
    reuse = [c for c in picked if c in rank and c not in exclude
             and (c in proven or results.get(c) is True)]
    if reuse:
        return min(reuse, key=lambda c: (picked[c], c not in proven, rank[c]))
    return None


def _gateway_probe(app):
    """The real probe: one tiny no-retry call through the normal gateway path."""
    from omniswarm import adapters

    async def _probe(cid: str) -> bool:
        ok, err = await adapters.probe(app.state.client, app.state.settings.omniroute_base_url, cid)
        if not ok:
            log.warning("failover probe: %s failed (%s)", cid, err[:160])
        return ok
    return _probe


async def execute(app, model: str, probe=None) -> dict | None:
    """Perform the failover for `model` on the given app. Returns the swap map, or None.

    Replacements are picked and probed FIRST, without the settings lock, so a slow
    probe never blocks a POST /settings. Then, under the lock, only slots still pinned
    to `model` are swapped, so an operator's change made meanwhile is never undone.
    `probe` (async id -> bool) defaults to a real gateway call; tests inject a fake."""
    from omniswarm import catalog as _catalog, events, registry, runtime as _runtime, store as _store
    tracker = app.state.failover
    try:
        slots = affected_slots(model)
        if not (slots["models"] or slots["judge"] or slots["synth"] or slots["members"]):
            return None
        allowed = getattr(app.state.settings, "allowed_providers", frozenset())
        cat = _catalog.filter_allowed(await _catalog.get_cached_catalog(app), allowed)
        if not cat:
            log.warning("failover for %s skipped: catalog unavailable", model)
            return None
        rel = _store.model_reliability(app.state.settings.db_path)
        quality = _store.latest_benchmarks(app.state.settings.db_path)
        # exclude failing/cooling-down models AND every model of an exhausted provider
        # (breaker), so we never swap onto a provider that's out of quota or can't run.
        exclude = {model} | tracker.unhealthy()
        breaker = getattr(app.state, "provider_breaker", None)
        if breaker is not None:
            exclude |= breaker.exhausted_models(cat)
        probe = probe or _gateway_probe(app)

        async def _probe(cid: str) -> bool:
            ok = await probe(cid)
            if not ok and breaker is not None:
                exclude.update(breaker.exhausted_models(cat))   # a failed probe may bench a provider
            return ok

        # 1) choose + probe, no lock held. Each slot gets its own pick: earlier picks
        #    are soft-excluded and their providers avoided, so one quota burst can't
        #    take every slot down again.
        wanted = [(name, name) for name in slots["models"]]
        if slots["judge"]:
            wanted.append(("judge", "judge"))
        if slots["synth"]:
            wanted.append(("synth", "synth"))
        wanted += [(f"member:{role}", "synth") for role in slots["members"]]
        picks: dict[str, str] = {}
        picked: dict[str, int] = {}
        probe_results: dict[str, bool] = {}
        for key, rank_slot in wanted:
            avoid = {_provider_of(model)} | {_provider_of(r) for r in picked}
            rep = await choose_replacement(rank_slot, cat, rel, quality, exclude, probe=_probe,
                                           probe_results=probe_results, picked=picked,
                                           avoid_providers=avoid)
            if rep:
                picks[key] = rep
                picked[rep] = picked.get(rep, 0) + 1
        if not picks:
            events.publish({"type": "failover", "from": model, "swaps": {},
                            "note": "no healthy replacement available"})
            log.warning("failover for %s: no healthy replacement available", model)
            return None

        # 2) apply under the lock, re-reading which slots are STILL on `model`
        swaps: dict[str, str] = {}
        async with app.state.settings_lock:
            now = affected_slots(model)
            rt = dict(app.state.runtime)
            models_ov = dict(rt.get("models") or {})
            home = dict(rt.get("failover_home") or {})

            def _swap(key: str, rep: str) -> None:
                swaps[key] = rep
                home.setdefault(key, model)          # chained hops keep the TRUE original
                if home[key] == rep:
                    home.pop(key)                    # landed back home: not degraded

            for name in now["models"]:
                if name in picks:
                    models_ov[name] = picks[name]
                    _swap(name, picks[name])
            if now["judge"] and "judge" in picks:
                rt["judge"] = picks["judge"]
                _swap("judge", picks["judge"])
            if now["synth"] and "synth" in picks:
                rt["synth"] = picks["synth"]
                _swap("synth", picks["synth"])
            moved = {role: picks[f"member:{role}"] for role in now["members"] if f"member:{role}" in picks}
            if moved:
                current = {m["role"]: m["model"] for m in registry.COUNCIL_MEMBERS}
                for role, rep in moved.items():
                    current[role] = rep
                    _swap(f"member:{role}", rep)
                rt["members"] = [{"role": r, "model": mm} for r, mm in current.items()]
            if not swaps:
                log.warning("failover for %s: slots were changed meanwhile, nothing to swap", model)
                return None
            rt["models"] = models_ov
            rt["failover_home"] = home
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
