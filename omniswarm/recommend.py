"""Pure per-slot model recommendation over the catalog + empirical reliability.

No I/O. Combines capability-match (from OmniRoute metadata) with your own
success%/latency/429 history (from store.model_reliability). Never recommends a
model that has demonstrably failed, and never recommends a reasoning-only model
for the judge/synth slots.
"""
import re
import time

SLOT_TASK_TYPES = ("general", "summarize", "classify", "draft", "code", "reasoning")
_CODER_RE = re.compile(r"\b(devstral|codestral|coder|code)\b", re.I)
BENCH_FRESH_SECONDS = 7 * 86400


def _capability_score(slot: str, m: dict) -> float:
    caps = m["capabilities"]
    ctx = m.get("context_length", 0)
    s = 0.0
    if slot == "reasoning":
        s += 40 if (caps["reasoning"] or caps["thinking"]) else 0
        s += min(ctx / 1000.0, 20)      # context helps, capped
        s += 5 if caps["tool_calling"] else 0
    elif slot == "code":
        s += 30 if caps["tool_calling"] else 0
        s += 30 if _CODER_RE.search(m["id"]) or _CODER_RE.search(m["name"]) else 0
        s += min(ctx / 2000.0, 15)
    elif slot in ("judge", "synth"):
        # capable general model; reasoning-only is excluded upstream
        s += 25 if caps["tool_calling"] else 10
        s += min(ctx / 4000.0, 15)
        s -= 10 if caps["thinking"] else 0   # judging wants decisive, not deliberating
    else:  # fast slots: classify/summarize/general/draft
        s += 20
        s -= 25 if caps["thinking"] else 0    # penalize slow thinkers
        s += 5 if caps["tool_calling"] else 0
    return s


def _empirical(m_id: str, reliability: dict) -> tuple[float, str]:
    r = reliability.get(m_id)
    if not r or not r.get("calls"):
        return 0.0, "no usage history yet"
    bonus = 0.0
    bonus += (r["success_pct"] - 80.0) * 0.6          # >80% helps, <80% hurts
    bonus -= min(r["avg_latency_ms"] / 500.0, 20.0)   # latency penalty, capped
    bonus -= 15.0 * (1 if r.get("http_429") else 0)   # rate-limited = risky
    calls, pct = r["calls"], r["success_pct"]
    why = f"{pct:g}% success over {calls} call{'s' if calls != 1 else ''} · {r['avg_latency_ms']/1000:.1f}s avg"
    if r.get("http_429"):
        why += f" · {r['http_429']} rate-limit(s)"
    return bonus, why


def _eligible(slot: str, m: dict) -> bool:
    if not m.get("chat_capable", True):
        return False
    if slot in ("judge", "synth"):
        caps = m["capabilities"]
        if caps["thinking"] and not caps["tool_calling"]:
            return False  # reasoning-only: not allowed as judge/synth
    return True


def _is_dead(m_id: str, reliability: dict) -> bool:
    r = reliability.get(m_id)
    return bool(r and r.get("calls", 0) >= 3 and r.get("success_pct", 0.0) == 0.0)


def _capability_why(slot: str, m: dict) -> str:
    caps = m["capabilities"]
    bits = []
    if slot == "reasoning" and (caps["reasoning"] or caps["thinking"]):
        bits.append("reasoning-capable")
    if slot == "code" and (_CODER_RE.search(m["id"]) or _CODER_RE.search(m["name"])):
        bits.append("coder model")
    if caps["tool_calling"]:
        bits.append("tool-calling")
    if m.get("context_length"):
        bits.append(f"{m['context_length'] // 1000}k context")
    return ", ".join(bits) or "general model"


def recommend(slot: str, catalog: list[dict], reliability: dict, quality: dict = None) -> dict:
    scored = []
    for m in catalog:
        if not _eligible(slot, m):
            continue
        cap = _capability_score(slot, m)
        emp, emp_why = _empirical(m["id"], reliability)
        dead = _is_dead(m["id"], reliability)
        # judge/synth aren't benchmarked directly; use the model's 'general' score as a proxy.
        _qslot = "general" if slot in ("judge", "synth") else slot
        q = (quality or {}).get((m["id"], _qslot))
        q_fresh = bool(q and q.get("quality") is not None
                       and (time.time() - q.get("ts", 0)) < BENCH_FRESH_SECONDS)
        if q_fresh:
            # measured quality DOMINATES capability+empirical when fresh
            qbonus = 1000.0 * q["quality"] - min(q.get("avg_latency_ms", 0) / 500.0, 20.0)
        else:
            qbonus = 0.0
        score = cap + emp + qbonus - (1_000_000.0 if dead else 0.0)   # dead sinks to the bottom
        cap_why = _capability_why(slot, m)
        why = f"{cap_why} · {emp_why}" if emp_why != "no usage history yet" else f"{cap_why} (no usage history yet)"
        if q_fresh:
            pct = int(round(q["quality"] * 100))
            why = f"benchmarked {pct}% ({q.get('samples','?')} samples) · {q.get('avg_latency_ms',0)/1000:.1f}s · " + why
        scored.append({"id": m["id"], "score": score, "why": why, "_dead": dead})
    scored.sort(key=lambda x: (-x["score"], x["id"]))
    ranked = [{"id": s["id"], "score": round(s["score"], 1), "why": s["why"]} for s in scored[:8]]
    top = None
    for s in scored:
        if not s["_dead"]:
            top = s
            break
    return {
        "recommended": top["id"] if top else None,
        "ranked": ranked,
        "why": top["why"] if top else "",
    }
