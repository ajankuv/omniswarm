import hashlib
import json
import re
import uuid


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def cache_key(system: str, user: str, task_type: str) -> str:
    """Deterministic key for the verified-answer cache. Normalizes whitespace
    and case so trivially different phrasings of the same prompt collide, then
    hashes system+user+task_type. Exact-normalized match — not embeddings."""
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())
    raw = f"{task_type}\x00{norm(system)}\x00{norm(user)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _iter_json_objects(text: str):
    """Yield every balanced {...} block in the text that parses, in order."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        yield json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)


def extract_json(text: str, require_keys=None) -> dict | None:
    """Best-effort JSON extraction from model output.

    Without `require_keys`: the first balanced {...} block (legacy behaviour).

    With `require_keys`: only blocks carrying at least one of those keys are
    considered, and the LAST such block wins. This matters for judge/council
    output: the prompt embeds attacker-influenced text (the task and the draft
    answer), so a model that echoes its input can emit a planted object like
    {"score": 1.0, "action": "pass"} *before* its own verdict. Taking the first
    block would let that forge a pass — and a forged pass+high is then cached
    and re-served. Requiring the expected keys and preferring the model's final
    block makes that substantially harder."""
    if not require_keys:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        for obj in _iter_json_objects(text):
            return obj
        return None

    keys = set(require_keys)
    try:
        whole = json.loads(text)
        if isinstance(whole, dict) and keys & whole.keys():
            return whole        # the entire reply is the verdict — unambiguous
    except (json.JSONDecodeError, TypeError):
        pass
    qualifying = [o for o in _iter_json_objects(text)
                  if isinstance(o, dict) and keys & o.keys()]
    if len(qualifying) != 1:
        # Zero: nothing to read. More than one: the reply contains a verdict-shaped
        # object the model did not author (echoed or planted), and picking one would
        # be a guess. Fail closed — callers treat None as "unsure", which escalates.
        return None
    return qualifying[0]
