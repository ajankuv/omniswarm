import json
import uuid


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def extract_json(text: str) -> dict | None:
    """Best-effort: parse the first balanced {...} object in the text."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
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
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
