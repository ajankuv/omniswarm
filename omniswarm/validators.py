import json
from typing import Callable


def non_empty(text: str, params: dict) -> bool:
    return bool(text.strip())


def max_words(text: str, params: dict) -> bool:
    limit = params.get("max_words")
    if limit is None:
        return True
    return len(text.split()) <= limit


def json_valid(text: str, params: dict) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def no_banned(text: str, params: dict) -> bool:
    banned = params.get("banned", [])
    lowered = text.lower()
    return not any(b.lower() in lowered for b in banned)


VALIDATORS: dict[str, Callable[[str, dict], bool]] = {
    "non_empty": non_empty,
    "max_words": max_words,
    "json_valid": json_valid,
    "no_banned": no_banned,
}


def run_validators(names: tuple[str, ...], text: str, params: dict) -> list[str]:
    failed = []
    for name in names:
        check = VALIDATORS.get(name)
        if check is None:
            raise KeyError(f"unknown validator: {name}")
        if not check(text, params):
            failed.append(name)
    return failed
