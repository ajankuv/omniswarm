from omniswarm.util import estimate_tokens, new_id, extract_json


def test_estimate_tokens_is_quarter_of_chars():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 0


def test_new_id_is_unique_and_short():
    a, b = new_id(), new_id()
    assert a != b
    assert len(a) == 12


def test_extract_json_from_fenced_block():
    text = 'sure!\n```json\n{"score": 0.9, "action": "pass"}\n```\nthanks'
    assert extract_json(text) == {"score": 0.9, "action": "pass"}


def test_extract_json_returns_none_when_absent():
    assert extract_json("no json here") is None
