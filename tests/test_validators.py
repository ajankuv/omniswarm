from omniswarm.validators import run_validators


def test_all_pass_returns_empty():
    assert run_validators(("non_empty",), "hello", {}) == []


def test_non_empty_fails_on_blank():
    assert run_validators(("non_empty",), "   ", {}) == ["non_empty"]


def test_max_words_fails_when_too_long():
    failed = run_validators(("max_words",), "one two three", {"max_words": 2})
    assert failed == ["max_words"]


def test_json_valid_and_no_banned():
    assert run_validators(("json_valid",), '{"a": 1}', {}) == []
    assert run_validators(("json_valid",), "not json", {}) == ["json_valid"]
    assert run_validators(("no_banned",), "as an AI language model", {"banned": ["as an ai"]}) == ["no_banned"]
