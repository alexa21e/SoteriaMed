"""Tests for the generation seam (soteriamed/generation/base.py).

Entirely offline — no weights, no network. The schemas here are local fixtures,
not the real triage schema; that lands in phase 5 week 24, and `parse_response`
is deliberately generic over it.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from soteriamed.generation.base import (
    Generator,
    ResponseParseError,
    StubGenerator,
    parse_response,
)


class Condition(BaseModel):
    name: str
    rationale: str


class Answer(BaseModel):
    """Stands in for the week-24 triage schema."""

    conditions: list[Condition]
    urgency: str


VALID_JSON = (
    '{"conditions": [{"name": "Angina", "rationale": "chest tightness on exertion"}], '
    '"urgency": "URGENT_24_48H"}'
)


def _assert_parsed(obj: Answer) -> None:
    assert obj.urgency == "URGENT_24_48H"
    assert [c.name for c in obj.conditions] == ["Angina"]


# parse_response — the three rungs

def test_bare_json():
    _assert_parsed(parse_response(VALID_JSON, Answer))


def test_surrounding_whitespace_is_stripped():
    _assert_parsed(parse_response(f"\n\n  {VALID_JSON}\n  ", Answer))


def test_json_fence():
    _assert_parsed(parse_response(f"```json\n{VALID_JSON}\n```", Answer))


def test_bare_fence_without_language():
    _assert_parsed(parse_response(f"```\n{VALID_JSON}\n```", Answer))


def test_uppercase_fence_language():
    _assert_parsed(parse_response(f"```JSON\n{VALID_JSON}\n```", Answer))


def test_prose_around_fence():
    raw = f"Sure! Here is the output.\n\n```json\n{VALID_JSON}\n```\n\nHope that helps."
    _assert_parsed(parse_response(raw, Answer))


def test_prose_around_bare_json():
    """Rung 3: no fence, commentary on both sides."""
    raw = f"Here is the JSON: {VALID_JSON} Let me know if you need more."
    _assert_parsed(parse_response(raw, Answer))


def test_first_valid_candidate_wins():
    """A junk fence before a good one does not sink the parse."""
    raw = f"```json\n{{not json at all}}\n```\n```json\n{VALID_JSON}\n```"
    _assert_parsed(parse_response(raw, Answer))


def test_nested_braces_survive_rung_three():
    raw = f"Output below.\n{VALID_JSON}\nDone."
    _assert_parsed(parse_response(raw, Answer))


# parse_response — failures

@pytest.mark.parametrize("raw", ["", "   ", "\n\t "])
def test_empty_output_raises(raw):
    with pytest.raises(ResponseParseError, match="empty"):
        parse_response(raw, Answer)


def test_unparseable_output_raises():
    with pytest.raises(ResponseParseError):
        parse_response("I'm sorry, I can't help with that.", Answer)


def test_valid_json_wrong_shape_raises():
    """Syntactically fine, schema-invalid — must not return a partial object."""
    with pytest.raises(ResponseParseError, match="Answer"):
        parse_response('{"conditions": []}', Answer)


def test_wrong_field_type_raises():
    with pytest.raises(ResponseParseError):
        parse_response('{"conditions": "Angina", "urgency": "ROUTINE"}', Answer)


def test_error_carries_raw_output():
    """Week 24's retry loop needs the offending text back."""
    raw = "not json"
    with pytest.raises(ResponseParseError) as exc:
        parse_response(raw, Answer)
    assert exc.value.raw == raw


def test_error_is_a_valueerror():
    assert issubclass(ResponseParseError, ValueError)


# StubGenerator

def test_stub_parses_string_responses():
    stub = StubGenerator(VALID_JSON)
    _assert_parsed(stub.generate("any prompt", Answer))


def test_stub_string_goes_through_the_real_parser():
    """Fenced canned output must work, or the stub is routing around parsing."""
    stub = StubGenerator(f"```json\n{VALID_JSON}\n```")
    _assert_parsed(stub.generate("any prompt", Answer))


def test_stub_returns_model_instances_unchanged():
    canned = Answer(conditions=[], urgency="ROUTINE")
    stub = StubGenerator(canned)
    assert stub.generate("any prompt", Answer) is canned


def test_stub_rejects_a_schema_mismatch():
    stub = StubGenerator(Condition(name="Angina", rationale="x"))
    with pytest.raises(ResponseParseError, match="Condition"):
        stub.generate("any prompt", Answer)


def test_stub_records_prompts():
    stub = StubGenerator(VALID_JSON)
    stub.generate("first", Answer)
    stub.generate("second", Answer)
    assert stub.prompts == ["first", "second"]


def test_stub_bad_canned_string_still_raises():
    stub = StubGenerator("nonsense")
    with pytest.raises(ResponseParseError):
        stub.generate("any prompt", Answer)


def test_stub_satisfies_the_generator_protocol():
    assert isinstance(StubGenerator(VALID_JSON), Generator)
