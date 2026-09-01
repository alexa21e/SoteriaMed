"""The generation seam: a `Generator` protocol, a response parser, and a stub.

The real generator is chosen in phase 5 week 24, so no model implementation
lives here. What is durable is the *seam* — everything downstream depends on
`Generator`, not on a vendor SDK — and the parsing half, which chapter 3 has to
describe precisely enough to reimplement (invariant 8: prompt assembly and
parsing live in project code, not in an orchestration framework).

Writing the parser now separates it from prompt-behaviour debugging in week 24,
when a malformed response and a bad prompt would otherwise fail together and be
hard to tell apart.
"""

from __future__ import annotations

import re
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


@runtime_checkable
class Generator(Protocol):
    """Anything that turns a prompt into a validated instance of *schema*.

    Implementations own their own retry policy. A `Generator` either returns a
    schema-valid object or raises; callers never see raw model text.
    """

    def generate(self, prompt: str, schema: type[M]) -> M: ...


class ResponseParseError(ValueError):
    """Raised when model output cannot be validated against the schema.

    Carries the offending text on `.raw` so a retry loop can put it back in the
    prompt, and so a failure in `results/` records what the model actually said
    rather than only that it failed.
    """

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _candidates(raw: str) -> list[str]:
    """JSON candidates from *raw*, most to least likely, first-match-wins.

    Three rungs, in order:

    1. the whole string — a well-behaved model in JSON mode;
    2. each fenced block — a model that wrapped it in ```json ... ```;
    3. the outermost brace/bracket span — a model that added prose around it
       ("Here is the JSON:").

    Rung 3 is deliberate leniency, not accident. Real models prepend commentary
    often enough that refusing it would inflate the retry rate for no gain, and
    the span is still validated against the schema like any other candidate.
    """
    out = [raw]
    out.extend(m.group(1).strip() for m in _FENCE_RE.finditer(raw))

    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end > start:
            out.append(raw[start : end + 1])

    seen: set[str] = set()
    return [c for c in out if c and not (c in seen or seen.add(c))]


def parse_response(raw: str, schema: type[M]) -> M:
    """Validate model output *raw* against *schema*.

    Tries each candidate from :func:`_candidates` in order and returns the first
    that validates. Raises :class:`ResponseParseError` if none do — never
    returns a partially-populated object, and never silently substitutes a
    default.
    """
    if not raw or not raw.strip():
        raise ResponseParseError("model returned empty output", raw)

    last: ValidationError | None = None
    for candidate in _candidates(raw.strip()):
        try:
            return schema.model_validate_json(candidate)
        except ValidationError as exc:
            last = exc

    detail = f": {last}" if last is not None else ""
    raise ResponseParseError(
        f"no candidate in model output validated against {schema.__name__}{detail}",
        raw,
    )


class StubGenerator:
    """A `Generator` that returns a canned response. Tests only — no weights.

    Give it a `str` and it goes through :func:`parse_response`, so tests
    exercise the real parsing path rather than routing around it. Give it a
    `BaseModel` and it is returned as-is, for tests that care about downstream
    behaviour and not about parsing.

    Every prompt it is asked is appended to `.prompts`, so a caller's prompt
    assembly can be asserted on without a model.
    """

    def __init__(self, response: str | BaseModel) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: type[M]) -> M:
        self.prompts.append(prompt)

        if isinstance(self.response, BaseModel):
            if not isinstance(self.response, schema):
                raise ResponseParseError(
                    f"stub holds {type(self.response).__name__}, caller asked for "
                    f"{schema.__name__}",
                    repr(self.response),
                )
            return self.response

        return parse_response(self.response, schema)
