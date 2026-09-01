"""Query-decomposition retriever (LLM-assisted fan-out).

Wraps any `BaseRetriever`. The generator splits one narrative into a few
focused sub-queries; each is retrieved separately and the results merged by
**max score per chunk**. When decomposition yields a single query the behaviour
is identical to the inner retriever, so this is a drop-in wrapper that can only
help on multi-symptom input.

The generator is injected at construction. In the proof-of-concept this class
imported `src.rag_chain` lazily inside its methods to break a circular import;
`rag_chain` is gone, so the workaround has nothing left to work around.
"""

from __future__ import annotations

from pydantic import BaseModel

from soteriamed.generation.base import Generator, ResponseParseError
from soteriamed.retrieval.base import BaseRetriever

DECOMPOSE_PROMPT = (
    "Split the patient description below into at most {n} short, focused search "
    "queries, one per symptom or body system.\n"
    'Reply with JSON only, in the form {{"queries": ["...", "..."]}}.\n\n'
    "Patient description: {question}"
)


class SubQueries(BaseModel):
    """What the decomposition prompt asks the generator to fill."""

    queries: list[str]


class DecomposingRetriever(BaseRetriever):
    """Fan a query out into sub-queries via *generator*, then merge results."""

    def __init__(
        self,
        inner: BaseRetriever,
        generator: Generator,
        max_subqueries: int = 3,
        k: int | None = None,
    ) -> None:
        self.inner = inner
        self.generator = generator
        self.max_subqueries = max_subqueries
        self.default_k = k if k is not None else getattr(inner, "default_k", 3)

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.default_k

        best: dict[tuple, dict] = {}
        for sq in self.get_subqueries(query):
            for r in self.inner.retrieve(sq, k=k):
                key = (
                    r["metadata"].get("source_index"),
                    r["metadata"].get("chunk_index"),
                )
                existing = best.get(key)
                if existing is None or r["score"] > existing["score"]:
                    best[key] = r

        merged = sorted(best.values(), key=lambda r: r["score"], reverse=True)
        return merged[:k]

    def get_subqueries(self, query: str) -> list[str]:
        """The generator's decomposition of *query*, deduplicated and capped.

        Falls back to ``[query]`` — leaving the wrapper a no-op over the inner
        retriever — when the generator returns nothing usable. A model that will
        not produce valid JSON degrades to "the arm did nothing", which shows up
        in the phase-4 comparison rather than crashing the run.
        """
        prompt = DECOMPOSE_PROMPT.format(n=self.max_subqueries, question=query)
        try:
            raw = self.generator.generate(prompt, SubQueries).queries
        except ResponseParseError:
            return [query]

        seen: set[str] = set()
        out: list[str] = []
        for sq in raw:
            sq = sq.strip()
            if not sq or sq.lower() in seen:
                continue
            seen.add(sq.lower())
            out.append(sq)
            if len(out) == self.max_subqueries:
                break
        return out or [query]
