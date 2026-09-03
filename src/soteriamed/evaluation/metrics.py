"""Evaluation metrics: retrieval quality, faithfulness, and lexical grounding."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

Relevance = Callable[[dict], bool]

# Retrieval metrics


def precision_at_k(retrieved: list[dict], is_relevant: Relevance, k: int = 3) -> float:
    """Fraction of the top-k results that *is_relevant* accepts."""
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if is_relevant(r)) / len(top)


def hit_at_k(retrieved: list[dict], is_relevant: Relevance, k: int = 3) -> int:
    """1 if any of the top-k results is relevant, else 0."""
    return int(any(is_relevant(r) for r in retrieved[:k]))


def reciprocal_rank(retrieved: list[dict], is_relevant: Relevance) -> float:
    """1 / rank of the first relevant result (0 if nothing retrieved is relevant)."""
    for i, r in enumerate(retrieved, start=1):
        if is_relevant(r):
            return 1.0 / i
    return 0.0


class Encoder(Protocol):
    """Minimal interface evaluation needs from a SentenceTransformer-like model."""

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


# Faithfulness


def _embed(encoder: Encoder, texts: list[str]) -> np.ndarray:
    """Encode *texts* into L2-normalized float32 vectors."""
    vecs = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype="float32")


def faithfulness_scores(
    answer: str,
    sources: list[dict],
    encoder: Encoder,
) -> dict[str, float]:
    """Cosine similarity between *answer* and each retrieved source.

    Embeddings are L2-normalized so the dot product equals cosine similarity
    in ``[-1, 1]``. Returns the max/mean/min across sources and the source
    count. Empty inputs yield zeros so downstream aggregation never sees NaN.
    """
    if not sources or not answer.strip():
        return {"max_sim": 0.0, "mean_sim": 0.0, "min_sim": 0.0, "n_sources": 0}

    texts = [answer] + [s["text"] for s in sources]
    vecs = _embed(encoder, texts)
    sims = vecs[0] @ vecs[1:].T
    return {
        "max_sim": float(sims.max()),
        "mean_sim": float(sims.mean()),
        "min_sim": float(sims.min()),
        "n_sources": len(sources),
    }


# Hallucination (lexical entity coverage)

_TOKEN_RE = re.compile(r"[a-z][a-z\-]{3,}")

# Generic English filler only. No negations -- "without" inverts clinical meaning
# -- and no prompt-scaffold words, which were tuned to a flan-t5 prompt that is
# gone. Provisional: HPO term matching is the intended
# successor, replacing this blacklist with a controlled vocabulary.
# A split string, not a set literal, so `ruff format` keeps it compact.
_STOPWORD_TEXT = """
    patient patients presents presented presenting presentation
    history diagnosis condition symptoms symptom likely
    based above below given since with
    this that these those their there where which
    would should could shall will have having been
    into from about than then also such more most
    some many much very just only they them
    your yours well after before during while when
"""

_STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())


def extract_terms(text: str) -> set[str]:
    """Lowercase content tokens of length >= 4 minus an English stopword list.

    Hyphenated terms (``post-operative``) are kept intact. Designed for coarse
    overlap, not linguistic accuracy.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def hallucination_metrics(answer: str, sources: list[dict]) -> dict[str, Any]:
    """Lexical grounding of *answer* in the union of *sources*.

    Returns:
        coverage: fraction of answer terms also present in sources (0 if no terms).
        n_unsupported: number of answer terms missing from every source.
        unsupported_terms: up to 10 of those missing terms (sorted, for stability).

    There is deliberately no ``hallucinated`` boolean. Coverage is continuous,
    and where the line between grounded and not falls is a reporting decision
    rather than a property of the data.
    """
    answer_terms = extract_terms(answer)
    if not answer_terms:
        return {"coverage": 0.0, "n_unsupported": 0, "unsupported_terms": []}

    source_terms: set[str] = set()
    for s in sources:
        source_terms |= extract_terms(s["text"])

    supported = answer_terms & source_terms
    unsupported = sorted(answer_terms - source_terms)

    return {
        "coverage": float(len(supported) / len(answer_terms)),
        "n_unsupported": len(unsupported),
        "unsupported_terms": unsupported[:10],
    }
