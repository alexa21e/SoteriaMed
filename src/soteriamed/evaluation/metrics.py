"""Evaluation metrics: retrieval quality, faithfulness, hallucination, specialty
accuracy, and an end-to-end runner.

Pure functions, no prints. Designed so the SentenceTransformer encoder loaded
inside :class:`~soteriamed.retrieval.dense.FAISSRetriever` can be reused
directly (pass ``retriever.model``).

This is the proof-of-concept's ``evaluation.py`` merged with the four retrieval
metrics that lived at the bottom of ``retrieval.py``. Splitting it -- phase 2
takes ``evaluation/retrieval.py``, phase 6 carves out ``evaluation/runner.py``
-- is later work; the module moves here whole so the move stays mechanical.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import pandas as pd

from soteriamed.retrieval.base import BaseRetriever

FAITHFUL_THRESHOLD: float = 0.55  # max cosine sim >= this  -> "faithful"
COVERAGE_THRESHOLD: float = 0.5  # fraction of answer terms supported -> "grounded"


# Retrieval metrics


def precision_at_k(retrieved: list[dict], expected_specialty: str, k: int = 3) -> float:
    """Fraction of top-k results whose specialty matches *expected_specialty*."""
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(
        1 for r in top if r["metadata"]["medical_specialty"] == expected_specialty
    )
    return hits / len(top)


def hit_at_k(retrieved: list[dict], expected_specialty: str, k: int = 3) -> int:
    """1 if any of the top-k results match *expected_specialty*, else 0."""
    return int(
        any(
            r["metadata"]["medical_specialty"] == expected_specialty
            for r in retrieved[:k]
        )
    )


def reciprocal_rank(retrieved: list[dict], expected_specialty: str) -> float:
    """1 / rank of the first matching result (0 if no match)."""
    for i, r in enumerate(retrieved, start=1):
        if r["metadata"]["medical_specialty"] == expected_specialty:
            return 1.0 / i
    return 0.0


def evaluate_queries(
    retriever: BaseRetriever, queries: list[dict], k: int = 3
) -> pd.DataFrame:
    """Run all *queries* through *retriever* and compute per-query metrics.

    Returns a DataFrame with columns:
        id, query, expected_specialty, category, precision_at_k, hit_at_k, mrr
    """
    rows = []
    for q in queries:
        results = retriever.retrieve(q["query"], k=k)
        rows.append(
            {
                "id": q["id"],
                "query": q["query"],
                "expected_specialty": q["expected_specialty"],
                "category": q["category"],
                "precision_at_k": precision_at_k(results, q["expected_specialty"], k),
                "hit_at_k": hit_at_k(results, q["expected_specialty"], k),
                "mrr": reciprocal_rank(results, q["expected_specialty"]),
            }
        )
    return pd.DataFrame(rows)


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


def is_faithful(
    scores: dict[str, float], threshold: float = FAITHFUL_THRESHOLD
) -> bool:
    """A response is faithful when its closest source clears *threshold*."""
    return scores.get("n_sources", 0) > 0 and scores["max_sim"] >= threshold


# Hallucination (lexical entity coverage)

_TOKEN_RE = re.compile(r"[a-z][a-z\-]{3,}")

_STOPWORDS: frozenset[str] = frozenset(
    {
        # filler / scaffold
        "patient",
        "patients",
        "presents",
        "presented",
        "presenting",
        "presentation",
        "history",
        "diagnosis",
        "condition",
        "symptoms",
        "symptom",
        "likely",
        "based",
        "above",
        "below",
        "given",
        "since",
        "with",
        "without",
        "this",
        "that",
        "these",
        "those",
        "their",
        "there",
        "where",
        "which",
        "would",
        "should",
        "could",
        "shall",
        "will",
        "have",
        "having",
        "been",
        "into",
        "from",
        "about",
        "than",
        "then",
        "also",
        "such",
        "more",
        "most",
        "some",
        "many",
        "much",
        "very",
        "just",
        "only",
        "they",
        "them",
        "your",
        "yours",
        "well",
        "after",
        "before",
        "during",
        "while",
        "when",
        # prompt scaffold words
        "clinical",
        "assistant",
        "reference",
        "excerpts",
        "excerpt",
        "ground",
        "claim",
        "source",
        "sources",
        "recommend",
        "describe",
        "handle",
        "specialty",
        "medical",
        "physician",
        "answer",
        "question",
        "context",
    }
)


def extract_terms(text: str) -> set[str]:
    """Lowercase content tokens of length >= 4 minus an English/clinical stopword list.

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
        hallucinated: True when coverage < :data:`COVERAGE_THRESHOLD`.
    """
    answer_terms = extract_terms(answer)
    if not answer_terms:
        return {
            "coverage": 0.0,
            "n_unsupported": 0,
            "unsupported_terms": [],
            "hallucinated": True,
        }

    source_terms: set[str] = set()
    for s in sources:
        source_terms |= extract_terms(s["text"])

    supported = answer_terms & source_terms
    unsupported = sorted(answer_terms - source_terms)
    coverage = len(supported) / len(answer_terms)

    return {
        "coverage": float(coverage),
        "n_unsupported": len(unsupported),
        "unsupported_terms": unsupported[:10],
        "hallucinated": coverage < COVERAGE_THRESHOLD,
    }


# Specialty accuracy (two signals)


def predicted_specialty_from_sources(sources: list[dict]) -> str | None:
    """Majority vote on ``metadata['medical_specialty']`` across *sources*.

    Ties are broken by the highest sum of retrieval scores. ``None`` if no sources.
    """
    if not sources:
        return None

    counts: Counter[str] = Counter()
    score_sums: dict[str, float] = {}
    for s in sources:
        spec = s["metadata"].get("medical_specialty")
        if not spec:
            continue
        counts[spec] += 1
        score_sums[spec] = score_sums.get(spec, 0.0) + float(s.get("score", 0.0))

    if not counts:
        return None

    top_count = max(counts.values())
    candidates = [s for s, c in counts.items() if c == top_count]
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda s: score_sums.get(s, 0.0))


def predicted_specialty_from_answer(
    answer: str,
    known_specialties: list[str],
) -> str | None:
    """Case-insensitive substring match of any *known_specialties* in *answer*.

    The longest matching specialty name wins (so ``"Cardiovascular / Pulmonary"``
    beats ``"Cardiology"`` if both appear). ``None`` on no match.
    """
    if not answer or not known_specialties:
        return None

    haystack = answer.lower()
    matches = [s for s in known_specialties if s.lower() in haystack]
    if not matches:
        return None
    return max(matches, key=len)


# End-to-end runner


def run_full_evaluation(
    chain: Callable[[str], str],
    retriever: BaseRetriever,
    queries: list[dict],
    encoder: Encoder,
    known_specialties: list[str],
    k: int = 3,
    faithful_threshold: float = FAITHFUL_THRESHOLD,
) -> pd.DataFrame:
    """Run every query through *retriever* + *chain* and compute all metrics.

    *chain* is any ``narrative -> answer text`` callable. It is deliberately not a
    :class:`~soteriamed.generation.base.Generator`: `Generator.generate` returns a
    validated `BaseModel`, and this runner is the proof-of-concept's, which compares
    free text against retrieved sources. The two meet in phase 5, when
    ``evaluation/runner.py`` is carved out of this module and rewritten around the
    structured schema. Retyping it now would be a rewrite, not a move.

    Returns one row per query with the columns documented in the project plan.
    No prints; intended for downstream plotting/aggregation.
    """
    rows: list[dict[str, Any]] = []
    for q in queries:
        question = q["query"]
        sources = retriever.retrieve(question, k=k)
        answer = chain(question)

        faith = faithfulness_scores(answer, sources, encoder)
        halluc = hallucination_metrics(answer, sources)
        pred_src = predicted_specialty_from_sources(sources)
        pred_ans = predicted_specialty_from_answer(answer, known_specialties)

        expected = q["expected_specialty"]
        top_score = float(sources[0]["score"]) if sources else 0.0

        rows.append(
            {
                "id": q["id"],
                "query": question,
                "category": q["category"],
                "expected_specialty": expected,
                "answer": answer,
                "n_sources": faith["n_sources"],
                "retrieval_top_score": top_score,
                "faith_max": faith["max_sim"],
                "faith_mean": faith["mean_sim"],
                "faithful": is_faithful(faith, threshold=faithful_threshold),
                "coverage": halluc["coverage"],
                "n_unsupported": halluc["n_unsupported"],
                "hallucinated": halluc["hallucinated"],
                "pred_specialty_sources": pred_src,
                "pred_specialty_answer": pred_ans,
                "correct_sources": int(pred_src == expected) if pred_src else 0,
                "correct_answer": int(pred_ans == expected) if pred_ans else 0,
            }
        )
    return pd.DataFrame(rows)


def summarize_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate :func:`run_full_evaluation` output by *category*.

    Columns: ``mean_faith_max``, ``faithful_rate``, ``hallucination_rate``,
    ``mean_coverage``, ``acc_sources``, ``acc_answer``, ``n_queries``.
    """
    grouped = df.groupby("category", sort=False).agg(
        mean_faith_max=("faith_max", "mean"),
        faithful_rate=("faithful", "mean"),
        hallucination_rate=("hallucinated", "mean"),
        mean_coverage=("coverage", "mean"),
        acc_sources=("correct_sources", "mean"),
        acc_answer=("correct_answer", "mean"),
        n_queries=("id", "count"),
    )
    return grouped.reset_index()
