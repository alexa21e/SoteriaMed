"""Tests for the evaluation metrics (soteriamed/evaluation/metrics.py).

Uses a fake encoder (deterministic vectors, one axis per token) so the suite
stays fast and offline.

No test here asserts a verdict, because the module no longer produces one. The
proof-of-concept's `is_faithful` and `hallucinated` were thresholded booleans
whose cutoffs came from tuning against a different embedding model; the tests
that asserted them were really asserting that the cutoff had not moved.
"""

from __future__ import annotations

import numpy as np
import pytest

from soteriamed.evaluation.metrics import (
    extract_terms,
    faithfulness_scores,
    hallucination_metrics,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
)


class FakeEncoder:
    """Deterministic encoder for tests.

    Each token from the text contributes a unit vector along a fixed axis,
    so identical text yields identical normalized vectors and disjoint text
    yields orthogonal vectors. Behaves like a SentenceTransformer w.r.t. the
    kwargs evaluation passes in.
    """

    DIM = 64

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def _axis(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab) % self.DIM
        return self._vocab[token]

    def encode(self, sentences, **kwargs):
        mat = np.zeros((len(sentences), self.DIM), dtype="float32")
        for i, s in enumerate(sentences):
            for tok in s.lower().split():
                mat[i, self._axis(tok)] += 1.0
        if kwargs.get("normalize_embeddings", False):
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        return mat


@pytest.fixture()
def encoder():
    return FakeEncoder()


def _result(chapter_id: str, chapter_title: str, section: str) -> dict:
    return {
        "text": "...",
        "metadata": {
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "section": section,
            "chunk_index": 0,
        },
        "score": 0.5,
    }


@pytest.fixture()
def ranked():
    """Three results; only the second is about the pathology being asked about."""
    return [
        _result("SP-0003", "Colorectal Polyps", "Evaluation"),
        _result("SP-0002", "Acute Coronary Syndrome", "History and Physical"),
        _result("SP-0005", "Congestive Heart Failure", "Evaluation"),
    ]


def by_chapter_title(expected: str):
    """The shape of relevance predicate phase 2 week 9 will supply."""
    return lambda r: r["metadata"]["chapter_title"] == expected


# Retrieval metrics
#
# These are the generalised versions. The proof-of-concept compared one
# hardcoded metadata key against a string argument, which tied all three
# functions to the old corpus schema. Relevance is now an injected predicate, so
# the same three functions serve the pathology rule in phase 2 and whatever
# phase 4 needs, without being rewritten.


def test_precision_at_k_counts_only_the_top_k(ranked):
    is_relevant = by_chapter_title("Acute Coronary Syndrome")

    assert precision_at_k(ranked, is_relevant, k=1) == 0.0
    assert precision_at_k(ranked, is_relevant, k=2) == pytest.approx(0.5)
    assert precision_at_k(ranked, is_relevant, k=3) == pytest.approx(1 / 3)


def test_precision_at_k_on_no_results_is_zero_not_a_zero_division():
    assert precision_at_k([], by_chapter_title("anything"), k=3) == 0.0


def test_precision_at_k_divides_by_what_was_returned_not_by_k(ranked):
    """Two results and k=5 means the denominator is 2."""
    is_relevant = by_chapter_title("Acute Coronary Syndrome")
    assert precision_at_k(ranked[:2], is_relevant, k=5) == pytest.approx(0.5)


def test_hit_at_k(ranked):
    is_relevant = by_chapter_title("Acute Coronary Syndrome")

    assert hit_at_k(ranked, is_relevant, k=1) == 0
    assert hit_at_k(ranked, is_relevant, k=2) == 1
    assert hit_at_k(ranked, is_relevant, k=3) == 1


def test_hit_at_k_when_nothing_is_relevant(ranked):
    assert hit_at_k(ranked, by_chapter_title("Acute Appendicitis"), k=3) == 0


def test_reciprocal_rank_is_one_over_the_first_relevant_position(ranked):
    assert reciprocal_rank(ranked, by_chapter_title("Colorectal Polyps")) == 1.0
    assert reciprocal_rank(
        ranked, by_chapter_title("Acute Coronary Syndrome")
    ) == pytest.approx(0.5)
    assert reciprocal_rank(
        ranked, by_chapter_title("Congestive Heart Failure")
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_when_nothing_is_relevant(ranked):
    assert reciprocal_rank(ranked, by_chapter_title("Acute Appendicitis")) == 0.0


def test_the_predicate_is_genuinely_injectable(ranked):
    """The point of the generalisation: same results, different relevance rule.

    A section-based predicate scores the same ranking differently from a
    chapter-based one. Nothing in the metrics knows what a chapter or a section
    is, which is what makes them survive a change of corpus.
    """

    def by_section(r):
        return r["metadata"]["section"] == "Evaluation"

    assert precision_at_k(ranked, by_section, k=3) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranked, by_section) == 1.0
    assert precision_at_k(
        ranked, by_chapter_title("Colorectal Polyps"), k=3
    ) == pytest.approx(1 / 3)


# Faithfulness


def test_faithfulness_identical_text(encoder):
    src = [{"text": "acute knee pain after a fall", "metadata": {}, "score": 0.9}]
    scores = faithfulness_scores("acute knee pain after a fall", src, encoder)

    assert scores["n_sources"] == 1
    assert scores["max_sim"] == pytest.approx(1.0, abs=1e-5)


def test_faithfulness_unrelated_text(encoder):
    src = [
        {"text": "colonoscopy revealed sigmoid polyps", "metadata": {}, "score": 0.7}
    ]
    scores = faithfulness_scores("alpine skiing technique", src, encoder)

    assert scores["max_sim"] < 0.3


def test_faithfulness_empty_sources(encoder):
    scores = faithfulness_scores("anything", [], encoder)
    assert scores == {"max_sim": 0.0, "mean_sim": 0.0, "min_sim": 0.0, "n_sources": 0}


def test_faithfulness_reports_a_spread_not_a_verdict(encoder):
    """max/mean/min are all returned so results can show a distribution."""
    src = [
        {"text": "acute chest pain with elevated troponin", "metadata": {}},
        {"text": "alpine skiing technique", "metadata": {}},
    ]
    scores = faithfulness_scores(
        "acute chest pain with elevated troponin", src, encoder
    )

    assert scores["min_sim"] < scores["mean_sim"] < scores["max_sim"]


# Entity coverage / lexical grounding


def test_extract_terms_filters_stopwords_and_short_words():
    terms = extract_terms("Patient presents with acute substernal chest pain")
    assert "chest" in terms
    assert "pain" in terms
    assert "acute" in terms
    assert "substernal" in terms
    assert "patient" not in terms  # stopword
    assert "with" not in terms  # stopword


def test_extract_terms_keeps_hyphenated_terms():
    terms = extract_terms("post-operative recovery was uneventful")
    assert "post-operative" in terms


def test_extract_terms_keeps_negation():
    """`without` is a content term, not filler.

    While it was a stopword, an answer reading "chest pain without exertion"
    scored as fully supported by a source reading "chest pain on exertion" --
    the metric discarded the one word that inverts the clinical meaning.
    """
    assert "without" in extract_terms("chest pain without exertion")


def test_extract_terms_drops_no_prompt_scaffold_words():
    """The scaffold list was tuned to a prompt that no longer exists.

    Several of its entries are content words in this corpus, so keeping them
    would understate coverage on exactly the sentences that matter.
    """
    terms = extract_terms("clinical assessment by a physician")
    assert "clinical" in terms
    assert "physician" in terms


def test_hallucination_metrics_no_overlap():
    src = [{"text": "colonoscopy revealed sigmoid polyps", "metadata": {}}]
    metrics = hallucination_metrics("fractured tibia requires surgery", src)

    assert metrics["coverage"] == 0.0
    assert metrics["n_unsupported"] > 0
    assert metrics["unsupported_terms"] == sorted(metrics["unsupported_terms"])


def test_hallucination_metrics_full_overlap():
    src = [{"text": "acute chest pain with elevated troponin", "metadata": {}}]
    metrics = hallucination_metrics("chest pain elevated troponin", src)

    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["n_unsupported"] == 0
    assert metrics["unsupported_terms"] == []


def test_hallucination_metrics_partial_overlap():
    src = [{"text": "acute chest pain", "metadata": {}}]
    metrics = hallucination_metrics("chest pain with elevated troponin", src)

    assert 0.0 < metrics["coverage"] < 1.0
    assert set(metrics["unsupported_terms"]) == {"elevated", "troponin"}


def test_hallucination_metrics_reports_no_verdict():
    """There is deliberately no `hallucinated` key.

    Where the line falls between grounded and not is a reporting decision, and
    the proof-of-concept's 0.5 came from tuning against a different encoder.
    """
    src = [{"text": "acute chest pain", "metadata": {}}]
    metrics = hallucination_metrics("chest pain", src)

    assert "hallucinated" not in metrics
    assert set(metrics) == {"coverage", "n_unsupported", "unsupported_terms"}


def test_hallucination_metrics_on_an_answer_with_no_content_terms():
    metrics = hallucination_metrics("it is", [{"text": "anything", "metadata": {}}])
    assert metrics == {"coverage": 0.0, "n_unsupported": 0, "unsupported_terms": []}
