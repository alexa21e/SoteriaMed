"""Tests for the evaluation module (src/evaluation.py).

Uses a fake encoder (deterministic vectors keyed on a substring match) and a
fake chain so the suite stays fast and offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from langchain_core.runnables import RunnableLambda

from src.evaluation import (
    COVERAGE_THRESHOLD,
    FAITHFUL_THRESHOLD,
    extract_terms,
    faithfulness_scores,
    hallucination_metrics,
    is_faithful,
    predicted_specialty_from_answer,
    predicted_specialty_from_sources,
    run_full_evaluation,
    summarize_evaluation,
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


# Faithfulness

def test_faithfulness_identical_text(encoder):
    src = [{"text": "acute knee pain after a fall",
            "metadata": {"medical_specialty": "Orthopedic"}, "score": 0.9}]
    scores = faithfulness_scores("acute knee pain after a fall", src, encoder)

    assert scores["n_sources"] == 1
    assert scores["max_sim"] == pytest.approx(1.0, abs=1e-5)
    assert is_faithful(scores) is True


def test_faithfulness_unrelated_text(encoder):
    src = [{"text": "colonoscopy revealed sigmoid polyps",
            "metadata": {"medical_specialty": "Gastroenterology"}, "score": 0.7}]
    scores = faithfulness_scores("alpine skiing technique", src, encoder)

    assert scores["max_sim"] < 0.3
    assert is_faithful(scores) is False


def test_faithfulness_empty_sources(encoder):
    scores = faithfulness_scores("anything", [], encoder)
    assert scores == {"max_sim": 0.0, "mean_sim": 0.0, "min_sim": 0.0, "n_sources": 0}
    assert is_faithful(scores) is False


# Entity coverage / hallucination

def test_extract_terms_filters_stopwords_and_short_words():
    terms = extract_terms("Patient presents with acute substernal chest pain")
    assert "chest" in terms
    assert "pain" in terms
    assert "acute" in terms
    assert "substernal" in terms
    assert "patient" not in terms  # stopword
    assert "with" not in terms     # too short / stopword


def test_extract_terms_keeps_hyphenated_terms():
    terms = extract_terms("post-operative recovery was uneventful")
    assert "post-operative" in terms


def test_hallucination_metrics_no_overlap():
    src = [{"text": "colonoscopy revealed sigmoid polyps",
            "metadata": {"medical_specialty": "Gastroenterology"}}]
    metrics = hallucination_metrics(
        "fractured tibia requires orthopedic surgery", src,
    )
    assert metrics["coverage"] == 0.0
    assert metrics["hallucinated"] is True
    assert metrics["n_unsupported"] > 0


def test_hallucination_metrics_full_overlap():
    src = [{"text": "acute chest pain with elevated troponin",
            "metadata": {"medical_specialty": "Cardiovascular / Pulmonary"}}]
    metrics = hallucination_metrics("chest pain elevated troponin", src)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["hallucinated"] is False
    assert metrics["n_unsupported"] == 0


# Specialty prediction

def test_predicted_specialty_majority_vote():
    sources = [
        {"text": "...", "metadata": {"medical_specialty": "Cardiovascular / Pulmonary"}, "score": 0.6},
        {"text": "...", "metadata": {"medical_specialty": "Cardiovascular / Pulmonary"}, "score": 0.5},
        {"text": "...", "metadata": {"medical_specialty": "Gastroenterology"}, "score": 0.9},
    ]
    assert predicted_specialty_from_sources(sources) == "Cardiovascular / Pulmonary"


def test_predicted_specialty_tie_broken_by_score_sum():
    sources = [
        {"text": "...", "metadata": {"medical_specialty": "Orthopedic"}, "score": 0.2},
        {"text": "...", "metadata": {"medical_specialty": "Cardiovascular / Pulmonary"}, "score": 0.9},
    ]
    assert predicted_specialty_from_sources(sources) == "Cardiovascular / Pulmonary"


def test_predicted_specialty_from_answer_longest_match():
    known = ["Cardiovascular / Pulmonary", "Cardiology", "Orthopedic"]
    answer = "The patient should be seen in cardiovascular / pulmonary clinic."
    assert predicted_specialty_from_answer(answer, known) == "Cardiovascular / Pulmonary"


def test_predicted_specialty_from_answer_no_match():
    known = ["Cardiology", "Orthopedic"]
    assert predicted_specialty_from_answer("see a dietitian", known) is None


# End-to-end runner

def _fake_retriever(chunks):
    class _R:
        def retrieve(self, query, k=None):
            return chunks[: (k or len(chunks))]
    return _R()


def test_run_full_evaluation_columns(encoder):
    chunks = [
        {"text": "acute chest pain with elevated troponin",
         "metadata": {"medical_specialty": "Cardiovascular / Pulmonary",
                      "source_index": 0, "chunk_index": 0}, "score": 0.9},
        {"text": "colonoscopy revealed sigmoid polyps",
         "metadata": {"medical_specialty": "Gastroenterology",
                      "source_index": 1, "chunk_index": 0}, "score": 0.4},
    ]
    chain = RunnableLambda(lambda _: "chest pain elevated troponin")
    queries = [
        {"id": "qA", "query": "chest pain", "category": "direct",
         "expected_specialty": "Cardiovascular / Pulmonary"},
        {"id": "qB", "query": "rectal bleeding", "category": "synonym",
         "expected_specialty": "Gastroenterology"},
    ]
    known = ["Cardiovascular / Pulmonary", "Gastroenterology", "Orthopedic"]

    df = run_full_evaluation(
        chain=chain,
        retriever=_fake_retriever(chunks),
        queries=queries,
        encoder=encoder,
        known_specialties=known,
        k=2,
    )

    expected_cols = {
        "id", "query", "category", "expected_specialty", "answer",
        "n_sources", "retrieval_top_score", "faith_max", "faith_mean",
        "faithful", "coverage", "n_unsupported", "hallucinated",
        "pred_specialty_sources", "pred_specialty_answer",
        "correct_sources", "correct_answer",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 2
    assert df.loc[0, "n_sources"] == 2
    assert df.loc[0, "pred_specialty_sources"] == "Cardiovascular / Pulmonary"
    assert df.loc[0, "correct_sources"] == 1


def test_summarize_evaluation_groups_by_category():
    df = pd.DataFrame([
        {"id": "1", "category": "direct", "faith_max": 0.9, "faithful": True,
         "hallucinated": False, "coverage": 1.0, "correct_sources": 1, "correct_answer": 1},
        {"id": "2", "category": "direct", "faith_max": 0.7, "faithful": True,
         "hallucinated": False, "coverage": 0.8, "correct_sources": 0, "correct_answer": 1},
        {"id": "3", "category": "complex", "faith_max": 0.2, "faithful": False,
         "hallucinated": True, "coverage": 0.1, "correct_sources": 0, "correct_answer": 0},
    ])
    summary = summarize_evaluation(df)
    assert set(summary["category"]) == {"direct", "complex"}
    direct = summary[summary["category"] == "direct"].iloc[0]
    assert direct["n_queries"] == 2
    assert direct["mean_faith_max"] == pytest.approx(0.8)
    assert direct["acc_sources"] == pytest.approx(0.5)
