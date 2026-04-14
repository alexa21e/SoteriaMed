"""Retrieval module — TF-IDF baseline and evaluation helpers."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Base retriever

class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Return top-k results for *query*.

        Each result is ``{"text": str, "metadata": dict, "score": float}``.
        """

    def retrieve_batch(self, queries: list[str], k: int | None = None) -> list[list[dict]]:
        """Run *retrieve* for every query in *queries*."""
        return [self.retrieve(q, k=k) for q in queries]

# TF-IDF retriever

class TFIDFRetriever(BaseRetriever):
    """Sparse retriever based on TF-IDF + cosine similarity."""

    def __init__(self, chunks: list[dict], k: int = 3) -> None:
        self.chunks = chunks
        self.default_k = k

        self.vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        texts = [c["text"] for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    # -- public API ---------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.default_k
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Top-k indices (handle k > corpus size)
        n_results = min(k, len(self.chunks))
        top_idx = np.argsort(scores)[::-1][:n_results]

        results = []
        for i in top_idx:
            results.append({
                "text": self.chunks[i]["text"],
                "metadata": self.chunks[i]["metadata"],
                "score": float(scores[i]),
            })
        return results

    def get_vocabulary_size(self) -> int:
        """Number of terms in the fitted vocabulary."""
        return len(self.vectorizer.vocabulary_)

    def get_top_terms(self, query: str, n: int = 10) -> list[tuple[str, float]]:
        """Return the *n* highest-weight terms for *query* according to TF-IDF."""
        query_vec = self.vectorizer.transform([query])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = query_vec.toarray().flatten()
        top_idx = np.argsort(scores)[::-1][:n]
        return [(feature_names[i], float(scores[i])) for i in top_idx if scores[i] > 0]

# FAISS retriever (stub for S9-S12)

class FAISSRetriever(BaseRetriever):
    """Dense retriever using FAISS — to be implemented in S9-S12."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("FAISSRetriever will be implemented in S9-S12.")

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        raise NotImplementedError

# Evaluation helpers

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
    return int(any(
        r["metadata"]["medical_specialty"] == expected_specialty
        for r in retrieved[:k]
    ))


def reciprocal_rank(retrieved: list[dict], expected_specialty: str) -> float:
    """1 / rank of the first matching result (0 if no match)."""
    for i, r in enumerate(retrieved, start=1):
        if r["metadata"]["medical_specialty"] == expected_specialty:
            return 1.0 / i
    return 0.0


def evaluate_queries(retriever: BaseRetriever, queries: list[dict], k: int = 3) -> pd.DataFrame:
    """Run all *queries* through *retriever* and compute per-query metrics.

    Returns a DataFrame with columns:
        id, query, expected_specialty, category, precision_at_k, hit_at_k, mrr
    """
    rows = []
    for q in queries:
        results = retriever.retrieve(q["query"], k=k)
        rows.append({
            "id": q["id"],
            "query": q["query"],
            "expected_specialty": q["expected_specialty"],
            "category": q["category"],
            "precision_at_k": precision_at_k(results, q["expected_specialty"], k),
            "hit_at_k": hit_at_k(results, q["expected_specialty"], k),
            "mrr": reciprocal_rank(results, q["expected_specialty"]),
        })
    return pd.DataFrame(rows)
