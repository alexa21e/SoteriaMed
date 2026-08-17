"""Retrieval module — TF-IDF baseline, FAISS dense retriever, and evaluation helpers."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

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

# FAISS retriever

class FAISSRetriever(BaseRetriever):
    """Dense retriever: SentenceTransformer + FAISS IndexFlatIP on L2-normalized
    embeddings (inner product on unit vectors == cosine similarity)."""

    CACHE_FORMAT = "cosine-v1"

    def __init__(
        self,
        chunks: list[dict],
        model_name: str = "NeuML/pubmedbert-base-embeddings",
        k: int = 3,
        index_path: str | Path | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
        device: str = "auto",
    ) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.default_k = k
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = _resolve_device(device)

        self.model = SentenceTransformer(model_name, device=self.device)

        cache = Path(index_path) if index_path else None
        faiss_file = cache / "index.faiss" if cache else None
        chunks_file = cache / "chunks.pkl" if cache else None
        format_file = cache / "format.txt" if cache else None

        cache_valid = (
            cache
            and faiss_file.exists()
            and chunks_file.exists()
            and format_file.exists()
            and format_file.read_text().strip() == self.CACHE_FORMAT
        )

        if cache_valid:
            self.index = faiss.read_index(str(faiss_file))
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            texts = [c["text"] for c in chunks]
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)

            if cache:
                cache.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(faiss_file))
                with open(chunks_file, "wb") as f:
                    pickle.dump(self.chunks, f)
                format_file.write_text(self.CACHE_FORMAT)

    # -- public API ---------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.default_k
        n_results = min(k, len(self.chunks))
        if n_results == 0:
            return []

        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(q_vec, n_results)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(score),
            })
        return results

    def retrieve_batch(self, queries: list[str], k: int | None = None) -> list[list[dict]]:
        k = k or self.default_k
        n_results = min(k, len(self.chunks))
        if n_results == 0:
            return [[] for _ in queries]

        q_vecs = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(q_vecs, n_results)

        batched = []
        for row_scores, row_idx in zip(scores, indices):
            row = []
            for score, idx in zip(row_scores, row_idx):
                if idx < 0:
                    continue
                row.append({
                    "text": self.chunks[idx]["text"],
                    "metadata": self.chunks[idx]["metadata"],
                    "score": float(score),
                })
            batched.append(row)
        return batched

    def get_embedding_dim(self) -> int:
        return int(self.index.d)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' if a GPU is available, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

# Query-decomposition retriever (LLM-assisted fan-out)

class DecomposingRetriever(BaseRetriever):
    """Fan out a complex query into sub-queries via an LLM, then merge results.

    Wraps an existing ``inner`` retriever. For each query the LLM produces a
    handful of focused sub-queries (one per symptom/organ-system); the wrapper
    retrieves ``k`` results for each sub-query and merges them by **max score**
    per chunk, returning the global top-``k``.

    When the LLM returns a single line the behaviour is identical to the inner
    retriever — making this a drop-in replacement that only helps on multi-
    symptom queries.
    """

    def __init__(
        self,
        inner: BaseRetriever,
        llm: "Runnable",  # noqa: F821 - typing-only forward ref
        max_subqueries: int = 3,
        k: int | None = None,
    ) -> None:
        self.inner = inner
        self.llm = llm
        self.max_subqueries = max_subqueries
        self.default_k = k if k is not None else getattr(inner, "default_k", 3)

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        from src.rag_chain import decompose_query

        k = k or self.default_k
        subqueries = decompose_query(self.llm, query, self.max_subqueries)

        best: dict[tuple, dict] = {}
        for sq in subqueries:
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
        """Expose the LLM's decomposition for inspection in notebooks/tests."""
        from src.rag_chain import decompose_query
        return decompose_query(self.llm, query, self.max_subqueries)


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
