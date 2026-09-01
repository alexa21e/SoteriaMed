"""Sparse retrieval: Okapi BM25 over a tokenised corpus.

BM25 scores are **unbounded and corpus-dependent** — they are not similarities
and not comparable with the dense retriever's cosine scores in ``[-1, 1]``.
Never threshold or fuse the two by raw score; phase 4 fuses by rank (RRF) for
exactly this reason.

Tokenisation is injected rather than fixed. The default is a plain word regex,
which keeps this retriever fast and dependency-free for tests. Phase 2 passes
`soteriamed.corpus.text.clean_text` (stopword removal + lemmatisation) through
`tokenizer` when indexing the real corpus — that transformation belongs to the
sparse path only, and is wrong for the dense one.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

from soteriamed.retrieval.base import BaseRetriever

_WORD_RE = re.compile(r"[a-z0-9]+")

Tokenizer = Callable[[str], list[str]]


def default_tokenizer(text: str) -> list[str]:
    """Lowercase, then split on runs of alphanumerics. No stemming, no stopwords."""
    return _WORD_RE.findall(text.lower())


class BM25Retriever(BaseRetriever):
    """Sparse retriever using Okapi BM25 (`rank-bm25`).

    Replaces the proof-of-concept's `TFIDFRetriever` — BM25 is the field-standard
    sparse baseline; the old class is recoverable from git history at `2742f59`.
    """

    def __init__(
        self,
        chunks: list[dict],
        k: int = 3,
        tokenizer: Tokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.default_k = k
        self.tokenizer = tokenizer or default_tokenizer
        self.k1 = k1
        self.b = b

        corpus = [self.tokenizer(c["text"]) for c in chunks]
        # BM25Okapi divides by the average document length; an empty corpus has none.
        self.bm25 = BM25Okapi(corpus, k1=k1, b=b) if corpus else None

    # -- public API ---------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.default_k
        n_results = min(k, len(self.chunks))
        if n_results == 0 or self.bm25 is None:
            return []

        tokens = self.tokenizer(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
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
        """Number of distinct terms BM25 computed an IDF for."""
        return len(self.bm25.idf) if self.bm25 is not None else 0
