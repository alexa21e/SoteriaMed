"""Abstract retriever interface.

Every retriever in the system subclasses :class:`BaseRetriever` (invariant 5),
including the composing ones — ``Reranked(Hybrid([dense, sparse]))`` works
because a wrapper is itself a retriever.
"""

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Return top-k results for *query*.

        Each result is ``{"text": str, "metadata": dict, "score": float}``.
        """

    def retrieve_batch(
        self, queries: list[str], k: int | None = None
    ) -> list[list[dict]]:
        """Run *retrieve* for every query in *queries*."""
        return [self.retrieve(q, k=k) for q in queries]
