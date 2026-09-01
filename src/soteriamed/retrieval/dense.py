"""Dense retrieval: SentenceTransformer embeddings over a FAISS index.

Scores are cosine similarity in ``[-1, 1]`` — embeddings are L2-normalised and
the index is ``IndexFlatIP``, so inner product *is* cosine. They are not
distances and higher is better. They are also not comparable with BM25's
unbounded scores; fusion happens by rank (phase 4), never by raw score.
"""

import pickle
from pathlib import Path

from soteriamed.retrieval.base import BaseRetriever


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
