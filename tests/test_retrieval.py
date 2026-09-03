"""Tests for the retrieval package (soteriamed/retrieval/).

`TestBM25Retriever` and `TestDecomposingRetriever` are offline. `TestFAISSRetriever`
downloads model weights on first run.
"""

import json
from typing import ClassVar

import pytest

from soteriamed.generation.base import StubGenerator
from soteriamed.retrieval.base import BaseRetriever
from soteriamed.retrieval.decompose import DecomposingRetriever, _chunk_key
from soteriamed.retrieval.dense import FAISSRetriever
from soteriamed.retrieval.sparse import BM25Retriever, default_tokenizer


class TestBM25Retriever:
    """Tests for BM25Retriever using the shared sample_chunks fixture."""

    @pytest.fixture(autouse=True)
    def _build_retriever(self, sample_chunks):
        self.retriever = BM25Retriever(sample_chunks, k=3)

    def test_returns_k_results(self):
        results = self.retriever.retrieve("knee pain")
        assert len(results) == 3

    def test_result_structure(self):
        results = self.retriever.retrieve("chest pain")
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "score" in r
            assert isinstance(r["score"], float)
        # No range assertion: BM25 scores are unbounded and corpus-dependent.
        # This is why phase 4 fuses by rank (RRF) and never by raw score.

    def test_relevance(self):
        results = self.retriever.retrieve("tibial plateau fracture knee")
        assert results[0]["metadata"]["chapter_title"] == "Tibial Plateau Fracture"

    def test_scores_descending(self):
        results = self.retriever.retrieve("chest pain ecg")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_greater_than_corpus(self, sample_chunks):
        retriever = BM25Retriever(sample_chunks[:2], k=5)
        results = retriever.retrieve("knee pain")
        assert len(results) == 2  # no crash, returns available

    def test_empty_query(self):
        assert self.retriever.retrieve("") == []

    def test_empty_corpus(self):
        assert BM25Retriever([], k=3).retrieve("knee pain") == []

    def test_vocabulary_size(self):
        assert self.retriever.get_vocabulary_size() > 0

    def test_tokenizer_is_injectable(self, sample_chunks):
        """Phase 2 passes a clean_text-based tokenizer here; it must be honoured."""
        calls = []

        def counting_tokenizer(text):
            calls.append(text)
            return default_tokenizer(text)

        retriever = BM25Retriever(sample_chunks, k=2, tokenizer=counting_tokenizer)
        retriever.retrieve("knee pain")
        assert len(calls) == len(sample_chunks) + 1  # every chunk, then the query


@pytest.fixture(scope="module")
def faiss_retriever(sample_chunks):
    return FAISSRetriever(sample_chunks, k=3, show_progress=False)


class TestFAISSRetriever:
    """Tests for FAISSRetriever. Uses a module-scoped fixture so the embedding
    model is loaded only once per test run."""

    def test_returns_k_results(self, faiss_retriever):
        results = faiss_retriever.retrieve("knee pain")
        assert len(results) == 3

    def test_result_structure(self, faiss_retriever):
        results = faiss_retriever.retrieve("chest pain")
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "score" in r
            assert isinstance(r["score"], float)
            assert -1.0 <= r["score"] <= 1.0  # cosine: IndexFlatIP on unit vectors

    def test_relevance(self, faiss_retriever):
        results = faiss_retriever.retrieve("my knee hurts and it is swollen")
        assert results[0]["metadata"]["chapter_id"] in {"SP-0001", "SP-0004"}

    def test_scores_descending(self, faiss_retriever):
        results = faiss_retriever.retrieve("chest pain ecg")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_greater_than_corpus(self, sample_chunks):
        retriever = FAISSRetriever(sample_chunks[:2], k=5, show_progress=False)
        results = retriever.retrieve("knee pain")
        assert len(results) == 2

    def test_embedding_dim(self, faiss_retriever):
        assert faiss_retriever.get_embedding_dim() > 0

    def test_rechunking_at_the_same_path_does_not_reload_the_old_corpus(
        self, sample_chunks, tmp_path
    ):
        """Re-chunking with the same `index_path` must invalidate the cache.

        Regression test. The cache used to be keyed on `CACHE_FORMAT` alone,
        which describes embedding semantics and says nothing about which chunks
        were embedded. Re-chunk the corpus, keep the path, and the constructor
        discarded the `chunks` it had been handed and silently loaded the
        previous corpus from `chunks.pkl` -- every downstream number then
        described a corpus that no longer existed, with no error to notice.
        """
        cache = tmp_path / "index"

        first = FAISSRetriever(
            sample_chunks[:2], k=1, index_path=cache, show_progress=False
        )
        assert first.retrieve("chest pain")[0]["metadata"]["chapter_id"] in {
            "SP-0001",
            "SP-0002",
        }

        replacement = [
            {
                "text": "Acute appendicitis presents with periumbilical pain "
                "migrating to the right iliac fossa, with anorexia and low-grade "
                "fever.",
                "metadata": {
                    "chapter_id": "SP-9001",
                    "chapter_title": "Acute Appendicitis",
                    "section": "History and Physical",
                    "chunk_index": 0,
                },
            }
        ]
        second = FAISSRetriever(replacement, k=1, index_path=cache, show_progress=False)

        results = second.retrieve("stomach ache")
        assert len(results) == 1
        assert results[0]["metadata"]["chapter_id"] == "SP-9001"

    def test_an_unchanged_corpus_still_hits_the_cache(self, sample_chunks, tmp_path):
        """The fingerprint must not defeat caching in the case it exists to allow."""
        cache = tmp_path / "index"

        FAISSRetriever(sample_chunks, k=1, index_path=cache, show_progress=False)
        stamp = (cache / "format.txt").read_text()

        FAISSRetriever(sample_chunks, k=1, index_path=cache, show_progress=False)
        assert (cache / "format.txt").read_text() == stamp

        fmt, fingerprint = stamp.split()
        assert fmt == FAISSRetriever.CACHE_FORMAT
        assert len(fingerprint) == 64  # sha256 hex


class _FakeInner(BaseRetriever):
    """Returns a different result set per query, used to verify merging.

    Every chunk here carries `chunk_index` 0 and a distinct `chapter_id`, which
    is the ordinary shape of chunked StatPearls output and the exact shape that
    broke the old merge key.
    """

    RESULTS: ClassVar[dict[str, list[dict]]] = {
        "chest pain": [
            {
                "text": "acute chest pain with ST elevation",
                "metadata": {
                    "chapter_id": "SP-0002",
                    "chapter_title": "Acute Coronary Syndrome",
                    "section": "History and Physical",
                    "chunk_index": 0,
                },
                "score": 0.95,
            },
            {
                "text": "stable angina on exertion",
                "metadata": {
                    "chapter_id": "SP-0006",
                    "chapter_title": "Stable Angina",
                    "section": "History and Physical",
                    "chunk_index": 0,
                },
                "score": 0.60,
            },
        ],
        "shortness of breath": [
            {
                "text": "exertional dyspnea and peripheral oedema",
                "metadata": {
                    "chapter_id": "SP-0005",
                    "chapter_title": "Congestive Heart Failure",
                    "section": "History and Physical",
                    "chunk_index": 0,
                },
                "score": 0.80,
            },
            # The same chunk as the chest-pain top hit, at a lower score: the
            # merge must keep 0.95, not 0.50.
            {
                "text": "acute chest pain with ST elevation",
                "metadata": {
                    "chapter_id": "SP-0002",
                    "chapter_title": "Acute Coronary Syndrome",
                    "section": "History and Physical",
                    "chunk_index": 0,
                },
                "score": 0.50,
            },
        ],
    }

    def retrieve(self, query, k=None):
        results = self.RESULTS.get(query, [])
        return results[: (k or len(results))]


def _stub(*queries: str) -> StubGenerator:
    """A generator that decomposes into exactly *queries*."""
    return StubGenerator(json.dumps({"queries": list(queries)}))


class TestChunkKey:
    """`_chunk_key` carries the whole correctness of the merge in one function."""

    def test_prefers_the_statpearls_coordinates(self):
        result = {
            "text": "whatever",
            "metadata": {
                "chapter_id": "SP-0002",
                "chapter_title": "Acute Coronary Syndrome",
                "section": "Evaluation",
                "chunk_index": 3,
            },
        }
        assert _chunk_key(result) == ("SP-0002", "Evaluation", 3)

    def test_falls_back_to_the_text_rather_than_to_a_constant(self):
        """Missing metadata must not collapse distinct chunks onto one key."""
        a = {"text": "first chunk", "metadata": {"doc_id": 0, "chunk_index": 0}}
        b = {"text": "second chunk", "metadata": {"doc_id": 1, "chunk_index": 0}}

        assert _chunk_key(a) == "first chunk"
        assert _chunk_key(a) != _chunk_key(b)

    def test_a_partial_metadata_shape_still_falls_back(self):
        result = {"text": "only a chapter id", "metadata": {"chapter_id": "SP-0001"}}
        assert _chunk_key(result) == "only a chapter id"


class TestDecomposingRetriever:
    """Generator-assisted fan-out retriever. Offline -- no weights."""

    def test_merges_subqueries_keeping_max_score(self):
        retriever = DecomposingRetriever(
            _FakeInner(),
            _stub("chest pain", "shortness of breath"),
            max_subqueries=3,
            k=3,
        )

        results = retriever.retrieve("chest pain and shortness of breath", k=3)

        assert len(results) == 3
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        # The duplicated chunk must keep its highest score (0.95).
        first = next(r for r in results if r["metadata"]["chapter_id"] == "SP-0002")
        assert first["score"] == pytest.approx(0.95)

    def test_chunks_from_different_chapters_are_not_merged(self):
        """`chunk_index` alone is not an identity.

        All four results across the two sub-queries carry `chunk_index` 0, so a
        merge key that misses `chapter_id` collapses them to one entry and the
        retriever returns one chunk where it should return three -- without
        raising, so the arm just looks worse than it is.
        """
        retriever = DecomposingRetriever(
            _FakeInner(), _stub("chest pain", "shortness of breath"), k=5
        )

        results = retriever.retrieve("chest pain and shortness of breath", k=5)

        chapters = [r["metadata"]["chapter_id"] for r in results]
        assert sorted(chapters) == ["SP-0002", "SP-0005", "SP-0006"]

    def test_falls_back_to_original_query_when_decomposition_is_empty(self):
        retriever = DecomposingRetriever(_FakeInner(), _stub(), k=2)

        results = retriever.retrieve("chest pain", k=2)
        assert len(results) == 2
        assert results[0]["score"] == pytest.approx(0.95)

    def test_falls_back_when_the_generator_returns_unusable_output(self):
        """A model that will not emit JSON degrades to a no-op, not a crash."""
        retriever = DecomposingRetriever(
            _FakeInner(), StubGenerator("sorry, no JSON here"), k=2
        )

        assert retriever.get_subqueries("chest pain") == ["chest pain"]
        assert len(retriever.retrieve("chest pain", k=2)) == 2

    def test_get_subqueries_exposes_decomposition(self):
        retriever = DecomposingRetriever(
            _FakeInner(), _stub("chest pain", "shortness of breath")
        )
        assert retriever.get_subqueries("anything") == [
            "chest pain",
            "shortness of breath",
        ]

    def test_subqueries_are_deduplicated_and_stripped(self):
        stub = _stub("  chest pain ", "CHEST PAIN", "", "cough")
        retriever = DecomposingRetriever(_FakeInner(), stub)
        assert retriever.get_subqueries("anything") == ["chest pain", "cough"]

    def test_subqueries_are_capped_at_max_subqueries(self):
        stub = _stub("one", "two", "three", "four")
        retriever = DecomposingRetriever(_FakeInner(), stub, max_subqueries=2)
        assert retriever.get_subqueries("anything") == ["one", "two"]

    def test_the_query_reaches_the_generator(self):
        stub = _stub("chest pain")
        DecomposingRetriever(_FakeInner(), stub).get_subqueries("my chest feels tight")
        assert "my chest feels tight" in stub.prompts[0]
