"""Tests for the retrieval package (soteriamed/retrieval/).

`TestBM25Retriever` and `TestDecomposingRetriever` are offline. `TestFAISSRetriever`
downloads model weights on first run.
"""

from typing import ClassVar

import pytest

from soteriamed.generation.base import StubGenerator
from soteriamed.retrieval.base import BaseRetriever
from soteriamed.retrieval.decompose import DecomposingRetriever
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
        results = self.retriever.retrieve("knee fracture orthopedic")
        assert results[0]["metadata"]["medical_specialty"] == "Orthopedic"

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
def sample_chunks_module():
    return [
        {
            "text": "Patient presents with acute knee pain after a fall. "
            "X-ray shows a fracture of the tibial plateau. "
            "Orthopedic consultation requested for surgical repair.",
            "metadata": {
                "source_index": 0,
                "medical_specialty": "Orthopedic",
                "sample_name": "Knee Fracture",
                "chunk_index": 0,
            },
        },
        {
            "text": "The patient was admitted with substernal chest pain "
            "radiating to the left arm. ECG shows ST elevation. "
            "Troponin levels elevated consistent with acute MI.",
            "metadata": {
                "source_index": 1,
                "medical_specialty": "Cardiovascular / Pulmonary",
                "sample_name": "Acute MI",
                "chunk_index": 0,
            },
        },
        {
            "text": "Colonoscopy performed for evaluation of rectal bleeding. "
            "Multiple polyps found in the sigmoid colon. "
            "Biopsies taken and sent for pathology.",
            "metadata": {
                "source_index": 2,
                "medical_specialty": "Gastroenterology",
                "sample_name": "Colonoscopy",
                "chunk_index": 0,
            },
        },
        {
            "text": "Right knee MRI reveals a complete tear of the anterior "
            "cruciate ligament with associated bone bruise. "
            "Arthroscopic ACL reconstruction recommended.",
            "metadata": {
                "source_index": 3,
                "medical_specialty": "Orthopedic",
                "sample_name": "ACL Tear",
                "chunk_index": 0,
            },
        },
        {
            "text": "Echocardiogram shows ejection fraction of 35 percent. "
            "Patient has dyspnea on exertion and bilateral lower "
            "extremity edema consistent with congestive heart failure.",
            "metadata": {
                "source_index": 4,
                "medical_specialty": "Cardiovascular / Pulmonary",
                "sample_name": "CHF",
                "chunk_index": 0,
            },
        },
    ]


@pytest.fixture(scope="module")
def faiss_retriever(sample_chunks_module):
    return FAISSRetriever(sample_chunks_module, k=3, show_progress=False)


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
        results = faiss_retriever.retrieve("knee fracture")
        assert results[0]["metadata"]["medical_specialty"] == "Orthopedic"

    def test_scores_descending(self, faiss_retriever):
        results = faiss_retriever.retrieve("chest pain ecg")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_greater_than_corpus(self, sample_chunks_module):
        retriever = FAISSRetriever(sample_chunks_module[:2], k=5, show_progress=False)
        results = retriever.retrieve("knee pain")
        assert len(results) == 2

    def test_embedding_dim(self, faiss_retriever):
        assert faiss_retriever.get_embedding_dim() > 0


class _FakeInner(BaseRetriever):
    """Returns a different result set per query, used to verify merging."""

    RESULTS: ClassVar[dict[str, list[dict]]] = {
        "chest pain": [
            {
                "text": "acute chest pain ECG",
                "metadata": {
                    "source_index": 0,
                    "chunk_index": 0,
                    "medical_specialty": "Cardiovascular / Pulmonary",
                },
                "score": 0.95,
            },
            {
                "text": "stable angina",
                "metadata": {
                    "source_index": 1,
                    "chunk_index": 0,
                    "medical_specialty": "Cardiovascular / Pulmonary",
                },
                "score": 0.60,
            },
        ],
        "shortness of breath": [
            {
                "text": "exertional dyspnea",
                "metadata": {
                    "source_index": 2,
                    "chunk_index": 0,
                    "medical_specialty": "Cardiovascular / Pulmonary",
                },
                "score": 0.80,
            },
            # duplicate of the chest-pain top result, but lower score -> should be dropped
            {
                "text": "acute chest pain ECG",
                "metadata": {
                    "source_index": 0,
                    "chunk_index": 0,
                    "medical_specialty": "Cardiovascular / Pulmonary",
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
    import json

    return StubGenerator(json.dumps({"queries": list(queries)}))


class TestDecomposingRetriever:
    """Generator-assisted fan-out retriever. Offline — no weights."""

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
        # The duplicate chunk (source_index=0) must keep its highest score (0.95).
        first = next(r for r in results if r["metadata"]["source_index"] == 0)
        assert first["score"] == pytest.approx(0.95)

    def test_falls_back_to_original_query_when_decomposition_is_empty(self):
        retriever = DecomposingRetriever(_FakeInner(), _stub(), k=2)

        results = retriever.retrieve("chest pain", k=2)
        assert len(results) == 2
        assert results[0]["score"] == pytest.approx(0.95)

    def test_falls_back_when_the_generator_returns_unusable_output(self):
        """A model that will not emit JSON degrades to a no-op, not a crash."""
        retriever = DecomposingRetriever(
            _FakeInner(), StubGenerator("sorry, I can't"), k=2
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
