"""Smoke tests for src/rag_chain.py using a mock LLM (no model downloads)."""

from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import Runnable

from src.rag_chain import build_rag_chain, query_rag
from src.retrieval import TFIDFRetriever


class TestRagChain:
    def test_build_rag_chain_returns_runnable(self, sample_chunks):
        retriever = TFIDFRetriever(sample_chunks, k=3)
        llm = FakeListLLM(responses=["Patient has acute MI."])
        chain = build_rag_chain(retriever, llm)
        assert isinstance(chain, Runnable)

    def test_query_rag_returns_answer_and_sources(self, sample_chunks):
        retriever = TFIDFRetriever(sample_chunks, k=3)
        llm = FakeListLLM(responses=["Patient has acute MI."])
        chain = build_rag_chain(retriever, llm)

        out = query_rag(chain, retriever, "chest pain")
        assert "answer" in out
        assert "sources" in out
        assert isinstance(out["answer"], str)
        assert len(out["sources"]) > 0
        for src in out["sources"]:
            assert "text" in src
            assert "metadata" in src
            assert "score" in src

    def test_chain_invoke_returns_string(self, sample_chunks):
        retriever = TFIDFRetriever(sample_chunks, k=3)
        llm = FakeListLLM(responses=["stub answer"])
        chain = build_rag_chain(retriever, llm)

        result = chain.invoke({"question": "knee injury"})
        assert isinstance(result, str)
        assert result == "stub answer"
