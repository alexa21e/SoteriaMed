"""RAG chain — PubMedBERT + FAISS retrieval wired to a flan-t5 generator via LCEL."""

from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from src.retrieval import BaseRetriever, FAISSRetriever


PROMPT_TEMPLATE = (
    "You are a clinical assistant. Use the reference excerpts below to describe "
    "the most likely clinical condition and recommend which medical specialty "
    "should handle the patient presentation. Ground every claim in the excerpts.\n\n"
    "Reference excerpts:\n{context}\n\n"
    "Patient presentation: {question}\n\n"
    "Likely condition and recommended specialty:"
)

DECOMPOSE_PROMPT = (
    "Split the patient presentation below into 2 or 3 short, focused sub-queries, "
    "one per clinical symptom or organ system. Output one sub-query per line, "
    "no numbering, no extra commentary.\n\n"
    "Patient presentation: {question}\n\n"
    "Sub-queries:"
)


def build_vector_store(
    chunks: list[dict],
    cache_dir: str | Path = "data/faiss_index",
    model_name: str = "NeuML/pubmedbert-base-embeddings",
    k: int = 3,
    device: str = "auto",
) -> FAISSRetriever:
    """Build (or load from cache) a FAISSRetriever for *chunks*.

    ``device`` is forwarded to the encoder: ``"auto"`` uses CUDA if a GPU is
    visible, otherwise CPU. Pass ``"cuda"`` / ``"cpu"`` to force.
    """
    return FAISSRetriever(
        chunks=chunks,
        model_name=model_name,
        k=k,
        index_path=cache_dir,
        device=device,
    )


def build_llm(
    model_name: str = "google/flan-t5-base",
    max_new_tokens: int = 256,
) -> Runnable:
    """Return a LangChain Runnable wrapping a local seq2seq HuggingFace model.

    Uses ``model.generate()`` directly instead of the ``text2text-generation``
    pipeline (removed in Transformers 5.0). Accepts PromptValue/str inputs and
    returns a decoded answer string.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    def _generate(prompt_value: Any) -> str:
        text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return RunnableLambda(_generate)


def _format_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, start=1):
        specialty = r["metadata"].get("medical_specialty", "")
        parts.append(f"[Source {i} | {specialty}]\n{r['text']}")
    return "\n\n".join(parts)


def build_rag_chain(retriever: BaseRetriever, llm: Any) -> Runnable:
    """Wire retriever -> prompt -> llm -> parser as an LCEL runnable.

    Input: ``{"question": str}``. Output: generated answer string.
    """
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def _retrieve(inputs: dict) -> dict:
        results = retriever.retrieve(inputs["question"])
        return {"context": _format_context(results), "question": inputs["question"]}

    return RunnableLambda(_retrieve) | prompt | llm | StrOutputParser()


def query_rag(chain: Runnable, retriever: BaseRetriever, question: str) -> dict:
    """Run *question* through *chain*; also return the retrieved source chunks."""
    sources = retriever.retrieve(question)
    answer = chain.invoke({"question": question})
    return {"answer": answer, "sources": sources}


def decompose_query(llm: Runnable, query: str, max_subqueries: int = 3) -> list[str]:
    """Ask *llm* to split *query* into focused sub-queries.

    Returns up to *max_subqueries* non-empty lines from the LLM output. Falls
    back to ``[query]`` if the LLM returns nothing parseable, so callers can
    treat the result as a non-empty list of strings.
    """
    prompt = DECOMPOSE_PROMPT.format(question=query)
    raw = llm.invoke(prompt)
    text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))

    candidates: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip(" -*\t0123456789.").strip()
        if not line or line.lower() in seen:
            continue
        seen.add(line.lower())
        candidates.append(line)
        if len(candidates) >= max_subqueries:
            break

    return candidates or [query]
