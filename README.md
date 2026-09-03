# SoteriaMed

A hybrid RAG system for symptom triage, with safety rules built into the pipeline.

You describe your symptoms in your own words. The system returns a ranked list of possible conditions, a suggested specialty, and how urgently you should be seen. Every answer is grounded in retrieved clinical text and cites what it used.

> [!IMPORTANT]
> **This is a master's dissertation research project. It is not a medical device.** It is not clinically validated, has not been reviewed by any regulator, and must not be used for real medical decisions by anyone. It is not deployed to real users, and it processes no real patient data. Every evaluation in this repository is run against synthetic cases.

**Status:** early development. See [Roadmap](#roadmap) for what is built and what is not.

---

## The research question

Patients and clinical literature do not use the same vocabulary. A patient writes *"my chest feels tight going up the stairs"*. The literature says *"exertional chest tightness"*. A retrieval system that matches on surface form loses the connection between them.

The thesis this project tests:

> A hybrid RAG architecture with safety constraints gives more accurate and better grounded triage recommendations than a plain RAG baseline, when the symptoms are described in everyday language.

The architecture has four parts, and each one is a claim to be tested separately: deterministic interception of red flags, query normalisation guided by an ontology, hybrid dense and sparse retrieval with cross-encoder reranking, and generation constrained to a schema. Each part is an ablation arm, and each arm is measured against a frozen benchmark.

---

## Architecture

Query flows top to bottom. Nothing below the guardrail runs if the guardrail fires.

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | Guardrail | `guardrail/` | Deterministic red flag rules. **No model participates.** A hit exits immediately with an emergency recommendation. |
| 2 | Normalisation | `ontology/` | Biomedical NER finds symptom mentions. HPO maps them to canonical terms and expands them with synonyms. |
| 3 | Retrieval | `retrieval/` | Dense (FAISS) and sparse (BM25) retrieval, fused by rank, then reranked by a cross-encoder. |
| 4 | Generation | `generation/` | Prompt assembly, a JSON call constrained to a schema, and validation of what comes back. |
| 5 | Evaluation | `evaluation/` | Retrieval quality, answer faithfulness, and grounding, against a frozen benchmark. |

Three design rules shape the code more than anything else:

- **The emergency path is deterministic.** No probability decides whether something is an emergency.
- **Retrievers compose.** Every retriever implements one interface, and wrappers are retrievers themselves. So `HPOExpansion(Reranked(Hybrid([dense, sparse])))` is a valid configuration, and every ablation arm is one line.
- **No orchestration framework.** Prompt construction, context assembly, and response parsing are all project code. The thesis has to describe them precisely, so they are not delegated to a framework's abstractions.

---

## Data

Two sources, kept strictly separate. The corpus is what the system reads. The benchmark is what it is graded against. Mixing them would let the system retrieve its own answer key.

| | Source | Role |
|---|---|---|
| **Corpus** | [StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK430685/) | Clinical reference text, retrieved at answer time |
| **Benchmark** | [DDXPlus](https://github.com/mila-iqia/ddxplus) | Synthetic patient cases with known correct pathologies |
| **Ontology** | [HPO](https://hpo.jax.org/) | Symptom term normalisation and synonyms |

DDXPlus stores coded evidences rather than patient text, so the narratives are generated from those codes. They are written deliberately in **everyday** vocabulary. Writing them in clinical phrasing would erase the vocabulary gap this project exists to measure. A clinical variant is generated too, but only as an ablation arm.

Once an evaluation set is written it is immutable. A change means a new version number, and every result records which version produced it.

---

## Getting started

Requires Python 3.11+.

```bash
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1

pip install -e ".[dev]"
python scripts/bootstrap.py       # spaCy model and NLTK corpora; pip cannot install these
python -m pytest
```

`scripts/bootstrap.py` is not optional. Pip cannot install the spaCy model or the NLTK corpora it fetches, and text processing fails without them.

The test suite runs offline, apart from the dense retrieval tests. Those download embedding model weights the first time.

---

## Repository layout

```
src/soteriamed/
├── config.py       typed experiment config loading
├── data/           benchmark loading, narrative generation
├── corpus/         corpus fetch, chunking, text cleaning
├── ontology/       HPO parsing, lookup, expansion
├── retrieval/      base, dense, sparse, decompose, hybrid, rerank, hpo
├── guardrail/      deterministic red flag rules
├── generation/     generator interface, prompts, schema, parsing
└── evaluation/     metrics, runners, aggregation

scripts/            entry point scripts, one command each
configs/            one YAML per experiment
results/            committed results, each with the config that produced it
tests/
notebooks/poc/      the superseded proof of concept, kept as a record
app/                demo API and page
```

Every experiment is a config file, so there are no model names hardcoded in experiment code. Every committed result carries the config that produced it. Notebooks import from the package, never the reverse.

---

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Frozen evaluation set and narrative generator | in progress |
| 2 | Corpus, dense index, **baseline measurement** | |
| 3 | Ontology normalisation layer | |
| 4 | Hybrid retrieval and reranking | |
| 5 | Guardrail and generation over the full pipeline | |
| 6 | Full evaluation and error analysis | |
| 7 | Ablations | |
| 8 | Demo and write up | |

Nothing gets built before there is a measurement that can show whether it helped. Phase 2's baseline gates everything after it.

---

## Licence

The code is MIT licensed. See [LICENSE](LICENSE).

That does not cover the datasets or the ontology. They are not redistributed here and they keep their own terms. StatPearls is CC BY-NC-ND 4.0. DDXPlus is CC BY 4.0. Both are used for non-commercial academic research only. `scripts/` fetches them, and `data/` is ignored by git apart from the frozen evaluation sets.
