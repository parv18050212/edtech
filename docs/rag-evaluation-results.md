# RAG Evaluation Results (RAGAS)

Run via `python scripts/evaluate_rag.py` — evaluates the pipeline
(query embedding → pgvector retrieval → generation with the production
chat prompt) over 8 questions spanning both books and varied chapters.

All three metrics are reference-free (no hand-labelled answers required):

- **faithfulness** — are the answer's claims grounded in the retrieved
  chunks (anti-hallucination)?
- **answer_relevancy** — does the answer actually address the question?
- **llm_context_precision_without_reference** — are the retrieved chunks
  relevant to the question (retrieval quality)?

## Authoritative run — fully local (2026-07-19)

Generation + RAGAS judge: local Ollama `llama3.1:8b` (via Ollama's
OpenAI-compatible endpoint). Embeddings: local Ollama `embeddinggemma`.
Fully local, no API rate limits — every question scored cleanly, no
`nan`.

| Metric | Score |
|---|---|
| llm_context_precision_without_reference | **0.99** |
| answer_relevancy | **0.82** |
| faithfulness | **0.68** |

### Per-question

| # | Question | faithfulness | answer_relevancy | context_precision |
|---|---|---|---|---|
| 1 | What is a star and how is it different from a planet? | 0.73 | 0.81 | 1.00 |
| 2 | What is the solar system made up of? | 0.77 | 0.97 | 1.00 |
| 3 | What are the different sources of water? | 0.46 | 0.89 | 1.00 |
| 4 | What are the main constituents of food? | 0.24 | 1.00 | 1.00 |
| 5 | What are the five kingdoms in the classification of living organisms? | 0.69 | 0.67 | 0.95 |
| 6 | What is pressure and how is it related to force and area? | 0.77 | 0.73 | 1.00 |
| 7 | What are the sub-atomic particles present in an atom? | 0.80 | 0.87 | 1.00 |
| 8 | How is sound produced and how does it travel? | 1.00 | 0.66 | 1.00 |

## Interpretation

- **Retrieval is excellent (context precision 0.99).** The
  metadata-filtered pgvector search reliably returns the right chunks —
  7 of 8 questions scored a perfect 1.00. This is the core RAG mechanic
  and it is solid.
- **Answers are relevant (0.82)** — the pipeline answers what was asked.
- **Faithfulness (0.68) is the metric to improve.** Q4 ("constituents of
  food", 0.24) is the weakest. Two contributing factors: (1) the local
  8B generator hallucinates somewhat more than a larger model, and (2)
  the production chat prompt deliberately asks for a worked example and a
  "common mistake students make" — pedagogically valuable, but those
  additions aren't always verbatim in the retrieved chunks, which strict
  faithfulness penalizes. Options: accept it as an intentional tutoring
  trade-off, tighten the prompt to ground every claim, or add a citation
  step. This is a product decision, not a bug.

## Earlier reference run — Groq judge (2026-07-19, partial)

An earlier run used Groq `llama-3.3-70b-versatile` for both generation
and judging. It is **not authoritative**: the Groq free-tier
tokens-per-minute rate limit caused several judge calls to time out into
`nan`, and the aggregates were computed over the surviving rows only.
Recorded here only for context.

| Metric | Score | Note |
|---|---|---|
| answer_relevancy | 0.95 | over non-nan rows |
| llm_context_precision_without_reference | 0.86 | dragged down by rate-limit nans |
| faithfulness | 0.77 | over non-nan rows |

Relevancy/faithfulness read higher there because the 70B generator is
stronger than the local 8B used in the authoritative run. In production,
generation uses Groq 70B while this local harness uses 8B — so expect
production relevancy/faithfulness to land nearer the Groq column, with
retrieval quality (context precision) matching the clean local 0.99.

## Setup notes

- ragas deps are pinned in a separate `requirements-eval.txt` (langchain
  0.3.x) because ragas 0.4.x breaks against the current langchain 1.x
  (it imports `langchain_community.chat_models.vertexai`, removed in
  langchain-community 0.4.x).
- The harness points RAGAS at local Ollama through its OpenAI-compatible
  endpoint (`http://localhost:11434/v1`), so no extra langchain-ollama
  dependency is needed. Requires `ollama pull llama3.1:8b`.
