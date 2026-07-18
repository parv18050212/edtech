# RAG Evaluation Results (RAGAS)

Run via `python scripts/evaluate_rag.py` — evaluates the **real** pipeline
(query embedding → pgvector retrieval → Groq generation with the production
chat prompt) over 8 questions spanning both books and varied chapters.

Judge LLM: Groq `llama-3.3-70b-versatile`. Embeddings: local Ollama
`embeddinggemma`. All three metrics are reference-free (no hand-labelled
answers required).

## Aggregate scores (first run, 2026-07-19)

| Metric | Score | Meaning |
|---|---|---|
| answer_relevancy | 0.95 | Answer addresses the question |
| llm_context_precision_without_reference | 0.86 | Retrieved chunks are relevant (retrieval quality) |
| faithfulness | 0.77 | Answer claims are grounded in retrieved chunks (anti-hallucination) |

## Per-question

| # | Question | faithfulness | answer_relevancy | context_precision |
|---|---|---|---|---|
| 1 | What is a star and how is it different from a planet? | 0.87 | 0.94 | 1.00 |
| 2 | What is the solar system made up of? | 0.81 | 0.97 | 0.95 |
| 3 | What are the different sources of water? | 0.33 | 1.00 | 1.00 |
| 4 | What are the main constituents of food? | nan* | 0.87 | 0.37 |
| 5 | What are the five kingdoms in the classification of living organisms? | 1.00 | 1.00 | 0.83 |
| 6 | What is pressure and how is it related to force and area? | nan* | 1.00 | nan* |
| 7 | What are the sub-atomic particles present in an atom? | 0.75 | 0.92 | 0.87 |
| 8 | How is sound produced and how does it travel? | 0.83 | 0.88 | 1.00 |

\* `nan` = Groq free-tier rate-limit timeout during the judge call, not a
quality failure. Re-running with `RunConfig(max_workers=2)` and a higher
timeout clears these.

## Interpretation

- **Retrieval is solid** (context precision 0.86). The metadata-filtered
  pgvector search returns on-topic chunks; the one weak case (Q4,
  "constituents of food", 0.37) suggests that chapter's chunking mixes in
  less-relevant material and is worth a spot-check.
- **Answers are highly relevant** (0.95) — the pipeline answers what was
  asked.
- **Faithfulness (0.77) is the metric to improve.** The production chat
  prompt deliberately asks for a worked example and a "common mistake
  students make" — pedagogically valuable, but these additions aren't
  always verbatim in the retrieved chunks, which strict faithfulness
  penalizes (clearest in Q3, 0.33). Options: (a) accept it as an
  intentional tutoring trade-off, (b) tighten the prompt to ground every
  claim, or (c) add a citation step. This is a product decision, not a bug.
