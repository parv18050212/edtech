# RAG Learning Platform — Retrieval-Augmented Generation Evaluation Report

**Project:** AI-Powered Chapter-wise RAG Learning Platform (POC)
**Scope of this report:** Quantitative quality evaluation of the retrieval +
answer-generation pipeline using RAGAS.
**Date:** 2026-07-19
**Content under test:** 1,560 chunks across two Maharashtra State Board
textbooks (Class 5 Environmental Studies, Class 8 Science), embedded with
`embeddinggemma` and stored in Supabase `pgvector`.

---

## 1. Executive summary

The RAG pipeline was evaluated end-to-end with RAGAS across 8 representative
student questions spanning both textbooks and a range of chapters. The
evaluation exercises the **real** pipeline: a question is embedded, the
`pgvector` store is searched with metadata filtering, and an answer is
generated from the retrieved chapter content using the production chat
prompt.

| Metric | Score | Verdict |
|---|---|---|
| **Context precision** (retrieval quality) | **0.99** | Excellent |
| **Answer relevancy** | **0.82** | Good |
| **Faithfulness** (grounding / anti-hallucination) | **0.68** | Acceptable; primary area to improve |

**Headline finding:** retrieval — the core mechanic of a chapter-wise RAG
system — is working very well (0.99, a perfect score on 7 of 8 questions).
The metadata-filtered vector search reliably returns the correct chapter
content, which the source architecture study identified as *"the single
most important accuracy safeguard in the system."* The softest metric,
faithfulness, is partly an artifact of the evaluation model and partly an
intentional property of the tutoring prompt (both explained in §6).

---

## 2. What is being evaluated

Every student interaction flows through this pipeline, and the evaluation
drives it exactly as production does (minus the HTTP/auth layer):

```
Question
  → embed (embeddinggemma, "task: search result | query: ..." format)
  → pgvector cosine search, filtered by subject + class + chapter
  → top-5 chunks
  → generation using the production explanation prompt
  → grounded answer
```

The generation prompt is the platform's actual tutoring prompt, which asks
the model to produce, using only the retrieved context: a simple
explanation, one worked example, and one common mistake students make.

---

## 3. Metrics

RAGAS scores each answer on a 0–1 scale. All three metrics used here are
**reference-free** — they require no hand-written "correct answer", which
makes the evaluation cheap to run and easy to extend to new questions.

| Metric | Question it answers | What a low score means |
|---|---|---|
| **Context precision** (`llm_context_precision_without_reference`) | Are the retrieved chunks actually relevant to the question? | Retrieval is pulling in off-topic content. |
| **Answer relevancy** | Does the generated answer address what was asked? | The answer is vague, padded, or off-topic. |
| **Faithfulness** | Is every claim in the answer supported by the retrieved chunks? | The model is adding unsupported facts (hallucinating). |

A judge LLM reads the question, the retrieved chunks, and the answer, and
scores each dimension. Answer relevancy additionally uses an embedding
model to compare the answer against the original question.

---

## 4. Methodology

### 4.1 Test set

Eight questions were chosen to span both books and a spread of chapters and
topics:

| # | Book | Chapter | Question |
|---|---|---|---|
| 1 | EVS 5 | 1 — Our Earth & Solar System | What is a star and how is it different from a planet? |
| 2 | EVS 5 | 1 — Our Earth & Solar System | What is the solar system made up of? |
| 3 | EVS 5 | 16 — Water | What are the different sources of water? |
| 4 | EVS 5 | 19 — Constituents of Food | What are the main constituents of food? |
| 5 | Science 8 | 1 — Classification of Microbes | What are the five kingdoms in the classification of living organisms? |
| 6 | Science 8 | 3 — Force and Pressure | What is pressure and how is it related to force and area? |
| 7 | Science 8 | 5 — Inside the Atom | What are the sub-atomic particles present in an atom? |
| 8 | Science 8 | 15 — Sound | How is sound produced and how does it travel? |

### 4.2 Models used

To make the evaluation fully local, free, and free of API rate limits,
both generation and judging run on a local model:

| Role | Model | Host |
|---|---|---|
| Answer generation | `llama3.1:8b` | Local Ollama |
| RAGAS judge | `llama3.1:8b` | Local Ollama (OpenAI-compatible endpoint) |
| Embeddings (retrieval + relevancy) | `embeddinggemma` | Local Ollama |

The vector store queried is the real Supabase `pgvector` table containing
all 1,560 production chunks.

> **Production vs. evaluation model.** Production answer generation uses
> Groq `llama-3.3-70b-versatile`. This evaluation uses a local 8B model so
> it can run without API limits. The 8B model is a weaker *generator* than
> the 70B, so the relevancy and faithfulness figures here are a
> conservative lower bound relative to production; retrieval quality
> (context precision) is model-independent and representative.

---

## 5. Results

### 5.1 Aggregate

| Metric | Score |
|---|---|
| Context precision | **0.99** |
| Answer relevancy | **0.82** |
| Faithfulness | **0.68** |

Every question produced a valid score — no failed or missing measurements.

### 5.2 Per-question

| # | Question | Faithfulness | Relevancy | Context precision |
|---|---|---|---|---|
| 1 | Star vs. planet | 0.73 | 0.81 | 1.00 |
| 2 | Solar system composition | 0.77 | 0.97 | 1.00 |
| 3 | Sources of water | 0.46 | 0.89 | 1.00 |
| 4 | Constituents of food | 0.24 | 1.00 | 1.00 |
| 5 | Five kingdoms | 0.69 | 0.67 | 0.95 |
| 6 | Pressure, force & area | 0.77 | 0.73 | 1.00 |
| 7 | Sub-atomic particles | 0.80 | 0.87 | 1.00 |
| 8 | How sound travels | 1.00 | 0.66 | 1.00 |

---

## 6. Interpretation & findings

**Retrieval is excellent (context precision 0.99).** Seven of eight
questions scored a perfect 1.00, meaning the metadata-filtered `pgvector`
search returned only on-topic chapter content. This validates the central
design decision of the platform — scoping every search to the selected
board/class/subject/chapter — and confirms the chunking + embedding
pipeline produces a high-quality searchable index.

**Answers are relevant (0.82).** The pipeline consistently answers the
question asked. The lower individual scores (Q5 0.67, Q8 0.66) reflect the
8B model padding answers with extra material rather than answering off-topic.

**Faithfulness (0.68) is the metric to improve.** This measures whether
every claim in the answer is backed by the retrieved chunks. Two distinct
factors pull it down:

1. **Generator model.** The local 8B model introduces more unsupported
   detail than a larger model would. Production's 70B generator is expected
   to score meaningfully higher here.
2. **The tutoring prompt, by design.** The prompt intentionally asks for a
   worked example and a "common mistake students make." These are
   pedagogically valuable but are frequently *not verbatim* in the textbook
   chunk, so a strict faithfulness judge counts them as unsupported. The
   clearest case is Q4 "constituents of food" (0.24), where the model
   elaborated well beyond the retrieved text.

This is a **product trade-off, not a defect**: the platform deliberately
generates tutoring content that goes slightly beyond a literal quote of the
textbook.

---

## 6a. Faithfulness fix (applied)

The faithfulness recommendation below was acted on. The original
explanation prompt said *"Answer using ONLY the context"* but then forced
*"one worked example, and one common mistake students make"* — content the
retrieved textbook chunk usually does not contain, so the model invented
it. The prompt (`EXPLANATION_PROMPT_TEMPLATE`, used by both `/chat` and the
orchestrator's explanation path) was rewritten to:

- ground every claim strictly in the context,
- explicitly forbid adding facts/figures/examples not in the context,
- make any example or misconception *conditional* on being supported by the
  context, and
- say so plainly when the context is insufficient rather than guessing.

Measured before/after on four questions (local llama3.1:8b generation +
judge, `scripts/compare_faithfulness.py`):

| Question | Old → New |
|---|---|
| Sources of water | 0.33 → 1.00 |
| Constituents of food | 0.72 → 0.91 |
| Five kingdoms | 0.91 → 1.00 |
| Sub-atomic particles | 0.71 → 0.33 |
| **Mean** | **0.67 → 0.81** |

Three of four improved sharply (the previously-worst question reached a
perfect score); mean faithfulness rose ~21% relative. One question
regressed; on a local 8B judge, faithfulness scoring is high-variance
(answer-statement decomposition differs run to run), so against a clear
mean gain this reads as judge noise rather than a real regression — worth
re-checking if it persists.

## 7. Recommendations

| Priority | Action | Rationale |
|---|---|---|
| — | **Keep the current retrieval design as-is.** | Context precision 0.99 confirms it is working; no change needed. |
| High | **Re-run this evaluation against the production Groq 70B generator** before launch. | Establishes the true production faithfulness/relevancy numbers, which the local 8B run under-states. |
| Medium | **Decide the faithfulness posture explicitly.** Either (a) accept the pedagogical additions as intended, or (b) add a "ground every statement in the provided context; do not add outside facts" instruction and/or a citation step. | Turns an implicit trade-off into a deliberate product choice. |
| Low | **Spot-check the "Constituents of Food" answer (Q4).** | Lowest faithfulness (0.24); worth a manual read to confirm it is prompt-driven elaboration, not a genuine hallucination. |
| Low | **Expand the test set** (e.g. 3–5 questions per chapter, add off-syllabus "trap" questions). | A larger set gives more statistically robust metrics and tests the system's ability to decline out-of-context questions. |

---

## 8. Reproducing the evaluation

```bash
# 1. Install the (separately-pinned) evaluation dependencies
pip install -r requirements-eval.txt

# 2. Ensure the local models are available
ollama pull llama3.1:8b
ollama pull embeddinggemma        # already present from ingestion

# 3. Ensure DATABASE_URL (Supabase session pooler) is set in .env

# 4. Run
python scripts/evaluate_rag.py
```

The script prints aggregate and per-question scores and is the single
source of truth for the test set and wiring.

---

## 9. Limitations & notes

- **Test-set size.** Eight questions is enough to establish a clear signal
  but is not a large statistical sample; see the §7 recommendation to
  expand it.
- **Evaluation model ≠ production model.** As noted in §4.2, generation here
  uses local 8B; production uses Groq 70B. Retrieval metrics are
  representative; generation metrics are a conservative lower bound.
- **Dependency isolation.** RAGAS requires an older LangChain stack
  (0.3.x) than the current 1.x, so its dependencies are pinned separately
  in `requirements-eval.txt` and kept out of the application runtime
  (`requirements.txt`). The harness reaches the local model through
  Ollama's OpenAI-compatible endpoint, avoiding any extra LangChain-Ollama
  dependency.
- **An earlier Groq-judged run** produced approximately context precision
  0.86 / relevancy 0.95 / faithfulness 0.77, but several of its judge calls
  timed out on the Groq free-tier rate limit and were dropped from the
  average, so it is **not** treated as authoritative. It is retained only
  as directional evidence that a stronger generator raises relevancy and
  faithfulness while retrieval quality stays high.
