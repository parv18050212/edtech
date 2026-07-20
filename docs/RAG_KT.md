# RAG Models & Architecture — Knowledge Transfer (KT) Document

**Project:** EdTech RAG Learning Platform (NCERT/Maharashtra-board textbook tutor)
**Last updated:** 2026-07-20
**Scope:** Everything about the models, vector dimensions, retrieval, and generation that power the Retrieval-Augmented Generation (RAG) stack — so a new engineer can pick it up without reading every source file.

---

## 1. High-level picture

The platform ingests school textbooks (EVS Class 4, Science Class 6 — 25 + 19 chapters), chunks and embeds them into a `pgvector` store, and answers student questions by retrieving the most relevant chunks and feeding them to an LLM. A LangGraph orchestrator classifies each request and fans out to the right generator (explanation, step-by-step, quiz, or practice).

```
PDF textbook
  → extract → clean → structure → chunk        (src/ingestion/)
  → embed (embeddinggemma, 768-dim)            (src/ingestion/embed.py)
  → load into Supabase Postgres + pgvector     (chunks table, HNSW index)

Student question
  → embed query (same model)                   (embed_query)
  → cosine similarity search, top-5            (src/api/retrieval.py)
  → LangGraph orchestrator                     (src/api/orchestrator.py)
     ├─ detect_intent  (gpt-oss-20b)
     ├─ retrieve       (pgvector)
     └─ generate       (llama-3.3-70b / gpt-oss-*)
```

---

## 2. Embedding model (the "R" in RAG)

| Property | Value |
|---|---|
| **Model** | `embeddinggemma` |
| **Served via** | Local **Ollama** server at `http://localhost:11434` (endpoint `/api/embed`) |
| **Ollama version** | 0.32.1 (model requires ≥ 0.11.10) |
| **Model size** | 621 MB |
| **Output dimensionality** | **768** (confirmed empirically via a live call, not assumed) |
| **Matryoshka truncation** | Supported by the model (512 / 256 / 128) but **not used** — we keep full 768 |
| **Similarity metric** | **Cosine distance** (model's documented recommendation) |
| **Inference** | CPU-only, local; batching done **per chapter**, not per chunk |

**Code:** `src/ingestion/embed.py`

### Prompt formats (important — asymmetric encoding)

`embeddinggemma` expects different prompt templates for documents vs. queries. Getting this right materially affects retrieval quality:

- **Document (indexing) format:** `title: {title} | text: {content}`
  - `title` = the chunk's `topic` if present, else `chapter_name`, else `"none"`.
  - Built by `build_document_prompt()`.
- **Query (search) format:** `task: search result | query: {content}`
  - Built by `build_query_prompt()`, used at retrieval time via `embed_query()`.

### API functions

- `embed_texts(texts, titles)` → list of 768-float vectors (indexing).
- `embed_query(text)` → single 768-float vector (query time).
- Both call `embed_prompts()`, which POSTs to Ollama's `/api/embed` (accepts a string or array; returns one embedding per input, order-preserved). Timeout: 120s.

---

## 3. Vector store

| Property | Value |
|---|---|
| **Database** | Supabase Postgres 17.6.1 (project `aedsgktxvmddarqurbhi`, region `ap-southeast-1`) |
| **Extension** | `pgvector` **0.8.2** |
| **Table** | `chunks` (single flat, denormalized table) |
| **Embedding column** | `embedding vector(768) not null` |
| **Index** | `hnsw (embedding vector_cosine_ops)` — built upfront (write-once bulk load) |
| **Connection** | `psycopg2` via `DATABASE_URL` (from `.env`, never hardcoded) |

### `chunks` table schema

```sql
create extension if not exists vector;

create table chunks (
  id bigint generated always as identity primary key,
  chunk_id text not null unique,
  board text not null,
  class text not null,
  subject text not null,
  book_title text not null,
  chapter_number int not null,
  chapter_name text not null,
  topic text,
  chunk_type text not null,
  page_start int not null,
  page_end int not null,
  chunk_text text not null,
  embedding vector(768) not null
);

create index on chunks using hnsw (embedding vector_cosine_ops);
```

**Why denormalized?** No LMS/user layer yet justifies a normalized Books/Chapters/Topics schema. The flat table mirrors the `ChunkRecord` dataclass (`src/ingestion/schema.py`) plus the embedding column, and can be normalized later without touching these core columns.

**Note on `class`:** `class` is reserved in Python but not SQL, so the DB column is literally `class`. The `class_` → `class` translation happens in `ChunkRecord.to_dict()` before load.

---

## 4. Chunking strategy

**Code:** `src/ingestion/chunk.py`

- **Target chunk size:** ~**250 tokens** (`split_paragraph(target_tokens=250)`).
- **Token estimate:** heuristic — `words / 0.75`, rounded (no real tokenizer; `estimate_tokens()`).
- **Splitting:** sentence-aware. Paragraphs split on sentence boundaries (`(?<=[.!?])\s+`); sentences are accumulated until the next one would exceed the target, then a new chunk starts. Non-paragraph blocks (tables, etc.) are kept whole.
- **No overlap** between chunks (sentence packing only).
- **Cleanup** (`finalize_chunk_text`): strips `[[PAGE:n]]` artifacts, removes `**` markdown, collapses whitespace.
- **Chunk IDs:** `MSB_{subjectCode}{class}_CH{chapter:02d}_TOP{topic:02d}_{seq:03d}` (e.g. `MSB_SCI6_CH03_TOP01_004`).
- **Volume:** ~1,500+ chunks across both books (44 chapters).

---

## 5. Retrieval

**Code:** `src/api/retrieval.py`

- **Metric:** cosine — `1 - (embedding <=> query)` returned as `similarity`.
- **Ordering:** `order by embedding <=> query::vector` (ascending distance = most similar first).
- **`top_k`:** default **5** (see `RETRIEVAL_TOP_K = 5` in the orchestrator).
- **Hard filters (always applied):** `subject`, `class`, `chapter_number`. Retrieval is always scoped to one chapter — students ask within a chapter context, which keeps recall tight and avoids cross-chapter bleed.
- Query is embedded with `embed_query()` (query prompt format), then the literal vector is passed to the SQL as `[v1,v2,...]`.

---

## 6. Generation models (the "G" in RAG)

All generation runs on **Groq**. Two families are used, chosen deliberately:

| Model | ID | Used for | Why |
|---|---|---|---|
| **Llama 3.3 70B** | `llama-3.3-70b-versatile` | Free-text generation: explanations, step-by-step solutions | Strongest general prose generator |
| **gpt-oss 20B** | `openai/gpt-oss-20b` | Intent classification, practice questions, follow-up questions | Small/fast **and** supports Groq's strict `json_schema` response format (llama models 400 on it) |
| **gpt-oss 120B** | `openai/gpt-oss-120b` | Quiz generation | Larger gpt-oss for higher-quality structured quizzes; strict JSON schema |

**Key constraint driving model choice:** Groq's strict `json_schema` structured-output mode is only reliable on **gpt-oss** models — Llama returns HTTP 400 on it. So anything requiring guaranteed-valid JSON (intent label, question arrays, quiz objects) uses gpt-oss; free prose uses Llama 70B.

- **Temperature:** `0` everywhere (deterministic).
- **LLM plumbing:**
  - `src/api/groq_client.py` — raw `call_groq(prompt, model)` HTTP call with 429 retry (exponential backoff, honors `Retry-After`, max 5 retries). Used for Llama free-text calls.
  - `langchain_groq.ChatGroq` — used where structured output / `json_schema` binding is needed (agents, quiz agent).

---

## 7. Orchestrator (LangGraph)

**Code:** `src/api/orchestrator.py` — a 7-node `StateGraph`.

**Flow:** `START → detect_intent → retrieve → [conditional fan-out] → gen_follow_up → END`

1. **detect_intent** — classifies the question into one of 4 intents using `gpt-oss-20b` + strict schema (`src/api/agents.py`):
   - `explanation` (default when unsure)
   - `step_by_step` (solve/calculate/numericals)
   - `quiz` (structured, gradable)
   - `practice` (open-ended questions to self-try)
2. **retrieve** — embeds the query, pulls top-5 chunks, joins their `chunk_text` into `context`, records `source_chunk_ids`.
3. **Conditional edge** routes on `intent` to exactly one generator node:
   - `gen_explanation` → Llama 70B, grounded `EXPLANATION_PROMPT_TEMPLATE`
   - `gen_step_by_step` → Llama 70B, numbered-steps prompt
   - `gen_quiz` → gpt-oss-120b quiz agent (5 MCQs, mixed difficulty)
   - `gen_practice` → gpt-oss-20b, 3 open-ended questions
4. **gen_follow_up** — always runs after the generator; gpt-oss-20b produces exactly 3 follow-up questions.

> **LangGraph gotcha (recorded):** node names must not collide with `OrchestratorState` field names, so nodes are prefixed `gen_*` (e.g. node `gen_quiz` vs. state key `quiz`).

**Entry point:** `run_ask(conn, question, subject, class_, chapter_number)` → served by the `/ask` route (`src/api/routes/ask.py`, auth-gated). The simpler `/chapters/{n}/chat` route (`src/api/routes/chat.py`) does explanation-only RAG directly (Llama 70B), bypassing the orchestrator.

---

## 8. Grounding / faithfulness

The explanation prompt (`EXPLANATION_PROMPT_TEMPLATE`, shared by `/chat` and the orchestrator's explanation node) is deliberately **strictly grounded**:

- Answer using **only** the retrieved context.
- Do **not** add facts, figures, or examples not in the context.
- Say so plainly if context is insufficient (no guessing).
- Examples / misconception notes are **conditional** — included only if supported by context.

**Why it's written this way:** the earlier version mandated pedagogical elaborations (worked examples, common mistakes) beyond the retrieved chunks, which caused the model to hallucinate. Rewriting it to forbid outside facts and make elaborations conditional raised mean faithfulness **+21% (0.67 → 0.81)**. See `docs/RAG_Evaluation_Report.md` / `docs/rag-evaluation-results.md` and `scripts/compare_faithfulness.py`.

---

## 9. Quick reference card

| Question | Answer |
|---|---|
| Embedding model? | `embeddinggemma` via local Ollama |
| Embedding dimensions? | **768** (no Matryoshka truncation) |
| Similarity metric? | Cosine (`vector_cosine_ops`, HNSW index) |
| Vector DB? | Supabase Postgres + pgvector 0.8.2 |
| Chunk size? | ~250 tokens, sentence-packed, no overlap |
| Retrieval top_k? | 5, filtered by subject + class + chapter |
| Explanation / step-by-step LLM? | `llama-3.3-70b-versatile` (Groq) |
| Intent / practice / follow-up LLM? | `openai/gpt-oss-20b` (Groq, strict JSON) |
| Quiz LLM? | `openai/gpt-oss-120b` (Groq, strict JSON) |
| Orchestration? | LangGraph 7-node StateGraph |
| Temperature? | 0 everywhere |

---

## 10. Key files

| Concern | File |
|---|---|
| Embedding + prompt formats | `src/ingestion/embed.py` |
| Chunking | `src/ingestion/chunk.py` |
| Chunk record shape | `src/ingestion/schema.py` |
| Similarity search SQL | `src/api/retrieval.py` |
| LangGraph orchestrator | `src/api/orchestrator.py` |
| Intent + generators (agents) | `src/api/agents.py` |
| Quiz agent | `src/api/quiz_agent.py` |
| Groq HTTP client (retry) | `src/api/groq_client.py` |
| Explanation prompt + chat route | `src/api/routes/chat.py` |
| `/ask` route | `src/api/routes/ask.py` |
| DB connection | `src/api/db.py` |
| Full design spec | `docs/superpowers/specs/2026-07-18-full-rollout-embedding-pgvector-design.md` |
| Eval results | `docs/RAG_Evaluation_Report.md`, `docs/rag-evaluation-results.md` |
