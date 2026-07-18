# RAG Chat + Quiz Generation API — Design

## Context

The data layer is done: 1,560 chunks across both textbooks, embedded and
loaded into Supabase's `chunks` table (pgvector, HNSW cosine index). This
slice builds the first product-facing capability on top of it — a FastAPI
backend exposing chapter-scoped RAG chat and quiz generation — matching
the "PDF ingestion and chapter-wise RAG chat" + "Quiz generation" items
from Phase 1 of the original architecture study's roadmap (Section 21).

Explicitly scoped to **backend only** this round. No frontend. No custom
auth code — Supabase Auth handles sign-up/login directly; this backend
only verifies the JWTs it's handed. No Teacher/Parent/Admin roles or
portals (Phase 2/3 in the original doc).

## Verified facts

- **Groq chat completions**: `POST https://api.groq.com/openai/v1/chat/completions`,
  `Authorization: Bearer $GROQ_API_KEY`, OpenAI-compatible request body.
- **Chat/explanation model**: `llama-3.3-70b-versatile` (131K context,
  $0.59/$0.79 per 1M tokens).
- **Quiz model**: `openai/gpt-oss-120b` — one of only two Groq models
  supporting `response_format: {"type": "json_schema", "json_schema": {...,
  "strict": true}}`, which guarantees the output matches our quiz schema
  exactly. `llama-3.3-70b-versatile` does **not** support strict schema
  mode, so quiz generation uses a different model than chat, deliberately.
- **Query embedding prompt format** (from `embeddinggemma`'s Hugging Face
  card, distinct from the document format used during ingestion):
  `"task: search result | query: {question}"`.
- **Supabase JWT verification**: confirmed empirically by fetching this
  project's own JWKS endpoint
  (`https://aedsgktxvmddarqurbhi.supabase.co/auth/v1/.well-known/jwks.json`)
  — it returns an ES256 (asymmetric) key, not the legacy HS256 shared
  secret. Verification needs no secret credential at all: fetch the public
  key from the JWKS endpoint and verify with a standard JWT library
  (`PyJWT` + `PyJWKClient`), checking `iss=https://aedsgktxvmddarqurbhi.supabase.co/auth/v1`
  and `aud=authenticated`. The verified token's `sub` claim is the user's
  UUID, matching `auth.users.id`.
- **pgvector query pattern**: cosine distance operator `<=>` (0 =
  identical), matching the `vector_cosine_ops` HNSW index already built on
  `chunks.embedding`. Similarity score is `1 - (embedding <=> query_vec)`.

## Architecture

```
Client (has a Supabase-issued JWT from signing in via Supabase Auth)
  |
  v
FastAPI backend
  - Verifies JWT via Supabase JWKS (no shared secret needed)
  - Connects directly to Postgres via the same session-pooler
    DATABASE_URL already in use for the ingestion pipeline (psycopg2/
    asyncpg) -- NOT through PostgREST/RLS, since this is a trusted
    first-party backend with its own direct DB connection
  - RAG chat: embed question (query format) -> pgvector search,
    filtered by subject/class/chapter_number -> Groq (llama-3.3-70b) ->
    grounded answer -> logged to chat_history
  - Quiz generation: chapter's chunks -> Groq (gpt-oss-120b, strict
    schema) -> validated quiz JSON, no retry logic needed
  - Quiz submission: score against quiz_json -> recorded to quiz_attempts
```

**RLS and the direct-connection tradeoff**: because this backend connects
directly to Postgres as the `postgres` role (table owner), Postgres RLS
does **not** apply to it — RLS only restricts roles it's written for
(`anon`/`authenticated` via PostgREST), and a direct owner-role connection
bypasses it regardless of policy content. RLS policies are still created
on the new tables (defense in depth, and required if a future frontend
ever queries these tables directly via `supabase-js`/PostgREST instead of
through this backend). But for *this* backend, the real authorization
boundary is application code: every query against `chat_history` or
`quiz_attempts` includes `where user_id = :verified_user_id` using the
`sub` claim from the verified JWT, not RLS.

## Data model additions

```sql
create table chat_history (
  id bigint generated always as identity primary key,
  user_id uuid not null,
  chapter_number int not null,
  subject text not null,
  class text not null,
  question text not null,
  answer text not null,
  created_at timestamptz not null default now()
);
alter table chat_history enable row level security;
create policy "select own chat history" on chat_history
  for select to authenticated
  using ((select auth.uid()) = user_id);
create policy "insert own chat history" on chat_history
  for insert to authenticated
  with check ((select auth.uid()) = user_id);

create table quiz_attempts (
  id bigint generated always as identity primary key,
  user_id uuid not null,
  chapter_number int not null,
  subject text not null,
  class text not null,
  quiz_json jsonb not null,
  student_answers jsonb,
  score int,
  attempted_at timestamptz not null default now()
);
alter table quiz_attempts enable row level security;
create policy "select own quiz attempts" on quiz_attempts
  for select to authenticated
  using ((select auth.uid()) = user_id);
create policy "insert own quiz attempts" on quiz_attempts
  for insert to authenticated
  with check ((select auth.uid()) = user_id);
create policy "update own quiz attempts" on quiz_attempts
  for update to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);

-- chunks already exists; add public read access since content isn't
-- sensitive and needs no ownership check
alter table chunks enable row level security;
create policy "public read access" on chunks
  for select to authenticated, anon
  using (true);
```

No foreign key from `chat_history`/`quiz_attempts` to `auth.users` is
declared (cross-schema FKs to `auth.users` are a known Supabase footgun
around cascading behavior); `user_id` is validated by the JWT check at the
API layer instead, matching the "trusted backend" model already described.

## RAG chat flow

`POST /chapters/{chapter_number}/chat`, with `subject`/`class` in the
request body (see API endpoints table below for the exact shape):

```python
def rag_chat(question: str, subject: str, class_: str, chapter_number: int) -> str:
    query_embedding = embed_texts(
        [question], titles=[None], model="embeddinggemma"
    )[0]  # reuses src/ingestion/embed.py, but with the query prompt format
    # (build_document_prompt's "title: ... | text: ..." format is wrong here --
    # a new build_query_prompt() is needed, see Task breakdown)

    chunks = search_chunks(query_embedding, subject, class_, chapter_number, top_k=5)

    context = "\n\n".join(c["chunk_text"] for c in chunks)
    prompt = (
        "You are a helpful tutor. Answer using ONLY the context below.\n"
        f"Context: {context}\n"
        f"Student question: {question}\n"
        "Provide: a simple explanation, one worked example, and one common mistake students make."
    )
    answer = call_groq(prompt, model="llama-3.3-70b-versatile")
    return answer
```

This reuses the exact explanation prompt template from the original
architecture study's Appendix A.1.

## Quiz generation flow

`POST /chapters/{chapter_number}/quiz`: fetch all chunks for the given
subject/class/chapter (no similarity search needed — the whole chapter's
content is the source), build the quiz prompt (Appendix A.2 of the
original study), call `gpt-oss-120b` with `response_format` set to a
`json_schema` matching:

```json
{
  "quiz": [
    {
      "question": "string",
      "options": ["string", "string", "string", "string"],
      "correct_answer": "string",
      "difficulty": "easy | medium | hard",
      "explanation": "string"
    }
  ]
}
```

Because this model supports strict schema mode, the response is
guaranteed to match this shape — no Pydantic-retry loop is needed (a
deliberate simplification vs. the original doc, made possible by choosing
a model that supports strict structured output).

## API endpoints

`subject` and `class` disambiguate which book `chapter_number` refers to
(chunk metadata has no single canonical book ID beyond that combination),
so both are required on every chapter-scoped request — as JSON body
fields on POST endpoints, not query params, since the body already
carries other required fields (the question text, quiz answers):

| Endpoint | Method | Auth | Body | Purpose |
|---|---|---|---|---|
| `/chapters/{chapter_number}/chat` | POST | required | `{subject, class, question}` | RAG chat, logs to `chat_history` |
| `/chapters/{chapter_number}/quiz` | POST | required | `{subject, class}` | Generate a quiz, does not persist until submitted |
| `/quiz/submit` | POST | required | `{subject, class, chapter_number, quiz_json, student_answers}` | Score a completed quiz, persist to `quiz_attempts` |
| `/progress/{user_id}` | GET | required, `user_id` must match JWT `sub` | — | Chat + quiz history summary for one user |

Auth is enforced via a FastAPI dependency that verifies the JWT (JWKS,
`iss`/`aud` checks) and raises `401` on failure; the same dependency
supplies `user_id` (the `sub` claim) to route handlers.

## Project structure

```
src/api/
  __init__.py
  main.py              # FastAPI app, route registration
  auth.py               # JWT verification dependency (JWKS-based)
  groq_client.py          # call_groq(), call_groq_json_schema()
  retrieval.py              # search_chunks() -- pgvector query
  routes/
    chat.py
    quiz.py
    progress.py
  schemas.py                 # Pydantic request/response models
supabase/migrations/
  0002_chat_quiz_tables_and_chunks_rls.sql
```

`src/ingestion/embed.py` gains one new function, `build_query_prompt()`,
alongside the existing `build_document_prompt()` — both are thin wrappers
around the same `embed_texts()` HTTP call, differing only in prompt
prefix, so no duplication of the Ollama-calling logic.

## Testing strategy

Consistent with this project's established preference for testing
against real systems over mocks:

- `search_chunks()`: unit test against the real Supabase `chunks` table
  (already populated with 1,560 real rows) — verify a known question
  about a known chapter returns chunks from the correct chapter, not
  chunks from an unrelated one (the retrieval-accuracy check the original
  doc's testing section calls for).
- `auth.py`'s JWT verification: unit tests using a locally-signed test
  token (via a throwaway ES256 keypair, not real Supabase credentials) to
  verify the dependency correctly accepts valid tokens and rejects
  expired/wrong-issuer/wrong-audience/badly-signed ones.
- `call_groq()`/`call_groq_json_schema()`: integration tests against the
  real Groq API (requires `GROQ_API_KEY`), verifying a real chat
  completion and a real strict-schema quiz generation call.
- End-to-end: run the FastAPI app locally, hit `/chapters/1/chat` and
  `/chapters/1/quiz` for a real EVS or Science chapter, manually inspect
  the grounded answer and generated quiz for quality (mirrors the pilot
  review process from the ingestion phase).

## Out of scope for this slice

- Frontend (Next.js, per the original doc) — separate next round.
- Teacher/Parent/Admin roles, portals, assignments.
- Learning modes beyond Chat/Explain and Quiz (Practice, Test, Revision,
  Flashcard, Doubt modes from Section 11 of the original study).
- Adaptive learning path / recommendation logic.
- Rate limiting, caching (Redis), or production-scale concerns — this is
  still POC scope.
