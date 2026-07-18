# RAG Chat + Quiz Generation API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend exposing chapter-scoped RAG chat and quiz generation on top of the existing pgvector `chunks` table, per `docs/superpowers/specs/2026-07-18-rag-chat-quiz-api-design.md`.

**Architecture:** JWT auth (JWKS-verified against Supabase, no shared secret) gates every route. A per-request Postgres connection (same session-pooler `DATABASE_URL` as the ingestion pipeline) runs pgvector similarity search for chat and full-chapter fetch for quizzes. Groq provides generation: `llama-3.3-70b-versatile` for chat, `openai/gpt-oss-120b` with strict JSON schema for quizzes. `chat_history` and `quiz_attempts` are written directly by the FastAPI app (bypassing RLS via the direct connection, with `user_id` ownership enforced in application code); RLS policies exist on all three tables as defense in depth for any future direct-from-frontend access path.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, PyJWT[crypto], httpx (test client), psycopg2-binary (existing), requests (existing), pytest.

## Global Constraints

- Supabase project id: `aedsgktxvmddarqurbhi`. JWKS URL:
  `https://aedsgktxvmddarqurbhi.supabase.co/auth/v1/.well-known/jwks.json`
  (confirmed live: ES256, one key). Issuer:
  `https://aedsgktxvmddarqurbhi.supabase.co/auth/v1`. Audience: `authenticated`.
- Groq chat completions: `POST https://api.groq.com/openai/v1/chat/completions`,
  `Authorization: Bearer $GROQ_API_KEY`, OpenAI-compatible body.
- Chat model: `llama-3.3-70b-versatile`. Quiz model: `openai/gpt-oss-120b`
  (one of only two Groq models supporting strict `json_schema` structured
  output; `llama-3.3-70b-versatile` does not support it).
- Query embedding prompt format (distinct from the document format used
  during ingestion): `"task: search result | query: {content}"`.
- New pinned dependencies (versions confirmed installed in this
  environment): `fastapi==0.139.2`, `uvicorn==0.51.0`, `pyjwt[crypto]==2.13.0`
  (pulls in `cryptography==49.0.0`), `httpx==0.28.1`. `pydantic==2.13.4` is
  pulled in transitively by FastAPI.
- `GROQ_API_KEY` is read from `.env` via `python-dotenv`, same handling as
  `DATABASE_URL`: never printed, logged, or committed; only
  `.env.example` documents the variable name.
- `class` is a Python keyword; Pydantic request models use
  `Field(alias="class")` with `populate_by_name=True` so the wire format
  stays `"class"` while the Python attribute is `class_`, consistent with
  `ChunkRecord`'s existing `class_`/`"class"` convention.
- `tests/api/__init__.py` and `src/api/__init__.py` (and
  `src/api/routes/__init__.py`) must exist from Task 1 onward — omitting
  `tests/api/__init__.py` reproduces the exact `ingestion`/`api` package
  name collision already diagnosed and fixed once in this project (see
  the ingestion plan's Task 1).

---

## File Structure

```
edtech/
  src/
    ingestion/
      embed.py            # MODIFY: add build_query_prompt, embed_prompts, embed_query
    api/
      __init__.py
      db.py                 # CREATE: get_connection() from DATABASE_URL
      auth.py                 # CREATE: JWT verification (JWKS-based)
      groq_client.py            # CREATE: call_groq, call_groq_json_schema
      retrieval.py                # CREATE: search_chunks
      schemas.py                    # CREATE: Pydantic request/response models
      main.py                         # CREATE: FastAPI app, route registration
      routes/
        __init__.py
        chat.py
        quiz.py
        progress.py
  tests/
    ingestion/
      test_embed.py        # MODIFY: add tests for the new embed.py functions
    api/
      __init__.py
      test_auth.py
      test_retrieval.py
      test_groq_client.py
      test_schemas.py
      test_chat_route.py
      test_quiz_route.py
      test_progress_route.py
  supabase/migrations/
    0002_chat_quiz_tables_and_chunks_rls.sql
  requirements.txt          # MODIFY: add fastapi, uvicorn, pyjwt[crypto], httpx
  .env.example                # MODIFY: add GROQ_API_KEY
```

---

### Task 1: Dependencies and environment scaffolding

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Create: `src/api/__init__.py`
- Create: `src/api/routes/__init__.py`
- Create: `tests/api/__init__.py`

**Interfaces:**
- Produces: `fastapi`, `uvicorn`, `pyjwt[crypto]`, `httpx` installed and importable; `src/api` and `tests/api` established as packages (avoiding the name-collision bug from the ingestion plan).

- [ ] **Step 1: Add dependencies to `requirements.txt`**

```
pymupdf==1.28.0
pytest==8.3.3
requests==2.34.2
psycopg2-binary==2.9.12
python-dotenv==1.2.2
fastapi==0.139.2
uvicorn==0.51.0
pyjwt[crypto]==2.13.0
httpx==0.28.1
```

- [ ] **Step 2: Install and verify**

```bash
"D:/Coding/edtech/.venv/Scripts/pip" install -r requirements.txt
"D:/Coding/edtech/.venv/Scripts/python" -c "import fastapi, uvicorn, jwt, httpx; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Add `GROQ_API_KEY` to `.env.example`**

```
# Use the Session Pooler connection string, not the direct connection --
# the direct host (db.<ref>.supabase.co) resolves IPv6-only, which fails
# on networks without IPv6 connectivity. Find this under Project Settings
# -> Database -> Connection string -> "Session pooler".
DATABASE_URL=postgresql://postgres.aedsgktxvmddarqurbhi:[YOUR-PASSWORD]@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres

# From https://console.groq.com/keys
GROQ_API_KEY=[YOUR-GROQ-API-KEY]
```

- [ ] **Step 4: Create package markers**

Create `src/api/__init__.py`, `src/api/routes/__init__.py`, and `tests/api/__init__.py` (all empty files).

- [ ] **Step 5: Verify pytest still collects cleanly**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" --collect-only 2>&1 | tail -5
```

Expected: collects the existing 68 tests with no errors (no new test files yet).

- [ ] **Step 6: Ask the user to add their real `GROQ_API_KEY` to `.env`**

Same handling as `DATABASE_URL`: the user adds it to their local `.env` directly (get a key from `https://console.groq.com/keys`); it is never typed into the conversation. Task 5 (the Groq client) requires this to be set before its tests can pass.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .env.example src/api/__init__.py src/api/routes/__init__.py tests/api/__init__.py
git commit -m "chore: add FastAPI/JWT dependencies and package scaffolding"
```

---

### Task 2: Database migration — chat_history, quiz_attempts, chunks RLS

**Files:**
- Create: `supabase/migrations/0002_chat_quiz_tables_and_chunks_rls.sql`

**Interfaces:**
- Produces: `chat_history` and `quiz_attempts` tables (both RLS-enabled, owner-scoped policies), and a public-read RLS policy on the existing `chunks` table. Used by `retrieval.py` (Task 4) and the chat/quiz routes (Tasks 8-9).

- [ ] **Step 1: Write the migration file**

```sql
-- supabase/migrations/0002_chat_quiz_tables_and_chunks_rls.sql

alter table chunks enable row level security;
create policy "public read access" on chunks
  for select to authenticated, anon
  using (true);

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
```

- [ ] **Step 2: Apply the migration**

Use the Supabase MCP tool `apply_migration` with `project_id="aedsgktxvmddarqurbhi"`, `name="chat_quiz_tables_and_chunks_rls"`, and the SQL contents above.

- [ ] **Step 3: Verify**

Use the Supabase MCP tool `list_tables` with `project_id="aedsgktxvmddarqurbhi"`, `schemas=["public"]`, `verbose=true`. Expected: `chat_history` and `quiz_attempts` both listed with `rls_enabled: true` and the columns above.

Use `execute_sql` with:

```sql
select tablename, policyname, cmd from pg_policies where tablename in ('chunks', 'chat_history', 'quiz_attempts') order by tablename, cmd;
```

Expected: 6 policies total (1 select on `chunks`, 2 on `chat_history` [select+insert], 3 on `quiz_attempts` [select+insert+update]).

- [ ] **Step 4: Commit**

```bash
git add supabase/migrations/0002_chat_quiz_tables_and_chunks_rls.sql
git commit -m "feat: add chat_history, quiz_attempts tables and chunks RLS policy"
```

---

### Task 3: `embed.py` — query embedding support

**Files:**
- Modify: `src/ingestion/embed.py`
- Modify: `tests/ingestion/test_embed.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_query_prompt(text: str) -> str`, `embed_prompts(prompts: list[str], model=..., host=...) -> list[list[float]]` (the new low-level HTTP call, factored out of the existing `embed_texts`), `embed_query(text: str, model=..., host=...) -> list[float]`. `embed_texts`'s existing signature and behavior are unchanged (it now calls `embed_prompts` internally) — `run_full_pipeline.py`'s existing call site needs no changes. Used by `retrieval.py` (Task 4).

- [ ] **Step 1: Write the failing tests**

Add to `tests/ingestion/test_embed.py`:

```python
from ingestion.embed import build_query_prompt, embed_prompts, embed_query


def test_build_query_prompt():
    assert (
        build_query_prompt("What is a star?")
        == "task: search result | query: What is a star?"
    )


def test_embed_prompts_returns_768_dim_vectors():
    embeddings = embed_prompts(["title: none | text: The sun is a star."])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768


def test_embed_query_returns_single_768_dim_vector():
    embedding = embed_query("What is a star?")
    assert len(embedding) == 768


def test_embed_texts_still_works_after_refactor():
    # Regression guard: embed_texts's existing public behavior (used by
    # scripts/run_full_pipeline.py) must be unchanged by factoring out
    # embed_prompts.
    embeddings = embed_texts(["The sun is a star."], titles=["Stars"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_embed.py -v
```

Expected: FAIL with `ImportError: cannot import name 'build_query_prompt'` (the last test, using the already-existing `embed_texts`, should still pass on its own — but collection fails first due to the import error, so all report as errors until the import succeeds).

- [ ] **Step 3: Refactor the implementation**

Replace the contents of `src/ingestion/embed.py`:

```python
from typing import Optional

import requests

DEFAULT_MODEL = "embeddinggemma"
DEFAULT_HOST = "http://localhost:11434"


def build_document_prompt(text: str, title: Optional[str]) -> str:
    title_value = title if title else "none"
    return f"title: {title_value} | text: {text}"


def build_query_prompt(text: str) -> str:
    return f"task: search result | query: {text}"


def embed_prompts(
    prompts: list[str],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
) -> list[list[float]]:
    response = requests.post(
        f"{host}/api/embed",
        json={"model": model, "input": prompts},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def embed_texts(
    texts: list[str],
    titles: list[Optional[str]],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
) -> list[list[float]]:
    prompts = [
        build_document_prompt(text, title) for text, title in zip(texts, titles)
    ]
    return embed_prompts(prompts, model=model, host=host)


def embed_query(
    text: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST
) -> list[float]:
    return embed_prompts([build_query_prompt(text)], model=model, host=host)[0]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_embed.py -v
```

Expected: 9 passed (5 existing + 4 new).

- [ ] **Step 5: Run the full suite to confirm no regressions**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" -v 2>&1 | tail -5
```

Expected: all 68 previous tests plus the 4 new ones pass (72 total; the earlier "9 passed" in Step 4 is scoped to just `test_embed.py`).

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/embed.py tests/ingestion/test_embed.py
git commit -m "refactor: factor embed_prompts out of embed_texts, add query embedding"
```

---

### Task 4: `retrieval.py` — pgvector similarity search

**Files:**
- Create: `src/api/retrieval.py`
- Create: `tests/api/test_retrieval.py`

**Interfaces:**
- Consumes: `embed_query` (Task 3), a live `psycopg2` connection (caller-supplied, not created internally — matches the FastAPI dependency-injection pattern used in Task 8).
- Produces: `search_chunks(conn, query_embedding: list[float], subject: str, class_: str, chapter_number: int, top_k: int = 5) -> list[dict]`. Used by `routes/chat.py` (Task 8).

**Note:** tests run against the real Supabase `chunks` table (1,560 real rows already loaded), consistent with this project's established preference for testing against real systems.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_retrieval.py
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import psycopg2
from dotenv import load_dotenv

from api.retrieval import search_chunks
from ingestion.embed import embed_query

load_dotenv()


def _connect():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def test_search_chunks_returns_only_the_requested_chapter():
    conn = _connect()
    try:
        query_embedding = embed_query("What is a star?")
        results = search_chunks(
            conn,
            query_embedding,
            subject="Environmental Studies",
            class_="5",
            chapter_number=1,
            top_k=5,
        )
        assert len(results) == 5
        for r in results:
            assert r["chunk_id"].startswith("MSB_EVS5_CH01_")
    finally:
        conn.close()


def test_search_chunks_excludes_other_chapters():
    conn = _connect()
    try:
        query_embedding = embed_query("What is a star?")
        results = search_chunks(
            conn,
            query_embedding,
            subject="Environmental Studies",
            class_="5",
            chapter_number=2,  # "Motions of the Earth", not the Stars chapter
            top_k=5,
        )
        for r in results:
            assert r["chunk_id"].startswith("MSB_EVS5_CH02_")
    finally:
        conn.close()


def test_search_chunks_orders_by_similarity_descending():
    conn = _connect()
    try:
        query_embedding = embed_query("What is a star?")
        results = search_chunks(
            conn,
            query_embedding,
            subject="Environmental Studies",
            class_="5",
            chapter_number=1,
            top_k=5,
        )
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)
    finally:
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_retrieval.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.retrieval'`.

- [ ] **Step 3: Write the implementation**

```python
# src/api/retrieval.py
SEARCH_SQL = """
select chunk_id, chapter_name, topic, chunk_type, chunk_text, page_start, page_end,
       1 - (embedding <=> %(embedding)s::vector) as similarity
from chunks
where subject = %(subject)s and class = %(class)s and chapter_number = %(chapter_number)s
order by embedding <=> %(embedding)s::vector
limit %(top_k)s
"""


def search_chunks(
    conn,
    query_embedding: list[float],
    subject: str,
    class_: str,
    chapter_number: int,
    top_k: int = 5,
) -> list[dict]:
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
    with conn.cursor() as cur:
        cur.execute(
            SEARCH_SQL,
            {
                "embedding": embedding_literal,
                "subject": subject,
                "class": class_,
                "chapter_number": chapter_number,
                "top_k": top_k,
            },
        )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_retrieval.py -v
```

Expected: 3 passed. This makes one real Ollama call (for the query embedding) and three real Supabase queries.

- [ ] **Step 5: Commit**

```bash
git add src/api/retrieval.py tests/api/test_retrieval.py
git commit -m "feat: add pgvector similarity search scoped by subject/class/chapter"
```

---

### Task 5: `groq_client.py` — Groq chat and structured-output calls

**Files:**
- Create: `src/api/groq_client.py`
- Create: `tests/api/test_groq_client.py`

**Interfaces:**
- Consumes: `GROQ_API_KEY` from the environment (via `.env`, Task 1 Step 6).
- Produces: `call_groq(prompt: str, model: str, api_key: Optional[str] = None) -> str` and `call_groq_json_schema(prompt: str, model: str, schema: dict, schema_name: str, api_key: Optional[str] = None) -> dict`. Used by `routes/chat.py` (Task 8) and `routes/quiz.py` (Task 9).

**Note:** requires a real `GROQ_API_KEY` in `.env` (Task 1 Step 6) — these tests make real, billed (at Groq's published per-token rates, fractions of a cent for these tiny test prompts) API calls, consistent with this project's established preference for testing against real systems.

- [ ] **Step 1: Write the failing tests**

```python
# tests/api/test_groq_client.py
from api.groq_client import call_groq, call_groq_json_schema


def test_call_groq_returns_text_response():
    answer = call_groq(
        "Reply with exactly one word: hello",
        model="llama-3.3-70b-versatile",
    )
    assert "hello" in answer.lower()


def test_call_groq_json_schema_returns_matching_shape():
    schema = {
        "type": "object",
        "properties": {
            "greeting": {"type": "string"},
        },
        "required": ["greeting"],
        "additionalProperties": False,
    }
    result = call_groq_json_schema(
        "Return a JSON object with a 'greeting' field containing the word hello.",
        model="openai/gpt-oss-120b",
        schema=schema,
        schema_name="greeting_response",
    )
    assert "greeting" in result
    assert isinstance(result["greeting"], str)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_groq_client.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.groq_client'`.

- [ ] **Step 3: Write the implementation**

```python
# src/api/groq_client.py
import json
import os
from typing import Optional

import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _headers(api_key: Optional[str]) -> dict:
    key = api_key or os.environ["GROQ_API_KEY"]
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def call_groq(prompt: str, model: str, api_key: Optional[str] = None) -> str:
    response = requests.post(
        GROQ_API_URL,
        headers=_headers(api_key),
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_groq_json_schema(
    prompt: str,
    model: str,
    schema: dict,
    schema_name: str,
    api_key: Optional[str] = None,
) -> dict:
    response = requests.post(
        GROQ_API_URL,
        headers=_headers(api_key),
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_groq_client.py -v
```

Expected: 2 passed. If `KeyError: 'GROQ_API_KEY'`, confirm Task 1 Step 6 was completed (real key added to local `.env`).

- [ ] **Step 5: Commit**

```bash
git add src/api/groq_client.py tests/api/test_groq_client.py
git commit -m "feat: add Groq chat and strict-JSON-schema client"
```

---

### Task 6: `auth.py` — JWT verification

**Files:**
- Create: `src/api/auth.py`
- Create: `tests/api/test_auth.py`

**Interfaces:**
- Consumes: nothing from earlier tasks in this plan.
- Produces: `_decode(token: str, key, issuer: str, audience: str, algorithms: list[str]) -> dict` (pure verification logic, unit-tested with a locally-generated keypair — no network), `verify_token(token: str) -> dict` (fetches the real signing key from Supabase's JWKS endpoint, then calls `_decode`), and the FastAPI dependency `get_current_user_id(authorization: str) -> str`. Used by every route in Tasks 8-10.

- [ ] **Step 1: Write the failing tests**

```python
# tests/api/test_auth.py
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from api.auth import _decode

ISSUER = "https://example.supabase.co/auth/v1"
AUDIENCE = "authenticated"


def _make_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    return private_key, private_key.public_key()


def _make_token(private_key, **claim_overrides):
    now = int(time.time())
    claims = {
        "sub": "user-123",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": now + 3600,
    }
    claims.update(claim_overrides)
    return jwt.encode(claims, private_key, algorithm="ES256")


def test_decode_accepts_valid_token():
    private_key, public_key = _make_keypair()
    token = _make_token(private_key)
    payload = _decode(token, public_key, ISSUER, AUDIENCE, ["ES256"])
    assert payload["sub"] == "user-123"


def test_decode_rejects_wrong_issuer():
    private_key, public_key = _make_keypair()
    token = _make_token(private_key, iss="https://evil.example.com/auth/v1")
    with pytest.raises(jwt.InvalidIssuerError):
        _decode(token, public_key, ISSUER, AUDIENCE, ["ES256"])


def test_decode_rejects_wrong_audience():
    private_key, public_key = _make_keypair()
    token = _make_token(private_key, aud="something-else")
    with pytest.raises(jwt.InvalidAudienceError):
        _decode(token, public_key, ISSUER, AUDIENCE, ["ES256"])


def test_decode_rejects_expired_token():
    private_key, public_key = _make_keypair()
    token = _make_token(private_key, exp=int(time.time()) - 10)
    with pytest.raises(jwt.ExpiredSignatureError):
        _decode(token, public_key, ISSUER, AUDIENCE, ["ES256"])


def test_decode_rejects_token_signed_by_a_different_key():
    _, real_public_key = _make_keypair()
    other_private_key, _ = _make_keypair()
    token = _make_token(other_private_key)  # signed by a DIFFERENT key
    with pytest.raises(jwt.InvalidSignatureError):
        _decode(token, real_public_key, ISSUER, AUDIENCE, ["ES256"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_auth.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.auth'`.

- [ ] **Step 3: Write the implementation**

```python
# src/api/auth.py
import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

SUPABASE_PROJECT_REF = "aedsgktxvmddarqurbhi"
ISSUER = f"https://{SUPABASE_PROJECT_REF}.supabase.co/auth/v1"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
AUDIENCE = "authenticated"

_jwk_client = PyJWKClient(JWKS_URL)


def _decode(token: str, key, issuer: str, audience: str, algorithms: list[str]) -> dict:
    return jwt.decode(token, key, algorithms=algorithms, issuer=issuer, audience=audience)


def verify_token(token: str) -> dict:
    signing_key = _jwk_client.get_signing_key_from_jwt(token)
    return _decode(token, signing_key.key, ISSUER, AUDIENCE, ["ES256"])


def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ")
    try:
        payload = verify_token(token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload["sub"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_auth.py -v
```

Expected: 5 passed. These tests use a locally-generated throwaway keypair and never contact the real Supabase JWKS endpoint or require any real credentials.

- [ ] **Step 5: Commit**

```bash
git add src/api/auth.py tests/api/test_auth.py
git commit -m "feat: add JWKS-based JWT verification"
```

---

### Task 7: `schemas.py` — request/response models

**Files:**
- Create: `src/api/schemas.py`
- Create: `tests/api/test_schemas.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ChatRequest`, `ChatResponse`, `QuizRequest`, `QuizQuestion`, `QuizResponse`, `QuizSubmitRequest` Pydantic models. Used by `routes/chat.py` and `routes/quiz.py` (Tasks 8-9).

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_schemas.py
from api.schemas import ChatRequest, QuizRequest


def test_chat_request_accepts_class_as_wire_field_name():
    request = ChatRequest.model_validate(
        {"subject": "Science", "class": "8", "question": "What is a star?"}
    )
    assert request.class_ == "8"
    assert request.subject == "Science"
    assert request.question == "What is a star?"


def test_quiz_request_accepts_class_as_wire_field_name():
    request = QuizRequest.model_validate({"subject": "Science", "class": "8"})
    assert request.class_ == "8"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_schemas.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.schemas'`.

- [ ] **Step 3: Write the implementation**

```python
# src/api/schemas.py
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    subject: str
    class_: str = Field(alias="class")
    question: str


class ChatResponse(BaseModel):
    answer: str
    source_chunk_ids: list[str]


class QuizRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    subject: str
    class_: str = Field(alias="class")


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    difficulty: str
    explanation: str


class QuizResponse(BaseModel):
    quiz: list[QuizQuestion]


class QuizSubmitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    subject: str
    class_: str = Field(alias="class")
    chapter_number: int
    quiz_json: dict
    student_answers: dict
```

- [ ] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_schemas.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/schemas.py tests/api/test_schemas.py
git commit -m "feat: add Pydantic request/response schemas"
```

---

### Task 8: `main.py` and `routes/chat.py` — RAG chat endpoint

**Files:**
- Create: `src/api/db.py`
- Create: `src/api/main.py`
- Create: `src/api/routes/chat.py`
- Create: `tests/api/test_chat_route.py`

**Interfaces:**
- Consumes: `search_chunks` (Task 4), `call_groq` (Task 5), `get_current_user_id` (Task 6), `ChatRequest`/`ChatResponse` (Task 7), `embed_query` (Task 3).
- Produces: `POST /chapters/{chapter_number}/chat`, tested via `fastapi.testclient.TestClient` against the real app (real Supabase data, real Ollama, real Groq — no mocks, consistent with this project's established testing approach).

- [ ] **Step 1: Write `src/api/db.py`**

```python
# src/api/db.py
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def get_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 2: Write the failing test**

```python
# tests/api/test_chat_route.py
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_chat_requires_authorization_header():
    response = client.post(
        "/chapters/1/chat",
        json={"subject": "Environmental Studies", "class": "5", "question": "What is a star?"},
    )
    assert response.status_code in (401, 422)


def test_chat_rejects_invalid_token():
    response = client.post(
        "/chapters/1/chat",
        json={"subject": "Environmental Studies", "class": "5", "question": "What is a star?"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401
```

Note: a full success-path test (valid Supabase-issued JWT, real grounded answer) is deliberately deferred to Task 11's manual end-to-end verification, since generating a real, currently-valid Supabase user JWT requires an actual sign-in flow, not something a unit test should fabricate.

- [ ] **Step 3: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_chat_route.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.main'`.

- [ ] **Step 4: Write `src/api/routes/chat.py`**

```python
# src/api/routes/chat.py
from fastapi import APIRouter, Depends

from api.auth import get_current_user_id
from api.db import get_db
from api.groq_client import call_groq
from api.retrieval import search_chunks
from api.schemas import ChatRequest, ChatResponse
from ingestion.embed import embed_query

router = APIRouter()

EXPLANATION_PROMPT_TEMPLATE = """You are a helpful tutor. Answer using ONLY the context below.
Context: {context}
Student question: {question}
Provide: a simple explanation, one worked example, and one common mistake students make."""


@router.post("/chapters/{chapter_number}/chat", response_model=ChatResponse)
def chat(chapter_number: int, request: ChatRequest, user_id: str = Depends(get_current_user_id)):
    conn = next(get_db())
    try:
        query_embedding = embed_query(request.question)
        chunks = search_chunks(
            conn, query_embedding, request.subject, request.class_, chapter_number
        )
        context = "\n\n".join(c["chunk_text"] for c in chunks)
        prompt = EXPLANATION_PROMPT_TEMPLATE.format(context=context, question=request.question)
        answer = call_groq(prompt, model="llama-3.3-70b-versatile")

        with conn.cursor() as cur:
            cur.execute(
                """
                insert into chat_history (user_id, chapter_number, subject, class, question, answer)
                values (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, chapter_number, request.subject, request.class_, request.question, answer),
            )
        conn.commit()

        return ChatResponse(answer=answer, source_chunk_ids=[c["chunk_id"] for c in chunks])
    finally:
        conn.close()
```

- [ ] **Step 5: Write `src/api/main.py`**

```python
# src/api/main.py
from fastapi import FastAPI

from api.routes import chat

app = FastAPI(title="RAG Learning Platform API")
app.include_router(chat.router)
```

- [ ] **Step 6: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_chat_route.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/api/db.py src/api/main.py src/api/routes/chat.py tests/api/test_chat_route.py
git commit -m "feat: add RAG chat endpoint"
```

---

### Task 9: `routes/quiz.py` — quiz generation and submission

**Files:**
- Create: `src/api/routes/quiz.py`
- Modify: `src/api/main.py`
- Create: `tests/api/test_quiz_route.py`

**Interfaces:**
- Consumes: `call_groq_json_schema` (Task 5), `get_current_user_id` (Task 6), `QuizRequest`/`QuizResponse`/`QuizSubmitRequest` (Task 7), a DB connection (Task 8's `get_db`).
- Produces: `POST /chapters/{chapter_number}/quiz`, `POST /quiz/submit`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/api/test_quiz_route.py
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_quiz_requires_authorization_header():
    response = client.post(
        "/chapters/1/quiz",
        json={"subject": "Environmental Studies", "class": "5"},
    )
    assert response.status_code in (401, 422)


def test_quiz_submit_requires_authorization_header():
    response = client.post(
        "/quiz/submit",
        json={
            "subject": "Environmental Studies",
            "class": "5",
            "chapter_number": 1,
            "quiz_json": {"quiz": []},
            "student_answers": {},
        },
    )
    assert response.status_code in (401, 422)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_quiz_route.py -v
```

Expected: FAIL — both routes don't exist yet, so FastAPI returns `404` instead of `401`/`422`.

- [ ] **Step 3: Write `src/api/routes/quiz.py`**

```python
# src/api/routes/quiz.py
import json

from fastapi import APIRouter, Depends

from api.auth import get_current_user_id
from api.db import get_db
from api.groq_client import call_groq_json_schema
from api.schemas import QuizRequest, QuizResponse, QuizSubmitRequest

router = APIRouter()

QUIZ_PROMPT_TEMPLATE = """You are a quiz generator for {subject} Class {class_}. Generate exactly 5 MCQs from the context below.
Context: {context}
Return valid JSON matching the required schema."""

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "quiz": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "correct_answer": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "explanation": {"type": "string"},
                },
                "required": ["question", "options", "correct_answer", "difficulty", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["quiz"],
    "additionalProperties": False,
}


@router.post("/chapters/{chapter_number}/quiz", response_model=QuizResponse)
def generate_quiz(chapter_number: int, request: QuizRequest, user_id: str = Depends(get_current_user_id)):
    conn = next(get_db())
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select chunk_text from chunks
                where subject = %s and class = %s and chapter_number = %s
                order by id
                """,
                (request.subject, request.class_, chapter_number),
            )
            context = "\n\n".join(row[0] for row in cur.fetchall())

        prompt = QUIZ_PROMPT_TEMPLATE.format(
            subject=request.subject, class_=request.class_, context=context
        )
        result = call_groq_json_schema(
            prompt, model="openai/gpt-oss-120b", schema=QUIZ_SCHEMA, schema_name="quiz"
        )
        return QuizResponse.model_validate(result)
    finally:
        conn.close()


@router.post("/quiz/submit")
def submit_quiz(request: QuizSubmitRequest, user_id: str = Depends(get_current_user_id)):
    correct_answers = {
        q["question"]: q["correct_answer"] for q in request.quiz_json.get("quiz", [])
    }
    score = sum(
        1
        for question, answer in request.student_answers.items()
        if correct_answers.get(question) == answer
    )

    conn = next(get_db())
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into quiz_attempts
                    (user_id, chapter_number, subject, class, quiz_json, student_answers, score)
                values (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    request.chapter_number,
                    request.subject,
                    request.class_,
                    json.dumps(request.quiz_json),
                    json.dumps(request.student_answers),
                    score,
                ),
            )
        conn.commit()
        return {"score": score, "total": len(correct_answers)}
    finally:
        conn.close()
```

- [ ] **Step 4: Register the router in `src/api/main.py`**

```python
# src/api/main.py
from fastapi import FastAPI

from api.routes import chat, quiz

app = FastAPI(title="RAG Learning Platform API")
app.include_router(chat.router)
app.include_router(quiz.router)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_quiz_route.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/quiz.py src/api/main.py tests/api/test_quiz_route.py
git commit -m "feat: add quiz generation and submission endpoints"
```

---

### Task 10: `routes/progress.py` — progress summary endpoint

**Files:**
- Create: `src/api/routes/progress.py`
- Modify: `src/api/main.py`
- Create: `tests/api/test_progress_route.py`

**Interfaces:**
- Consumes: `get_current_user_id` (Task 6), a DB connection (`get_db`, Task 8).
- Produces: `GET /progress/{user_id}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/api/test_progress_route.py
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_progress_requires_authorization_header():
    response = client.get("/progress/00000000-0000-0000-0000-000000000000")
    assert response.status_code in (401, 422)


def test_progress_rejects_invalid_token():
    response = client.get(
        "/progress/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_progress_route.py -v
```

Expected: FAIL with `404` (route doesn't exist yet).

- [ ] **Step 3: Write `src/api/routes/progress.py`**

```python
# src/api/routes/progress.py
from fastapi import APIRouter, Depends, HTTPException

from api.auth import get_current_user_id
from api.db import get_db

router = APIRouter()


@router.get("/progress/{user_id}")
def get_progress(user_id: str, current_user_id: str = Depends(get_current_user_id)):
    if user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Cannot view another user's progress")

    conn = next(get_db())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select chapter_number, subject, question, answer, created_at "
                "from chat_history where user_id = %s order by created_at desc limit 20",
                (user_id,),
            )
            chat_columns = [d[0] for d in cur.description]
            chat = [dict(zip(chat_columns, row)) for row in cur.fetchall()]

            cur.execute(
                "select chapter_number, subject, score, attempted_at "
                "from quiz_attempts where user_id = %s order by attempted_at desc limit 20",
                (user_id,),
            )
            quiz_columns = [d[0] for d in cur.description]
            quizzes = [dict(zip(quiz_columns, row)) for row in cur.fetchall()]

        return {"chat_history": chat, "quiz_attempts": quizzes}
    finally:
        conn.close()
```

- [ ] **Step 4: Register the router in `src/api/main.py`**

```python
# src/api/main.py
from fastapi import FastAPI

from api.routes import chat, progress, quiz

app = FastAPI(title="RAG Learning Platform API")
app.include_router(chat.router)
app.include_router(quiz.router)
app.include_router(progress.router)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_progress_route.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/progress.py src/api/main.py tests/api/test_progress_route.py
git commit -m "feat: add progress summary endpoint"
```

---

### Task 11: End-to-end verification

**Files:** none (verification only).

**Interfaces:** none — confirms Tasks 1-10 work together as a real running service, not just in isolated tests.

- [ ] **Step 1: Run the full test suite**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" -v 2>&1 | tail -10
```

Expected: all tests pass (68 from before this plan + the new tests added across Tasks 3-10).

- [ ] **Step 2: Start the server**

```bash
"D:/Coding/edtech/.venv/Scripts/uvicorn" api.main:app --app-dir src --reload
```

Expected: starts without error, listening on `http://127.0.0.1:8000`.

- [ ] **Step 3: Obtain a real Supabase JWT for manual testing**

Auth (sign-up/login) is owned by another developer and already implemented elsewhere — do not create test users via a raw signup API call here. Ask the user for a valid `access_token` from that existing flow (e.g. logging in through the frontend they built and copying the session token, or however they'd prefer to hand one over) before proceeding with Steps 4-7.

- [ ] **Step 4: Hit the chat endpoint with a real question**

```bash
curl -X POST "http://127.0.0.1:8000/chapters/1/chat" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"subject":"Environmental Studies","class":"5","question":"What is a star?"}'
```

Expected: a grounded answer referencing the actual EVS Chapter 1 content (stars, the sun, twinkling vs. not), with `source_chunk_ids` all prefixed `MSB_EVS5_CH01_`. Manually inspect the answer for quality and groundedness — this mirrors the pilot review process from the ingestion phase.

- [ ] **Step 5: Hit the quiz endpoint**

```bash
curl -X POST "http://127.0.0.1:8000/chapters/1/quiz" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"subject":"Environmental Studies","class":"5"}'
```

Expected: valid JSON matching the quiz schema exactly, 5 questions grounded in Chapter 1 content.

- [ ] **Step 6: Verify chat_history and quiz row were recorded**

Use the Supabase MCP tool `execute_sql` with `project_id="aedsgktxvmddarqurbhi"`:

```sql
select count(*) from chat_history;
```

Expected: at least 1 row (from Step 4).

- [ ] **Step 7: Hit the progress endpoint**

```bash
curl "http://127.0.0.1:8000/progress/<user id from the signup response>" \
  -H "Authorization: Bearer <access_token>"
```

Expected: JSON containing the chat entry from Step 4.

- [ ] **Step 8: Report results to the user**

Summarize: test suite status, a sample of the actual chat answer and quiz output for quality review, and confirm all endpoints behave as expected. No commit needed for this task (read-only/manual verification).

---

## Self-Review Notes

- **Spec coverage:** JWT verification via JWKS → Task 6. pgvector search scoped by subject/class/chapter → Task 4. RAG chat with the exact Appendix A.1 prompt template → Task 8. Quiz generation with strict schema on `gpt-oss-120b` → Task 9. `chat_history`/`quiz_attempts` persistence with RLS (defense in depth) plus application-layer ownership checks (the actual enforcement point given the direct-connection architecture) → Task 2 (schema) + Tasks 8-10 (application code). Progress endpoint → Task 10. Frontend, Teacher/Parent/Admin roles, other learning modes, adaptive paths, and rate limiting are explicitly out of scope per the spec and have no tasks here.
- **Type consistency:** `search_chunks(conn, query_embedding, subject, class_, chapter_number, top_k)` signature (Task 4) matches its call site in `routes/chat.py` (Task 8) exactly, including keyword `class_` vs the SQL column `class` (handled inside `search_chunks`, not leaked to callers). `ChatRequest.class_`/`QuizRequest.class_`/`QuizSubmitRequest.class_` (Task 7, all aliased from wire-field `"class"`) are used consistently as `request.class_` in Tasks 8-9. `get_current_user_id` returns a `str` (the JWT `sub` claim) used identically as `user_id` in Tasks 8-10's SQL parameters.
- **No placeholders:** every step has runnable code or an exact command with expected output. Task 11 Steps 3-7 are manual verification against a real running server and a real (test) Supabase user — intentionally not automated, since fabricating a valid Supabase-issued JWT in a unit test would test nothing real about the actual auth flow; the unit tests in Task 6 instead verify the verification *logic* in isolation with a locally-generated keypair, which is the correct scope for automated testing here.
