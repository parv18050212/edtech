# Orchestrator: Intent Detection + Multi-Output Fan-out — Design

## Context

The platform currently exposes retrieval-backed capabilities as separate,
explicit endpoints: `/chapters/{n}/chat` (RAG explanation) and
`/chapters/{n}/quiz` (quiz generation). The target architecture (the
student-flow diagram) has a single natural-language question flow through
**query processing → intent detection → chapter-wise retrieval → grounded
LLM → one of several output types** (explanation, step-by-step solution,
examples, practice questions, quiz, follow-up questions).

This slice builds the missing **orchestration half** of that diagram: a
LangGraph orchestrator that detects intent, retrieves once, routes to the
appropriate output generator, and always appends follow-up questions. It is
exposed as a new unified endpoint, leaving the existing endpoints intact.

## Verified facts

- **LangGraph** `0.2.76` is installed and imports cleanly with the pinned
  `langchain-core 0.3.86` (`from langgraph.graph import StateGraph, START,
  END` works). An orphaned `langgraph-prebuilt 1.1.0` from an earlier
  install is present but unused.
- **Reusable existing components**, all confirmed working this session:
  - `ingestion.embed.embed_query` — query-format embedding.
  - `api.retrieval.search_chunks(conn, embedding, subject, class_,
    chapter_number, top_k)` — chapter-scoped pgvector search (0.99 context
    precision in RAGAS).
  - `api.quiz_agent.generate_quiz(context, subject, class_, question_types,
    difficulty, count)` — LangChain `ChatGroq` quiz generator.
  - `api.routes.chat.EXPLANATION_PROMPT_TEMPLATE` — the explanation prompt.
  - `api.groq_client.call_groq` — retry-hardened Groq text call.
  - `api.auth.get_current_user_id`, `api.db.get_connection`.
- **LangChain quiz agent** drives Groq via `ChatGroq` + bound strict
  `response_format`. New LLM nodes here follow the same `ChatGroq` pattern.

## Scope

- Intent detection over the student's question.
- A LangGraph `StateGraph` orchestrating: detect intent → retrieve → route
  to one primary output generator → append follow-ups.
- Two new output generators: **step-by-step solution** and **practice
  questions**; plus a **follow-up questions** generator that always runs.
- Reuse existing generators for **explanation** (chat prompt) and **quiz**
  (quiz agent).
- A new `POST /chapters/{n}/ask` endpoint running the orchestrator, logging
  to `chat_history`.

Out of scope: changing/replacing existing `/chat` or `/quiz` endpoints;
multi-intent routing (one primary intent per question); diagram/formula
chunk types; the frontend.

## Intents and output generators

Intent detection classifies each question into exactly one primary intent.
Follow-up questions are always generated regardless of intent.

| Intent | Output field | Generator |
|---|---|---|
| `explanation` | `explanation: str` | Reuse `EXPLANATION_PROMPT_TEMPLATE` + `call_groq` |
| `step_by_step` | `step_by_step: str` | New prompt (numbered solution), `call_groq` |
| `quiz` | `quiz: list[QuizQuestion]` | Reuse `quiz_agent.generate_quiz` |
| `practice` | `practice_questions: list[str]` | New prompt (open questions), `ChatGroq` strict schema |
| _(always)_ | `follow_up_questions: list[str]` | New prompt (3 next questions), `ChatGroq` strict schema |

`quiz` = structured, gradable questions via the quiz agent (defaults:
`["mcq"]`, `mixed`, 5). `practice` = a few short open-ended questions to
try, no options/scoring — deliberately lighter than a quiz.

### Intent detection

A single classification call. Implemented as a `ChatGroq` call bound to a
strict `response_format` returning `{"intent": <one of the four>}`, using
`llama-3.1-8b-instant` (fast, cheap, sufficient for 4-way classification).
The prompt describes each intent with a short cue ("asks to be tested" →
quiz; "solve/calculate this" → step_by_step; "give me questions to
practice" → practice; otherwise → explanation) and defaults to
`explanation` when ambiguous. Strict schema guarantees a valid label.

## LangGraph orchestrator

`src/api/orchestrator.py` builds a `StateGraph`. State is a `TypedDict`:

```python
class OrchestratorState(TypedDict):
    # inputs
    question: str
    subject: str
    class_: str
    chapter_number: int
    # populated by nodes
    intent: str
    context: str
    source_chunk_ids: list[str]
    explanation: Optional[str]
    step_by_step: Optional[str]
    quiz: Optional[list]
    practice_questions: Optional[list[str]]
    follow_up_questions: list[str]
```

Nodes (each small, single-purpose, independently testable):

- `detect_intent_node` — sets `intent`.
- `retrieve_node` — `embed_query(question)` → `search_chunks(...)` (needs a
  DB connection; injected, see below) → sets `context` (joined chunk text)
  and `source_chunk_ids`.
- `explanation_node` / `step_by_step_node` / `quiz_node` /
  `practice_node` — each sets its own output field from `context`.
- `follow_up_node` — sets `follow_up_questions`.

Flow:

```
START → detect_intent → retrieve → (conditional route on intent)
      → { explanation | step_by_step | quiz | practice }
      → follow_up → END
```

The conditional edge after `retrieve` uses `state["intent"]` to select one
generator node. All generator nodes converge on `follow_up`, then END.

**DB connection.** `retrieve_node` and `quiz_node` need Postgres/retrieval.
The graph is built by a factory `build_orchestrator(conn)` that closes over
a live connection, so nodes call `search_chunks(conn, ...)` without the
connection living in the serialized state. The route handler opens the
connection, builds the graph, invokes it, and closes the connection —
mirroring the existing per-request connection pattern.

## API and response

`POST /chapters/{chapter_number}/ask`, body `{subject, class, question}`
(same shape as `ChatRequest`). Auth required (`get_current_user_id`).

Response model `AskResponse`:

```python
class AskResponse(BaseModel):
    intent: str
    explanation: Optional[str] = None
    step_by_step: Optional[str] = None
    quiz: Optional[list[QuizQuestion]] = None
    practice_questions: Optional[list[str]] = None
    follow_up_questions: list[str]
    source_chunk_ids: list[str]
```

Only the field for the detected intent is populated; the rest stay null.
The handler runs the graph, then logs to `chat_history` (question + a text
representation of the primary output — the explanation/step_by_step text,
or a short synthesized summary for quiz/practice) using the same insert as
`/chat`.

## Testing strategy

Consistent with the project's real-systems testing preference:

- **Intent detection** — unit test the classifier against real Groq on a
  handful of representative questions ("Explain photosynthesis" →
  explanation; "Quiz me on the water cycle" → quiz; "Solve: a force of 10N
  over 2m²" → step_by_step; "Give me practice questions on atoms" →
  practice).
- **Each new generator** (`step_by_step`, `practice`, `follow_up`) — real
  Groq call, assert output shape (non-empty text; list of N questions).
- **Orchestrator graph** — build with a real DB connection, invoke on a
  real question, assert the correct output field is populated for the
  detected intent, follow-ups present, and `source_chunk_ids` are from the
  requested chapter.
- **`/ask` route** — auth-rejection tests (401/422) via `TestClient`, like
  the other routes. Authenticated success path deferred to manual
  end-to-end (needs a real JWT), as with `/chat` and `/quiz`.

## Dependency

Add `langgraph==0.2.76` to `requirements.txt` (runtime dependency of the
orchestrator).
