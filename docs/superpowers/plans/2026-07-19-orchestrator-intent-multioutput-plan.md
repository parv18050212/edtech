# Orchestrator (Intent + Multi-Output) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LangGraph orchestrator that detects a student question's intent, retrieves chapter content once, routes to the right output generator (explanation / step-by-step / quiz / practice), always appends follow-up questions, and exposes it as `POST /chapters/{n}/ask`, per `docs/superpowers/specs/2026-07-19-orchestrator-intent-multioutput-design.md`.

**Architecture:** A `StateGraph` (`src/api/agents.py` for the LLM nodes, `src/api/orchestrator.py` for the graph) wires small single-purpose nodes: detect_intent → retrieve → conditional route to one generator → follow_up → END. New LLM nodes use LangChain `ChatGroq` (same pattern as the quiz agent); retrieval/explanation/quiz reuse existing functions. The graph is built per-request by a factory closing over a live DB connection.

**Tech Stack:** Python 3.12, LangGraph 0.2.76, langchain-groq (ChatGroq), FastAPI, psycopg2, pytest.

## Global Constraints

- Add `langgraph==0.2.76` to `requirements.txt` (confirmed installed and importing with the pinned `langchain-core 0.3.86`).
- New LLM nodes call Groq through LangChain `ChatGroq`, consistent with `api.quiz_agent`. Structured outputs (intent label, practice questions, follow-ups) bind Groq's native strict `response_format` json_schema — NOT LangChain's tool-calling or json_mode, both of which `gpt-oss` intermittently breaks (established while building the quiz agent).
- Intent classification model: `llama-3.1-8b-instant` (fast, cheap). Free-text generation (step_by_step) uses `call_groq` with `llama-3.3-70b-versatile` (same as chat).
- Reuse verbatim (do not reimplement): `ingestion.embed.embed_query`; `api.retrieval.search_chunks(conn, embedding, subject, class_, chapter_number, top_k=5)`; `api.quiz_agent.generate_quiz(context, subject, class_, question_types, difficulty, count)`; `api.routes.chat.EXPLANATION_PROMPT_TEMPLATE`; `api.groq_client.call_groq(prompt, model)`; `api.auth.get_current_user_id`; `api.db.get_connection`.
- `class` is the wire/DB name for Python `class_` (existing convention via Pydantic alias).
- One primary intent per question; follow-ups always generated. Intents: `explanation`, `step_by_step`, `quiz`, `practice`.
- Every DB-touching route opens its own connection via `get_connection()` and closes it in `finally` (NOT `next(get_db())` — that was a fixed bug).
- `tests/api/__init__.py` already exists; no package-init work needed.

---

## File Structure

```
edtech/
  src/api/
    agents.py           # CREATE: LLM node functions (intent, step_by_step, practice, follow_up) + their schemas/prompts
    orchestrator.py     # CREATE: OrchestratorState + build_orchestrator(conn) StateGraph factory
    schemas.py          # MODIFY: add AskResponse
    routes/
      ask.py            # CREATE: POST /chapters/{n}/ask
    main.py             # MODIFY: register ask.router
  tests/api/
    test_agents.py            # CREATE: intent classifier + generators (real Groq)
    test_orchestrator.py      # CREATE: graph end-to-end (real Groq + real DB)
    test_ask_route.py         # CREATE: auth-rejection tests
  requirements.txt      # MODIFY: add langgraph==0.2.76
```

**Why two modules (`agents.py` + `orchestrator.py`):** the LLM node logic (prompts, schemas, Groq calls) is independently testable and has one responsibility; the graph wiring is a separate concern. Keeping them apart keeps each file focused and lets the nodes be unit-tested without constructing a graph.

---

### Task 1: Dependency

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `langgraph` importable for later tasks.

- [ ] **Step 1: Add langgraph to `requirements.txt`**

Append this line to `requirements.txt` (after `langchain-groq==0.2.5`):

```
langgraph==0.2.76
```

- [ ] **Step 2: Install and verify**

```bash
"D:/Coding/edtech/.venv/Scripts/pip" install -r requirements.txt
"D:/Coding/edtech/.venv/Scripts/python" -c "from langgraph.graph import StateGraph, START, END; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add langgraph dependency"
```

---

### Task 2: Intent classifier

**Files:**
- Create: `src/api/agents.py`
- Test: `tests/api/test_agents.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (uses `langchain_groq.ChatGroq`).
- Produces: `INTENTS: set[str]` (`{"explanation", "step_by_step", "quiz", "practice"}`), `detect_intent(question: str, api_key: Optional[str] = None) -> str` returning one of `INTENTS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_agents.py
from api.agents import INTENTS, detect_intent


def test_intents_constant():
    assert INTENTS == {"explanation", "step_by_step", "quiz", "practice"}


def test_detect_intent_explanation():
    assert detect_intent("Explain how photosynthesis works") == "explanation"


def test_detect_intent_quiz():
    assert detect_intent("Quiz me on the water cycle") == "quiz"


def test_detect_intent_step_by_step():
    assert detect_intent(
        "Solve: a force of 10 N acts on an area of 2 square metres, find the pressure"
    ) == "step_by_step"


def test_detect_intent_practice():
    assert detect_intent("Give me some practice questions on atoms") == "practice"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_agents.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.agents'`.

- [ ] **Step 3: Write the implementation**

```python
# src/api/agents.py
import json
from typing import Optional

from langchain_groq import ChatGroq

from api.groq_client import call_groq

INTENT_MODEL = "llama-3.1-8b-instant"
GENERATION_MODEL = "llama-3.3-70b-versatile"

INTENTS = {"explanation", "step_by_step", "quiz", "practice"}

_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": sorted(INTENTS)},
    },
    "required": ["intent"],
    "additionalProperties": False,
}

_INTENT_PROMPT = """Classify the student's request into exactly one intent:
- "quiz": they want to be tested with a structured/gradable quiz.
- "practice": they want practice questions to try on their own.
- "step_by_step": they want a problem solved or worked out step by step (calculations, numericals, "solve", "find", "calculate").
- "explanation": they want a concept explained. This is the default when unsure.

Student request: {question}

Return JSON with the single "intent" field."""


def _groq(model: str, api_key: Optional[str], response_format: Optional[dict] = None):
    kwargs = {"model": model, "temperature": 0}
    if api_key:
        kwargs["api_key"] = api_key
    llm = ChatGroq(**kwargs)
    if response_format is not None:
        llm = llm.bind(response_format=response_format)
    return llm


def detect_intent(question: str, api_key: Optional[str] = None) -> str:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "intent", "strict": True, "schema": _INTENT_SCHEMA},
    }
    message = _groq(INTENT_MODEL, api_key, response_format).invoke(
        _INTENT_PROMPT.format(question=question)
    )
    return json.loads(message.content)["intent"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_agents.py -v
```

Expected: 5 passed. (Real Groq calls; requires `GROQ_API_KEY` in `.env`.)

- [ ] **Step 5: Commit**

```bash
git add src/api/agents.py tests/api/test_agents.py
git commit -m "feat: add intent classifier"
```

---

### Task 3: Step-by-step and practice generators

**Files:**
- Modify: `src/api/agents.py`
- Modify: `tests/api/test_agents.py`

**Interfaces:**
- Consumes: `call_groq` (via existing import), `_groq` helper (Task 2).
- Produces: `generate_step_by_step(question: str, context: str) -> str`; `generate_practice(question: str, context: str, api_key: Optional[str] = None) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_agents.py`:

```python
from api.agents import generate_practice, generate_step_by_step

_ATOM_CONTEXT = (
    "Pressure is the force acting per unit area. Pressure = Force / Area. "
    "Force is measured in newtons (N) and area in square metres. "
    "An atom contains protons, neutrons and electrons."
)


def test_generate_step_by_step_returns_nonempty_text():
    out = generate_step_by_step(
        "A force of 10 N acts on an area of 2 square metres. Find the pressure.",
        _ATOM_CONTEXT,
    )
    assert isinstance(out, str) and out.strip()
    assert "5" in out  # 10 / 2 = 5


def test_generate_practice_returns_list_of_questions():
    out = generate_practice("Give me practice questions on atoms", _ATOM_CONTEXT)
    assert isinstance(out, list)
    assert 1 <= len(out) <= 10
    assert all(isinstance(q, str) and q.strip() for q in out)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_agents.py -k "step_by_step or practice" -v
```

Expected: FAIL with `ImportError: cannot import name 'generate_step_by_step'`.

- [ ] **Step 3: Write the implementation**

Append to `src/api/agents.py`:

```python
_STEP_BY_STEP_PROMPT = """You are a patient tutor. Using ONLY the context below, solve the student's problem as a clear numbered list of steps, showing the reasoning and any calculation at each step, then state the final answer on its own line.

Context:
{context}

Problem: {question}"""

_PRACTICE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_PRACTICE_PROMPT = """You are a tutor. Using ONLY the context below, write 3 open-ended practice questions (no options, no answers) that let a student practise this topic. Base every question on the context; do not invent facts.

Context:
{context}

Topic / request: {question}

Return JSON with a "questions" array of strings."""


def generate_step_by_step(question: str, context: str) -> str:
    prompt = _STEP_BY_STEP_PROMPT.format(context=context, question=question)
    return call_groq(prompt, model=GENERATION_MODEL)


def generate_practice(
    question: str, context: str, api_key: Optional[str] = None
) -> list[str]:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "practice", "strict": True, "schema": _PRACTICE_SCHEMA},
    }
    message = _groq(GENERATION_MODEL, api_key, response_format).invoke(
        _PRACTICE_PROMPT.format(context=context, question=question)
    )
    return json.loads(message.content)["questions"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_agents.py -k "step_by_step or practice" -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/agents.py tests/api/test_agents.py
git commit -m "feat: add step-by-step and practice generators"
```

---

### Task 4: Follow-up questions generator

**Files:**
- Modify: `src/api/agents.py`
- Modify: `tests/api/test_agents.py`

**Interfaces:**
- Consumes: `_groq` helper (Task 2).
- Produces: `generate_follow_ups(question: str, context: str, api_key: Optional[str] = None) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_agents.py`:

```python
from api.agents import generate_follow_ups


def test_generate_follow_ups_returns_questions():
    out = generate_follow_ups("What is a star?", _ATOM_CONTEXT)
    assert isinstance(out, list)
    assert 1 <= len(out) <= 5
    assert all(isinstance(q, str) and q.strip() for q in out)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_agents.py -k follow_ups -v
```

Expected: FAIL with `ImportError: cannot import name 'generate_follow_ups'`.

- [ ] **Step 3: Write the implementation**

Append to `src/api/agents.py`:

```python
_FOLLOW_UP_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_FOLLOW_UP_PROMPT = """A student asked: "{question}"

Based on the context below, suggest exactly 3 natural follow-up questions the student might ask next to deepen their understanding. Keep them short.

Context:
{context}

Return JSON with a "questions" array of 3 strings."""


def generate_follow_ups(
    question: str, context: str, api_key: Optional[str] = None
) -> list[str]:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "followups", "strict": True, "schema": _FOLLOW_UP_SCHEMA},
    }
    message = _groq(GENERATION_MODEL, api_key, response_format).invoke(
        _FOLLOW_UP_PROMPT.format(context=context, question=question)
    )
    return json.loads(message.content)["questions"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_agents.py -k follow_ups -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/agents.py tests/api/test_agents.py
git commit -m "feat: add follow-up questions generator"
```

---

### Task 5: AskResponse schema

**Files:**
- Modify: `src/api/schemas.py`
- Modify: `tests/api/test_schemas.py`

**Interfaces:**
- Consumes: existing `QuizQuestion`.
- Produces: `AskResponse` with fields `intent: str`, `explanation: Optional[str] = None`, `step_by_step: Optional[str] = None`, `quiz: Optional[list[QuizQuestion]] = None`, `practice_questions: Optional[list[str]] = None`, `follow_up_questions: list[str]`, `source_chunk_ids: list[str]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_schemas.py`:

```python
from api.schemas import AskResponse


def test_ask_response_populates_only_relevant_field():
    resp = AskResponse(
        intent="explanation",
        explanation="A star is a ball of gas.",
        follow_up_questions=["What is a planet?"],
        source_chunk_ids=["MSB_EVS5_CH01_TOP00_000"],
    )
    assert resp.intent == "explanation"
    assert resp.explanation == "A star is a ball of gas."
    assert resp.step_by_step is None
    assert resp.quiz is None
    assert resp.practice_questions is None
    assert resp.follow_up_questions == ["What is a planet?"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_schemas.py -k ask_response -v
```

Expected: FAIL with `ImportError: cannot import name 'AskResponse'`.

- [ ] **Step 3: Write the implementation**

Add to `src/api/schemas.py` (after `QuizResponse`):

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

- [ ] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_schemas.py -k ask_response -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/schemas.py tests/api/test_schemas.py
git commit -m "feat: add AskResponse schema"
```

---

### Task 6: Orchestrator graph

**Files:**
- Create: `src/api/orchestrator.py`
- Test: `tests/api/test_orchestrator.py`

**Interfaces:**
- Consumes: `detect_intent`, `generate_step_by_step`, `generate_practice`, `generate_follow_ups` (Tasks 2-4); `embed_query`; `search_chunks`; `quiz_agent.generate_quiz`; `EXPLANATION_PROMPT_TEMPLATE`; `call_groq`.
- Produces: `OrchestratorState` (TypedDict); `build_orchestrator(conn)` returning a compiled graph whose `.invoke(state)` fills the output fields; `run_ask(conn, question, subject, class_, chapter_number) -> dict` convenience wrapper returning the final state.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_orchestrator.py
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import psycopg2
from dotenv import load_dotenv

from api.orchestrator import run_ask

load_dotenv()


def _conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def test_run_ask_explanation_populates_explanation_and_followups():
    conn = _conn()
    try:
        state = run_ask(
            conn,
            question="Explain what a star is",
            subject="Environmental Studies",
            class_="5",
            chapter_number=1,
        )
    finally:
        conn.close()
    assert state["intent"] == "explanation"
    assert state["explanation"] and state["explanation"].strip()
    assert state["step_by_step"] is None
    assert state["quiz"] is None
    assert len(state["follow_up_questions"]) >= 1
    assert state["source_chunk_ids"]
    assert all(c.startswith("MSB_EVS5_CH01_") for c in state["source_chunk_ids"])


def test_run_ask_quiz_populates_quiz():
    conn = _conn()
    try:
        state = run_ask(
            conn,
            question="Quiz me on the solar system",
            subject="Environmental Studies",
            class_="5",
            chapter_number=1,
        )
    finally:
        conn.close()
    assert state["intent"] == "quiz"
    assert state["quiz"] and len(state["quiz"]) >= 1
    assert state["explanation"] is None
    assert len(state["follow_up_questions"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_orchestrator.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.orchestrator'`.

- [ ] **Step 3: Write the implementation**

```python
# src/api/orchestrator.py
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from api.agents import (
    detect_intent,
    generate_follow_ups,
    generate_practice,
    generate_step_by_step,
    GENERATION_MODEL,
)
from api.groq_client import call_groq
from api.quiz_agent import generate_quiz
from api.retrieval import search_chunks
from api.routes.chat import EXPLANATION_PROMPT_TEMPLATE
from ingestion.embed import embed_query

RETRIEVAL_TOP_K = 5


class OrchestratorState(TypedDict, total=False):
    question: str
    subject: str
    class_: str
    chapter_number: int
    intent: str
    context: str
    source_chunk_ids: list
    explanation: Optional[str]
    step_by_step: Optional[str]
    quiz: Optional[list]
    practice_questions: Optional[list]
    follow_up_questions: list


def build_orchestrator(conn):
    def detect_intent_node(state: OrchestratorState) -> dict:
        return {"intent": detect_intent(state["question"])}

    def retrieve_node(state: OrchestratorState) -> dict:
        embedding = embed_query(state["question"])
        chunks = search_chunks(
            conn,
            embedding,
            state["subject"],
            state["class_"],
            state["chapter_number"],
            top_k=RETRIEVAL_TOP_K,
        )
        return {
            "context": "\n\n".join(c["chunk_text"] for c in chunks),
            "source_chunk_ids": [c["chunk_id"] for c in chunks],
        }

    def explanation_node(state: OrchestratorState) -> dict:
        prompt = EXPLANATION_PROMPT_TEMPLATE.format(
            context=state["context"], question=state["question"]
        )
        return {"explanation": call_groq(prompt, model=GENERATION_MODEL)}

    def step_by_step_node(state: OrchestratorState) -> dict:
        return {
            "step_by_step": generate_step_by_step(state["question"], state["context"])
        }

    def quiz_node(state: OrchestratorState) -> dict:
        result = generate_quiz(
            context=state["context"],
            subject=state["subject"],
            class_=state["class_"],
            question_types=["mcq"],
            difficulty="mixed",
            count=5,
        )
        return {"quiz": result["quiz"]}

    def practice_node(state: OrchestratorState) -> dict:
        return {
            "practice_questions": generate_practice(
                state["question"], state["context"]
            )
        }

    def follow_up_node(state: OrchestratorState) -> dict:
        return {
            "follow_up_questions": generate_follow_ups(
                state["question"], state["context"]
            )
        }

    graph = StateGraph(OrchestratorState)
    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("explanation", explanation_node)
    graph.add_node("step_by_step", step_by_step_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("practice", practice_node)
    graph.add_node("follow_up", follow_up_node)

    graph.add_edge(START, "detect_intent")
    graph.add_edge("detect_intent", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        lambda state: state["intent"],
        {
            "explanation": "explanation",
            "step_by_step": "step_by_step",
            "quiz": "quiz",
            "practice": "practice",
        },
    )
    for node in ("explanation", "step_by_step", "quiz", "practice"):
        graph.add_edge(node, "follow_up")
    graph.add_edge("follow_up", END)

    return graph.compile()


def run_ask(
    conn, question: str, subject: str, class_: str, chapter_number: int
) -> dict:
    graph = build_orchestrator(conn)
    initial: OrchestratorState = {
        "question": question,
        "subject": subject,
        "class_": class_,
        "chapter_number": chapter_number,
        "explanation": None,
        "step_by_step": None,
        "quiz": None,
        "practice_questions": None,
    }
    return graph.invoke(initial)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_orchestrator.py -v
```

Expected: 2 passed. (Real Groq + real DB; a few LLM calls per question, allow ~30-60s.)

- [ ] **Step 5: Commit**

```bash
git add src/api/orchestrator.py tests/api/test_orchestrator.py
git commit -m "feat: add LangGraph orchestrator (intent -> retrieve -> route -> follow-up)"
```

---

### Task 7: `/ask` route

**Files:**
- Create: `src/api/routes/ask.py`
- Modify: `src/api/main.py`
- Test: `tests/api/test_ask_route.py`

**Interfaces:**
- Consumes: `run_ask` (Task 6); `AskResponse` (Task 5); `ChatRequest` (existing, reused for the body); `get_current_user_id`; `get_connection`.
- Produces: `POST /chapters/{chapter_number}/ask`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/api/test_ask_route.py
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_ask_requires_authorization_header():
    response = client.post(
        "/chapters/1/ask",
        json={"subject": "Environmental Studies", "class": "5", "question": "What is a star?"},
    )
    assert response.status_code in (401, 422)


def test_ask_rejects_invalid_token():
    response = client.post(
        "/chapters/1/ask",
        json={"subject": "Environmental Studies", "class": "5", "question": "What is a star?"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_ask_route.py -v
```

Expected: FAIL — route missing, returns `404` instead of `401`/`422`.

- [ ] **Step 3: Write `src/api/routes/ask.py`**

```python
# src/api/routes/ask.py
import json

from fastapi import APIRouter, Depends

from api.auth import get_current_user_id
from api.db import get_connection
from api.orchestrator import run_ask
from api.schemas import AskResponse, ChatRequest

router = APIRouter()


def _answer_text(state: dict) -> str:
    """Text representation of the primary output for chat_history logging."""
    if state.get("explanation"):
        return state["explanation"]
    if state.get("step_by_step"):
        return state["step_by_step"]
    if state.get("practice_questions"):
        return "Practice questions:\n" + "\n".join(state["practice_questions"])
    if state.get("quiz"):
        return f"Generated a quiz of {len(state['quiz'])} question(s)."
    return ""


@router.post("/chapters/{chapter_number}/ask", response_model=AskResponse)
def ask(chapter_number: int, request: ChatRequest, user_id: str = Depends(get_current_user_id)):
    conn = get_connection()
    try:
        state = run_ask(
            conn,
            question=request.question,
            subject=request.subject,
            class_=request.class_,
            chapter_number=chapter_number,
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                insert into chat_history (user_id, chapter_number, subject, class, question, answer)
                values (%s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    chapter_number,
                    request.subject,
                    request.class_,
                    request.question,
                    _answer_text(state),
                ),
            )
        conn.commit()

        return AskResponse(
            intent=state["intent"],
            explanation=state.get("explanation"),
            step_by_step=state.get("step_by_step"),
            quiz=state.get("quiz"),
            practice_questions=state.get("practice_questions"),
            follow_up_questions=state.get("follow_up_questions", []),
            source_chunk_ids=state.get("source_chunk_ids", []),
        )
    finally:
        conn.close()
```

- [ ] **Step 4: Register the router in `src/api/main.py`**

Update the imports and registrations in `src/api/main.py` so it reads:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import ask, catalog, chat, progress, quiz

app = FastAPI(title="RAG Learning Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(catalog.router)
app.include_router(chat.router)
app.include_router(quiz.router)
app.include_router(progress.router)
app.include_router(ask.router)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/api/test_ask_route.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/ask.py src/api/main.py tests/api/test_ask_route.py
git commit -m "feat: add /ask orchestrator endpoint"
```

---

### Task 8: Full suite + manual end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" -q
```

Expected: all tests pass (112 before this plan + the new agent/orchestrator/ask/schema tests).

- [ ] **Step 2: Start the server**

```bash
"D:/Coding/edtech/.venv/Scripts/uvicorn" api.main:app --app-dir src --port 8000
```

Expected: starts cleanly.

- [ ] **Step 3: Drive `/ask` with a real token (needs a Supabase JWT)**

Auth is owned by the other developer; ask the user for a valid `access_token` from the existing login flow. Then, for each intent, confirm the right field is populated:

```bash
# explanation
curl -s -X POST "http://127.0.0.1:8000/chapters/1/ask" -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"subject":"Environmental Studies","class":"5","question":"Explain what a star is"}'
# quiz
curl -s -X POST "http://127.0.0.1:8000/chapters/1/ask" -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"subject":"Environmental Studies","class":"5","question":"Quiz me on the solar system"}'
# step_by_step (use a Science numerical chapter)
curl -s -X POST "http://127.0.0.1:8000/chapters/3/ask" -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"subject":"Science","class":"8","question":"A force of 20 N acts on 4 square metres, find the pressure step by step"}'
# practice
curl -s -X POST "http://127.0.0.1:8000/chapters/1/ask" -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"subject":"Environmental Studies","class":"5","question":"Give me practice questions on planets"}'
```

Expected: each response has the correct `intent`, the matching populated field, a non-empty `follow_up_questions`, and `source_chunk_ids` prefixed for the requested chapter.

- [ ] **Step 4: Report results to the user**

Summarize test status and a sample of each intent's real output. No commit (verification only).

---

## Self-Review Notes

- **Spec coverage:** Intent detection → Task 2. Step-by-step + practice generators → Task 3. Follow-up generator → Task 4. Reused explanation (chat prompt) + quiz (agent) → wired in Task 6's `explanation_node`/`quiz_node`. LangGraph `StateGraph` with detect→retrieve→route→follow-up flow and per-request connection factory → Task 6. `AskResponse` typed with only-relevant-field populated → Task 5. `POST /chapters/{n}/ask` + `chat_history` logging → Task 7. `langgraph==0.2.76` dependency → Task 1. Real-systems testing (Groq + DB) → Tasks 2-6; auth-rejection route tests + manual E2E → Tasks 7-8. All spec sections mapped.
- **Type consistency:** `detect_intent(question, api_key=None) -> str` (Task 2) called as `detect_intent(state["question"])` in Task 6. `generate_step_by_step(question, context) -> str`, `generate_practice(question, context, api_key=None) -> list[str]`, `generate_follow_ups(question, context, api_key=None) -> list[str]` (Tasks 3-4) called with matching args in Task 6. `search_chunks(conn, embedding, subject, class_, chapter_number, top_k=5)` and `generate_quiz(context, subject, class_, question_types, difficulty, count)` match their existing definitions. `OrchestratorState` field names (`explanation`, `step_by_step`, `quiz`, `practice_questions`, `follow_up_questions`, `source_chunk_ids`, `context`, `intent`) match `AskResponse` fields (Task 5) and the route's `state.get(...)` keys (Task 7). `GENERATION_MODEL` is defined in `agents.py` (Task 2) and imported by `orchestrator.py` (Task 6).
- **No placeholders:** every code step is complete; every command has an expected result. Task 8's manual E2E is the only non-automated part (needs a real JWT), consistent with `/chat` and `/quiz` from the prior plan.
- **Note on `quiz_node` context:** it feeds the question-focused retrieved `context` (top-5 chunks for the question) into the quiz agent, so an `/ask` quiz is about what the student asked within the chapter — intentional, and distinct from `/quiz` which quizzes the whole chapter/topic.
