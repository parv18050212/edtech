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
