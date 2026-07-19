import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import psycopg2
from dotenv import load_dotenv

from api.progress_service import get_quiz_progress

load_dotenv()


def _connect():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def _insert_attempt(cur, user_id, subject, class_, chapter, score, total):
    cur.execute(
        """
        insert into quiz_attempts
            (user_id, chapter_number, subject, class, quiz_json, student_answers, score, total)
        values (%s, %s, %s, %s, '{}'::jsonb, '{}'::jsonb, %s, %s)
        """,
        (user_id, chapter, subject, class_, score, total),
    )


def test_get_quiz_progress_aggregates_per_chapter():
    user_id = str(uuid.uuid4())
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # Science ch3: 3/5 (60%) and 5/5 (100%) -> avg 80, best 100
            _insert_attempt(cur, user_id, "Science", "8", 3, 3, 5)
            _insert_attempt(cur, user_id, "Science", "8", 3, 5, 5)
            # EVS ch1: 2/4 (50%) -> avg 50, best 50
            _insert_attempt(cur, user_id, "Environmental Studies", "5", 1, 2, 4)
        conn.commit()

        progress = get_quiz_progress(conn, user_id)
        by_key = {(p["subject"], p["chapter_number"]): p for p in progress}

        sci = by_key[("Science", 3)]
        assert sci["attempts"] == 2
        assert sci["avg_percent"] == 80
        assert sci["best_percent"] == 100
        assert sci["last_attempted"] is not None

        evs = by_key[("Environmental Studies", 1)]
        assert evs["attempts"] == 1
        assert evs["avg_percent"] == 50
        assert evs["best_percent"] == 50
    finally:
        with conn.cursor() as cur:
            cur.execute("delete from quiz_attempts where user_id = %s", (user_id,))
        conn.commit()
        conn.close()


def test_get_quiz_progress_empty_for_unknown_user():
    conn = _connect()
    try:
        assert get_quiz_progress(conn, str(uuid.uuid4())) == []
    finally:
        conn.close()
