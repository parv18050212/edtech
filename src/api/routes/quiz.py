import json

from fastapi import APIRouter, Depends

from api.auth import get_current_user_id
from api.db import get_connection
from api.quiz_agent import generate_quiz as agent_generate_quiz
from api.quiz_agent import score_answers
from api.retrieval import search_chunks
from api.schemas import QuizRequest, QuizResponse, QuizSubmitRequest
from ingestion.embed import embed_query

router = APIRouter()

# When a quiz is scoped to a topic, pull the most relevant chunks within the
# chapter rather than the whole chapter.
TOPIC_CONTEXT_CHUNKS = 8


def _chapter_context(conn, subject: str, class_: str, chapter_number: int) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            select chunk_text from chunks
            where subject = %s and class = %s and chapter_number = %s
            order by id
            """,
            (subject, class_, chapter_number),
        )
        return "\n\n".join(row[0] for row in cur.fetchall())


def _topic_context(
    conn, subject: str, class_: str, chapter_number: int, topic: str
) -> str:
    query_embedding = embed_query(topic)
    chunks = search_chunks(
        conn, query_embedding, subject, class_, chapter_number, top_k=TOPIC_CONTEXT_CHUNKS
    )
    return "\n\n".join(c["chunk_text"] for c in chunks)


@router.post("/chapters/{chapter_number}/quiz", response_model=QuizResponse)
def generate_quiz(chapter_number: int, request: QuizRequest, user_id: str = Depends(get_current_user_id)):
    conn = get_connection()
    try:
        if request.topic:
            context = _topic_context(
                conn, request.subject, request.class_, chapter_number, request.topic
            )
        else:
            context = _chapter_context(
                conn, request.subject, request.class_, chapter_number
            )

        result = agent_generate_quiz(
            context=context,
            subject=request.subject,
            class_=request.class_,
            question_types=request.question_types,
            difficulty=request.difficulty,
            count=request.count,
        )
        return QuizResponse.model_validate(result)
    finally:
        conn.close()


@router.post("/quiz/submit")
def submit_quiz(request: QuizSubmitRequest, user_id: str = Depends(get_current_user_id)):
    quiz_items = request.quiz_json.get("quiz", [])
    score, total = score_answers(quiz_items, request.student_answers)

    conn = get_connection()
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
        return {"score": score, "total": total}
    finally:
        conn.close()
