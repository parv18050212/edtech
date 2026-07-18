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
