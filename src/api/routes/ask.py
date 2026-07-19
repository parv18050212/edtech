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
