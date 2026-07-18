from fastapi import APIRouter, Depends, HTTPException

from api.auth import get_current_user_id
from api.db import get_connection

router = APIRouter()


@router.get("/progress/{user_id}")
def get_progress(user_id: str, current_user_id: str = Depends(get_current_user_id)):
    if user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Cannot view another user's progress")

    conn = get_connection()
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
