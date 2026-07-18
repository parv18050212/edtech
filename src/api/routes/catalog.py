from fastapi import APIRouter, Query

from api.db import get_connection

router = APIRouter()


@router.get("/books")
def list_books():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select distinct board, class, subject, book_title from chunks "
                "order by class, subject"
            )
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()


@router.get("/chapters")
def list_chapters(subject: str, class_: str = Query(alias="class")):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select distinct chapter_number, chapter_name from chunks "
                "where subject = %s and class = %s order by chapter_number",
                (subject, class_),
            )
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()
