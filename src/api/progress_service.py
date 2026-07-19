"""Read-side queries for student progress.

Kept separate from the route handler so the aggregation logic is
importable and testable directly against the database.
"""

_QUIZ_PROGRESS_SQL = """
select
    subject,
    chapter_number,
    count(*)::int as attempts,
    round(avg(score::numeric / nullif(total, 0)) * 100)::int as avg_percent,
    round(max(score::numeric / nullif(total, 0)) * 100)::int as best_percent,
    max(attempted_at) as last_attempted
from quiz_attempts
where user_id = %s and total is not null and total > 0
group by subject, chapter_number
order by subject, chapter_number
"""


def _rows_as_dicts(cur) -> list[dict]:
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_quiz_progress(conn, user_id: str) -> list[dict]:
    """Per-chapter quiz mastery for a user.

    Returns one row per (subject, chapter_number) the user has attempted,
    with attempt count and average/best score as a percentage. Attempts
    with no recorded total (legacy rows) are excluded from the percentages.
    """
    with conn.cursor() as cur:
        cur.execute(_QUIZ_PROGRESS_SQL, (user_id,))
        return _rows_as_dicts(cur)
