SEARCH_SQL = """
select chunk_id, chapter_name, topic, chunk_type, chunk_text, page_start, page_end,
       1 - (embedding <=> %(embedding)s::vector) as similarity
from chunks
where subject = %(subject)s and class = %(class)s and chapter_number = %(chapter_number)s
order by embedding <=> %(embedding)s::vector
limit %(top_k)s
"""


def search_chunks(
    conn,
    query_embedding: list[float],
    subject: str,
    class_: str,
    chapter_number: int,
    top_k: int = 5,
) -> list[dict]:
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
    with conn.cursor() as cur:
        cur.execute(
            SEARCH_SQL,
            {
                "embedding": embedding_literal,
                "subject": subject,
                "class": class_,
                "chapter_number": chapter_number,
                "top_k": top_k,
            },
        )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
