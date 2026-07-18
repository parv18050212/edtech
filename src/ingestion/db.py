INSERT_SQL = """
insert into chunks (
    chunk_id, board, class, subject, book_title, chapter_number,
    chapter_name, topic, chunk_type, page_start, page_end, chunk_text, embedding
) values %s
on conflict (chunk_id) do nothing
"""

INSERT_TEMPLATE = "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)"


def row_from_record(d: dict) -> tuple:
    embedding_literal = "[" + ",".join(str(x) for x in d["embedding"]) + "]"
    return (
        d["chunk_id"],
        d["board"],
        d["class"],
        d["subject"],
        d["book_title"],
        d["chapter_number"],
        d["chapter_name"],
        d["topic"],
        d["chunk_type"],
        d["page_start"],
        d["page_end"],
        d["chunk_text"],
        embedding_literal,
    )
