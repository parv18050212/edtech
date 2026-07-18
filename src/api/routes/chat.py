from fastapi import APIRouter, Depends

from api.auth import get_current_user_id
from api.db import get_connection
from api.groq_client import call_groq
from api.retrieval import search_chunks
from api.schemas import ChatRequest, ChatResponse
from ingestion.embed import embed_query

router = APIRouter()

EXPLANATION_PROMPT_TEMPLATE = """You are a helpful tutor. Answer using ONLY the context below.
Context: {context}
Student question: {question}
Provide: a simple explanation, one worked example, and one common mistake students make."""


@router.post("/chapters/{chapter_number}/chat", response_model=ChatResponse)
def chat(chapter_number: int, request: ChatRequest, user_id: str = Depends(get_current_user_id)):
    conn = get_connection()
    try:
        query_embedding = embed_query(request.question)
        chunks = search_chunks(
            conn, query_embedding, request.subject, request.class_, chapter_number
        )
        context = "\n\n".join(c["chunk_text"] for c in chunks)
        prompt = EXPLANATION_PROMPT_TEMPLATE.format(context=context, question=request.question)
        answer = call_groq(prompt, model="llama-3.3-70b-versatile")

        with conn.cursor() as cur:
            cur.execute(
                """
                insert into chat_history (user_id, chapter_number, subject, class, question, answer)
                values (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, chapter_number, request.subject, request.class_, request.question, answer),
            )
        conn.commit()

        return ChatResponse(answer=answer, source_chunk_ids=[c["chunk_id"] for c in chunks])
    finally:
        conn.close()
