from fastapi import FastAPI

from api.routes import chat

app = FastAPI(title="RAG Learning Platform API")
app.include_router(chat.router)
