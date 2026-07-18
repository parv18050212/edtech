from fastapi import FastAPI

from api.routes import chat, progress, quiz

app = FastAPI(title="RAG Learning Platform API")
app.include_router(chat.router)
app.include_router(quiz.router)
app.include_router(progress.router)
