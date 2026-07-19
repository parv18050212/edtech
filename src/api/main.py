from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import ask, catalog, chat, progress, quiz

app = FastAPI(title="RAG Learning Platform API")

# POC: allow any origin. Bearer-token auth works without credential cookies,
# so allow_credentials stays False. Tighten allow_origins to the real
# frontend origin before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(catalog.router)
app.include_router(chat.router)
app.include_router(quiz.router)
app.include_router(progress.router)
app.include_router(ask.router)
