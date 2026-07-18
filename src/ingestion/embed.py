from typing import Optional

import requests

DEFAULT_MODEL = "embeddinggemma"
DEFAULT_HOST = "http://localhost:11434"


def build_document_prompt(text: str, title: Optional[str]) -> str:
    title_value = title if title else "none"
    return f"title: {title_value} | text: {text}"


def build_query_prompt(text: str) -> str:
    return f"task: search result | query: {text}"


def embed_prompts(
    prompts: list[str],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
) -> list[list[float]]:
    response = requests.post(
        f"{host}/api/embed",
        json={"model": model, "input": prompts},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def embed_texts(
    texts: list[str],
    titles: list[Optional[str]],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
) -> list[list[float]]:
    prompts = [
        build_document_prompt(text, title) for text, title in zip(texts, titles)
    ]
    return embed_prompts(prompts, model=model, host=host)


def embed_query(
    text: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST
) -> list[float]:
    return embed_prompts([build_query_prompt(text)], model=model, host=host)[0]
