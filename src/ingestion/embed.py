from typing import Optional

import requests

DEFAULT_MODEL = "embeddinggemma"
DEFAULT_HOST = "http://localhost:11434"


def build_document_prompt(text: str, title: Optional[str]) -> str:
    title_value = title if title else "none"
    return f"title: {title_value} | text: {text}"


def embed_texts(
    texts: list[str],
    titles: list[Optional[str]],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
) -> list[list[float]]:
    prompts = [
        build_document_prompt(text, title) for text, title in zip(texts, titles)
    ]
    response = requests.post(
        f"{host}/api/embed",
        json={"model": model, "input": prompts},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"]
