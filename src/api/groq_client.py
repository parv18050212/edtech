import json
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _headers(api_key: Optional[str]) -> dict:
    key = api_key or os.environ["GROQ_API_KEY"]
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def call_groq(prompt: str, model: str, api_key: Optional[str] = None) -> str:
    response = requests.post(
        GROQ_API_URL,
        headers=_headers(api_key),
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_groq_json_schema(
    prompt: str,
    model: str,
    schema: dict,
    schema_name: str,
    api_key: Optional[str] = None,
) -> dict:
    response = requests.post(
        GROQ_API_URL,
        headers=_headers(api_key),
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)
