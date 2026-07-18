import json
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2.0


def _headers(api_key: Optional[str]) -> dict:
    key = api_key or os.environ["GROQ_API_KEY"]
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _post_with_retry(payload: dict, api_key: Optional[str]) -> requests.Response:
    """POST to Groq, retrying on 429 rate limits with exponential backoff.

    Honors the server's Retry-After header when present, else backs off
    exponentially. Raises for the final response if retries are exhausted.
    """
    for attempt in range(MAX_RETRIES):
        response = requests.post(
            GROQ_API_URL, headers=_headers(api_key), json=payload, timeout=60
        )
        if response.status_code != 429 or attempt == MAX_RETRIES - 1:
            response.raise_for_status()
            return response
        retry_after = response.headers.get("retry-after")
        wait = float(retry_after) if retry_after else BASE_BACKOFF_SECONDS * (2**attempt)
        time.sleep(wait)
    return response  # pragma: no cover - loop always returns/raises above


def call_groq(prompt: str, model: str, api_key: Optional[str] = None) -> str:
    response = _post_with_retry(
        {"model": model, "messages": [{"role": "user", "content": prompt}]}, api_key
    )
    return response.json()["choices"][0]["message"]["content"]


def call_groq_json_schema(
    prompt: str,
    model: str,
    schema: dict,
    schema_name: str,
    api_key: Optional[str] = None,
) -> dict:
    response = _post_with_retry(
        {
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
        api_key,
    )
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)
