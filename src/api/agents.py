import json
from typing import Optional

from langchain_groq import ChatGroq

from api.groq_client import call_groq

# Intent classification uses a gpt-oss model because strict json_schema
# response format is only supported by gpt-oss on Groq (llama models 400 on
# it). gpt-oss-20b is the small/fast option and guarantees a valid label.
INTENT_MODEL = "openai/gpt-oss-20b"
GENERATION_MODEL = "llama-3.3-70b-versatile"

INTENTS = {"explanation", "step_by_step", "quiz", "practice"}

_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": sorted(INTENTS)},
    },
    "required": ["intent"],
    "additionalProperties": False,
}

_INTENT_PROMPT = """Classify the student's request into exactly one intent:
- "quiz": they want to be tested with a structured/gradable quiz.
- "practice": they want practice questions to try on their own.
- "step_by_step": they want a problem solved or worked out step by step (calculations, numericals, "solve", "find", "calculate").
- "explanation": they want a concept explained. This is the default when unsure.

Student request: {question}

Return JSON with the single "intent" field."""


def _groq(model: str, api_key: Optional[str], response_format: Optional[dict] = None):
    kwargs = {"model": model, "temperature": 0}
    if api_key:
        kwargs["api_key"] = api_key
    llm = ChatGroq(**kwargs)
    if response_format is not None:
        llm = llm.bind(response_format=response_format)
    return llm


def detect_intent(question: str, api_key: Optional[str] = None) -> str:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "intent", "strict": True, "schema": _INTENT_SCHEMA},
    }
    message = _groq(INTENT_MODEL, api_key, response_format).invoke(
        _INTENT_PROMPT.format(question=question)
    )
    return json.loads(message.content)["intent"]
