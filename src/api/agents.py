import json
from typing import Optional

from langchain_groq import ChatGroq

from api.groq_client import call_groq

# Intent classification uses a gpt-oss model because strict json_schema
# response format is only supported by gpt-oss on Groq (llama models 400 on
# it). gpt-oss-20b is the small/fast option and guarantees a valid label.
INTENT_MODEL = "openai/gpt-oss-20b"
GENERATION_MODEL = "llama-3.3-70b-versatile"
# Structured list outputs (practice, follow-ups) also need strict
# json_schema, so they use a gpt-oss model rather than the llama generator.
STRUCTURED_MODEL = "openai/gpt-oss-20b"

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


_STEP_BY_STEP_PROMPT = """You are a patient tutor. Using ONLY the context below, solve the student's problem as a clear numbered list of steps, showing the reasoning and any calculation at each step, then state the final answer on its own line.

Context:
{context}

Problem: {question}"""

_PRACTICE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_PRACTICE_PROMPT = """You are a tutor. Using ONLY the context below, write 3 open-ended practice questions (no options, no answers) that let a student practise this topic. Base every question on the context; do not invent facts.

Context:
{context}

Topic / request: {question}

Return JSON with a "questions" array of strings."""


def generate_step_by_step(question: str, context: str) -> str:
    prompt = _STEP_BY_STEP_PROMPT.format(context=context, question=question)
    return call_groq(prompt, model=GENERATION_MODEL)


def generate_practice(
    question: str, context: str, api_key: Optional[str] = None
) -> list[str]:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "practice", "strict": True, "schema": _PRACTICE_SCHEMA},
    }
    message = _groq(STRUCTURED_MODEL, api_key, response_format).invoke(
        _PRACTICE_PROMPT.format(context=context, question=question)
    )
    return json.loads(message.content)["questions"]


_FOLLOW_UP_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_FOLLOW_UP_PROMPT = """A student asked: "{question}"

Based on the context below, suggest exactly 3 natural follow-up questions the student might ask next to deepen their understanding. Keep them short.

Context:
{context}

Return JSON with a "questions" array of 3 strings."""


def generate_follow_ups(
    question: str, context: str, api_key: Optional[str] = None
) -> list[str]:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "followups", "strict": True, "schema": _FOLLOW_UP_SCHEMA},
    }
    message = _groq(STRUCTURED_MODEL, api_key, response_format).invoke(
        _FOLLOW_UP_PROMPT.format(context=context, question=question)
    )
    return json.loads(message.content)["questions"]
