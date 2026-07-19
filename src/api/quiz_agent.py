"""Quiz generation agent.

A focused agent implemented with LangChain: it drives Groq's
`openai/gpt-oss-120b` through `ChatGroq.with_structured_output`, so the
model's reply is coerced (via tool-calling) into the `QuizResponse` schema
and returned already validated. Generates MCQ, fill-in-the-blank, and
assertion-reason questions at a requested difficulty and count from
supplied chapter/topic context.

This is a single self-contained agent; a LangGraph orchestrator can wire it
alongside other agents (chat, doubt-solver) later.
"""

import json
from typing import Optional

from langchain_groq import ChatGroq

from api.schemas import QuizResponse

QUIZ_MODEL = "openai/gpt-oss-120b"


def quiz_schema() -> dict:
    """Strict JSON schema covering all three question types.

    A single flat item shape (with a `type` discriminator) is used rather
    than a oneOf union, because Groq's strict structured-output mode is more
    reliable with a flat schema. `options` is an empty list for
    fill-in-the-blank items. Bound to ChatGroq via `response_format`.
    """
    return {
        "type": "object",
        "properties": {
            "quiz": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["mcq", "fill_in_the_blank", "assertion_reason"],
                        },
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "correct_answer": {"type": "string"},
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "type",
                        "question",
                        "options",
                        "correct_answer",
                        "difficulty",
                        "explanation",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["quiz"],
        "additionalProperties": False,
    }

# Per-type formatting rules, injected into the prompt for the types the
# caller actually requested.
_TYPE_RULES = {
    "mcq": (
        '- "mcq": a multiple-choice question with exactly 4 entries in '
        '"options"; "correct_answer" must be one of those options verbatim.'
    ),
    "fill_in_the_blank": (
        '- "fill_in_the_blank": a statement with a blank written as "____". '
        '"options" MUST be an empty list []; "correct_answer" is the exact '
        "word or phrase that fills the blank."
    ),
    "assertion_reason": (
        '- "assertion_reason": put both statements in "question" formatted as '
        '"Assertion: <A>. Reason: <R>." "options" must be exactly these 4: '
        '["Both A and R are true and R is the correct explanation of A", '
        '"Both A and R are true but R is not the correct explanation of A", '
        '"A is true but R is false", "A is false but R is true"]. '
        '"correct_answer" must be one of those 4 verbatim.'
    ),
}


def build_quiz_prompt(
    context: str,
    subject: str,
    class_: str,
    question_types: list[str],
    difficulty: str,
    count: int,
) -> str:
    type_rules = "\n".join(_TYPE_RULES[t] for t in question_types)

    if difficulty == "mixed":
        difficulty_instruction = (
            "Vary the difficulty of the questions across easy, medium, and "
            "hard; set each question's own difficulty accordingly."
        )
    else:
        difficulty_instruction = (
            f"Every question must be {difficulty} difficulty; set each "
            f'question\'s "difficulty" to "{difficulty}".'
        )

    return (
        f"You are a quiz generator for {subject} Class {class_}. "
        f"Generate exactly {count} question(s) based ONLY on the context "
        f"below.\n\n"
        f"Use only these question type(s): {', '.join(question_types)}.\n"
        f"Formatting rules per type:\n{type_rules}\n\n"
        f"{difficulty_instruction}\n\n"
        f"Every question needs a concise 'explanation' of the correct "
        f"answer. Do not invent facts beyond the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Return valid JSON matching the required schema."
    )


def _normalize(answer: str) -> str:
    return str(answer).strip().casefold()


def score_answers(
    quiz_items: list[dict], student_answers: dict
) -> tuple[int, int]:
    """Score a submission, matching answers by question text.

    Comparison is normalized (trimmed + case-folded) so free-text
    fill-in-the-blank answers match despite trivial case/whitespace
    differences. Returns (correct_count, total_questions).
    """
    total = len(quiz_items)
    correct = 0
    for item in quiz_items:
        expected = _normalize(item["correct_answer"])
        given = student_answers.get(item["question"])
        if given is not None and _normalize(given) == expected:
            correct += 1
    return correct, total


def _build_llm(api_key: Optional[str]):
    kwargs = {"model": QUIZ_MODEL, "temperature": 0}
    if api_key:
        kwargs["api_key"] = api_key
    # Bind Groq's native strict json_schema response format. This is more
    # reliable than LangChain's tool-calling / json_mode structured-output
    # methods, which gpt-oss intermittently breaks (emits a bogus 'json'
    # tool call, or drops the top-level object wrapper).
    return ChatGroq(**kwargs).bind(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "quiz", "strict": True, "schema": quiz_schema()},
        }
    )


def generate_quiz(
    context: str,
    subject: str,
    class_: str,
    question_types: list[str],
    difficulty: str,
    count: int,
    api_key: Optional[str] = None,
) -> dict:
    prompt = build_quiz_prompt(
        context, subject, class_, question_types, difficulty, count
    )
    message = _build_llm(api_key).invoke(prompt)
    # QuizResponse validates the model output; model_dump gives a plain dict
    # for the route to return and for scoring.
    return QuizResponse.model_validate(json.loads(message.content)).model_dump()
