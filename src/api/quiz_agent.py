"""Quiz generation agent.

A focused "agent" in the POC sense from the architecture study: a distinct
prompt template + strict output schema encapsulated as its own testable
unit, driving quiz generation on Groq's structured-output model. Generates
MCQ, fill-in-the-blank, and assertion-reason questions at a requested
difficulty and count from supplied chapter/topic context.
"""

from typing import Optional

from api.groq_client import call_groq_json_schema

QUIZ_MODEL = "openai/gpt-oss-120b"

QUESTION_TYPES = {"mcq", "fill_in_the_blank", "assertion_reason"}
DIFFICULTIES = {"easy", "medium", "hard", "mixed"}

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


def quiz_schema() -> dict:
    """Unified strict schema covering all three question types.

    A single flat item shape (with a `type` discriminator) is used rather
    than a oneOf union, because Groq's strict structured-output mode is more
    reliable with a flat schema. `options` is an empty list for
    fill-in-the-blank items.
    """
    return {
        "type": "object",
        "properties": {
            "quiz": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": sorted(QUESTION_TYPES)},
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
    return call_groq_json_schema(
        prompt,
        model=QUIZ_MODEL,
        schema=quiz_schema(),
        schema_name="quiz",
        api_key=api_key,
    )
