from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

QUESTION_TYPES = {"mcq", "fill_in_the_blank", "assertion_reason"}
DIFFICULTIES = {"easy", "medium", "hard", "mixed"}

QuestionType = Literal["mcq", "fill_in_the_blank", "assertion_reason"]
Difficulty = Literal["easy", "medium", "hard"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    subject: str
    class_: str = Field(alias="class")
    question: str


class ChatResponse(BaseModel):
    answer: str
    source_chunk_ids: list[str]


class QuizRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    subject: str
    class_: str = Field(alias="class")
    question_types: list[str] = Field(default_factory=lambda: ["mcq"])
    difficulty: str = "mixed"
    count: int = Field(default=5, ge=1, le=10)
    topic: Optional[str] = None

    @field_validator("question_types")
    @classmethod
    def _validate_question_types(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("question_types must not be empty")
        unknown = [t for t in value if t not in QUESTION_TYPES]
        if unknown:
            raise ValueError(
                f"unknown question type(s): {unknown}; allowed: {sorted(QUESTION_TYPES)}"
            )
        return value

    @field_validator("difficulty")
    @classmethod
    def _validate_difficulty(cls, value: str) -> str:
        if value not in DIFFICULTIES:
            raise ValueError(
                f"unknown difficulty {value!r}; allowed: {sorted(DIFFICULTIES)}"
            )
        return value


class QuizQuestion(BaseModel):
    type: QuestionType
    question: str
    options: list[str]
    correct_answer: str
    difficulty: Difficulty
    explanation: str


class QuizResponse(BaseModel):
    quiz: list[QuizQuestion]


class QuizSubmitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    subject: str
    class_: str = Field(alias="class")
    chapter_number: int
    quiz_json: dict
    student_answers: dict
