from pydantic import BaseModel, ConfigDict, Field


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


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    difficulty: str
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
