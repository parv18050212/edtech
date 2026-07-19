import pytest
from pydantic import ValidationError

from api.schemas import ChatRequest, QuizQuestion, QuizRequest


def test_chat_request_accepts_class_as_wire_field_name():
    request = ChatRequest.model_validate(
        {"subject": "Science", "class": "8", "question": "What is a star?"}
    )
    assert request.class_ == "8"
    assert request.subject == "Science"
    assert request.question == "What is a star?"


def test_quiz_request_accepts_class_as_wire_field_name():
    request = QuizRequest.model_validate({"subject": "Science", "class": "8"})
    assert request.class_ == "8"


def test_quiz_request_defaults():
    request = QuizRequest.model_validate({"subject": "Science", "class": "8"})
    assert request.question_types == ["mcq"]
    assert request.difficulty == "mixed"
    assert request.count == 5
    assert request.topic is None


def test_quiz_request_accepts_all_params():
    request = QuizRequest.model_validate(
        {
            "subject": "Science",
            "class": "8",
            "question_types": ["mcq", "fill_in_the_blank", "assertion_reason"],
            "difficulty": "hard",
            "count": 8,
            "topic": "photosynthesis",
        }
    )
    assert request.question_types == ["mcq", "fill_in_the_blank", "assertion_reason"]
    assert request.difficulty == "hard"
    assert request.count == 8
    assert request.topic == "photosynthesis"


def test_quiz_request_rejects_unknown_question_type():
    with pytest.raises(ValidationError):
        QuizRequest.model_validate(
            {"subject": "Science", "class": "8", "question_types": ["essay"]}
        )


def test_quiz_request_rejects_empty_question_types():
    with pytest.raises(ValidationError):
        QuizRequest.model_validate(
            {"subject": "Science", "class": "8", "question_types": []}
        )


def test_quiz_request_rejects_bad_difficulty():
    with pytest.raises(ValidationError):
        QuizRequest.model_validate(
            {"subject": "Science", "class": "8", "difficulty": "impossible"}
        )


def test_quiz_request_rejects_out_of_range_count():
    with pytest.raises(ValidationError):
        QuizRequest.model_validate(
            {"subject": "Science", "class": "8", "count": 0}
        )
    with pytest.raises(ValidationError):
        QuizRequest.model_validate(
            {"subject": "Science", "class": "8", "count": 11}
        )


def test_quiz_question_has_type_field_and_allows_empty_options():
    q = QuizQuestion.model_validate(
        {
            "type": "fill_in_the_blank",
            "question": "Water boils at ____ degrees Celsius.",
            "options": [],
            "correct_answer": "100",
            "difficulty": "easy",
            "explanation": "At sea level water boils at 100C.",
        }
    )
    assert q.type == "fill_in_the_blank"
    assert q.options == []
