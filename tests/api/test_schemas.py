from api.schemas import ChatRequest, QuizRequest


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
