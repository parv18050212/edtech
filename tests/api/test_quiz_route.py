from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_quiz_requires_authorization_header():
    response = client.post(
        "/chapters/1/quiz",
        json={"subject": "Environmental Studies", "class": "5"},
    )
    assert response.status_code in (401, 422)


def test_quiz_submit_requires_authorization_header():
    response = client.post(
        "/quiz/submit",
        json={
            "subject": "Environmental Studies",
            "class": "5",
            "chapter_number": 1,
            "quiz_json": {"quiz": []},
            "student_answers": {},
        },
    )
    assert response.status_code in (401, 422)
