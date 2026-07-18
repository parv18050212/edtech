from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_chat_requires_authorization_header():
    response = client.post(
        "/chapters/1/chat",
        json={"subject": "Environmental Studies", "class": "5", "question": "What is a star?"},
    )
    assert response.status_code in (401, 422)


def test_chat_rejects_invalid_token():
    response = client.post(
        "/chapters/1/chat",
        json={"subject": "Environmental Studies", "class": "5", "question": "What is a star?"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401
