from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_progress_requires_authorization_header():
    response = client.get("/progress/00000000-0000-0000-0000-000000000000")
    assert response.status_code in (401, 422)


def test_progress_rejects_invalid_token():
    response = client.get(
        "/progress/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401
