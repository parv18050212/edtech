from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_list_books_returns_both_seeded_books():
    response = client.get("/books")
    assert response.status_code == 200
    books = response.json()
    keys = {(b["subject"], b["class"]) for b in books}
    assert ("Environmental Studies", "5") in keys
    assert ("Science", "8") in keys
    for b in books:
        assert b["board"] == "Maharashtra State Board"
        assert b["book_title"]


def test_list_chapters_returns_ordered_chapters_for_a_book():
    response = client.get("/chapters", params={"subject": "Science", "class": "8"})
    assert response.status_code == 200
    chapters = response.json()
    assert len(chapters) == 19
    numbers = [c["chapter_number"] for c in chapters]
    assert numbers == sorted(numbers)
    assert numbers[0] == 1
    assert all(c["chapter_name"] for c in chapters)


def test_list_chapters_for_evs_returns_all_25():
    response = client.get(
        "/chapters", params={"subject": "Environmental Studies", "class": "5"}
    )
    assert response.status_code == 200
    assert len(response.json()) == 25


def test_catalog_is_public_no_auth_required():
    # No Authorization header at all -> still 200, unlike the chat/quiz routes.
    assert client.get("/books").status_code == 200


def test_cors_headers_present_on_catalog_response():
    response = client.get("/books", headers={"Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-allow-origin") == "*"
