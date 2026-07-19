import api.groq_client as groq_client
from api.groq_client import call_groq


class _FakeResponse:
    def __init__(self, status_code, json_body=None, headers=None):
        self.status_code = status_code
        self._json = json_body or {}
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)


def test_call_groq_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(429, headers={"retry-after": "0"})
        return _FakeResponse(
            200, {"choices": [{"message": {"content": "recovered"}}]}
        )

    monkeypatch.setattr(groq_client.requests, "post", fake_post)
    # avoid real sleeping in the test
    monkeypatch.setattr(groq_client.time, "sleep", lambda s: None)

    answer = call_groq("hi", model="llama-3.3-70b-versatile", api_key="test-key")
    assert answer == "recovered"
    assert calls["n"] == 2


def test_call_groq_gives_up_after_max_retries(monkeypatch):
    import requests

    def always_429(url, headers=None, json=None, timeout=None):
        return _FakeResponse(429, headers={"retry-after": "0"})

    monkeypatch.setattr(groq_client.requests, "post", always_429)
    monkeypatch.setattr(groq_client.time, "sleep", lambda s: None)

    try:
        call_groq("hi", model="llama-3.3-70b-versatile", api_key="test-key")
        assert False, "expected HTTPError after exhausting retries"
    except requests.exceptions.HTTPError:
        pass


def test_call_groq_returns_text_response():
    # Deterministic factual prompt -> robust assertion (avoids flaky
    # exact-word matching on a non-deterministic model).
    answer = call_groq(
        "What is 2 + 2? Reply with just the number.",
        model="llama-3.3-70b-versatile",
    )
    assert isinstance(answer, str) and answer.strip()
    assert "4" in answer
