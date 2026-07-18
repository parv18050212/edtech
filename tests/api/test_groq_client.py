from api.groq_client import call_groq, call_groq_json_schema


def test_call_groq_returns_text_response():
    answer = call_groq(
        "Reply with exactly one word: hello",
        model="llama-3.3-70b-versatile",
    )
    assert "hello" in answer.lower()


def test_call_groq_json_schema_returns_matching_shape():
    schema = {
        "type": "object",
        "properties": {
            "greeting": {"type": "string"},
        },
        "required": ["greeting"],
        "additionalProperties": False,
    }
    result = call_groq_json_schema(
        "Return a JSON object with a 'greeting' field containing the word hello.",
        model="openai/gpt-oss-120b",
        schema=schema,
        schema_name="greeting_response",
    )
    assert "greeting" in result
    assert isinstance(result["greeting"], str)
