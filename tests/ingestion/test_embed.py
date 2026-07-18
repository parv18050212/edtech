from ingestion.embed import build_document_prompt, embed_texts


def test_build_document_prompt_with_title():
    assert (
        build_document_prompt("The sun is a star.", "Stars")
        == "title: Stars | text: The sun is a star."
    )


def test_build_document_prompt_without_title():
    assert (
        build_document_prompt("Some content.", None)
        == "title: none | text: Some content."
    )


def test_embed_texts_returns_one_768_dim_vector_per_input():
    embeddings = embed_texts(
        ["The sun is a star.", "The earth revolves around the sun."],
        titles=["Stars", "Planets"],
    )
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    assert len(embeddings[1]) == 768


def test_embed_texts_produces_different_vectors_for_different_text():
    embeddings = embed_texts(
        ["The sun is a star.", "Completely unrelated content about rocks."],
        titles=["Stars", "Rocks"],
    )
    assert embeddings[0] != embeddings[1]


def test_embed_texts_handles_none_title():
    embeddings = embed_texts(["Some content with no topic."], titles=[None])
    assert len(embeddings[0]) == 768
