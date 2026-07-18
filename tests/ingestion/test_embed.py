from ingestion.embed import (
    build_document_prompt,
    build_query_prompt,
    embed_prompts,
    embed_query,
    embed_texts,
)


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


def test_build_query_prompt():
    assert (
        build_query_prompt("What is a star?")
        == "task: search result | query: What is a star?"
    )


def test_embed_prompts_returns_768_dim_vectors():
    embeddings = embed_prompts(["title: none | text: The sun is a star."])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768


def test_embed_query_returns_single_768_dim_vector():
    embedding = embed_query("What is a star?")
    assert len(embedding) == 768


def test_embed_texts_still_works_after_refactor():
    # Regression guard: embed_texts's existing public behavior (used by
    # scripts/run_full_pipeline.py) must be unchanged by factoring out
    # embed_prompts.
    embeddings = embed_texts(["The sun is a star."], titles=["Stars"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768
