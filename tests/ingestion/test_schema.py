import json

from ingestion.schema import ChunkRecord


def test_to_dict_uses_class_as_json_key():
    record = ChunkRecord(
        chunk_id="MSB_EVS5_CH01_TOP00_000",
        board="Maharashtra State Board",
        class_="5",
        subject="Environmental Studies",
        book_title="Environmental Studies (Part One), Standard Five",
        chapter_number=1,
        chapter_name="Our Earth and Our Solar System",
        topic="Stars",
        chunk_type="paragraph",
        page_start=10,
        page_end=10,
        chunk_text="The sun is a star.",
    )
    d = record.to_dict()
    assert d["class"] == "5"
    assert "class_" not in d
    assert d["chunk_id"] == "MSB_EVS5_CH01_TOP00_000"
    assert d["topic"] == "Stars"
    # must be JSON-serializable
    assert json.loads(json.dumps(d))["chunk_text"] == "The sun is a star."


def test_to_dict_allows_null_topic():
    record = ChunkRecord(
        chunk_id="MSB_SCI8_CH01_TOP00_000",
        board="Maharashtra State Board",
        class_="8",
        subject="Science",
        book_title="General Science, Standard Eight",
        chapter_number=1,
        chapter_name="Living World and Classification of Microbes",
        topic=None,
        chunk_type="exercise",
        page_start=14,
        page_end=14,
        chunk_text="What is the hierarchy for classification of living organisms?",
    )
    d = record.to_dict()
    assert d["topic"] is None


def test_to_dict_includes_embedding_when_set():
    record = ChunkRecord(
        chunk_id="MSB_EVS5_CH01_TOP00_000",
        board="Maharashtra State Board",
        class_="5",
        subject="Environmental Studies",
        book_title="Environmental Studies (Part One), Standard Five",
        chapter_number=1,
        chapter_name="Our Earth and Our Solar System",
        topic="Stars",
        chunk_type="paragraph",
        page_start=10,
        page_end=10,
        chunk_text="The sun is a star.",
    )
    record.embedding = [0.1, 0.2, 0.3]
    d = record.to_dict()
    assert d["embedding"] == [0.1, 0.2, 0.3]


def test_to_dict_embedding_defaults_to_none():
    record = ChunkRecord(
        chunk_id="MSB_EVS5_CH01_TOP00_001",
        board="Maharashtra State Board",
        class_="5",
        subject="Environmental Studies",
        book_title="Environmental Studies (Part One), Standard Five",
        chapter_number=1,
        chapter_name="Our Earth and Our Solar System",
        topic="Stars",
        chunk_type="paragraph",
        page_start=10,
        page_end=10,
        chunk_text="Another chunk.",
    )
    assert record.to_dict()["embedding"] is None
