from ingestion.db import row_from_record


def test_row_from_record_builds_tuple_with_vector_literal():
    d = {
        "chunk_id": "MSB_EVS5_CH01_TOP00_000",
        "board": "Maharashtra State Board",
        "class": "5",
        "subject": "Environmental Studies",
        "book_title": "Environmental Studies (Part One), Standard Five",
        "chapter_number": 1,
        "chapter_name": "Our Earth and Our Solar System",
        "topic": "Stars",
        "chunk_type": "paragraph",
        "page_start": 10,
        "page_end": 10,
        "chunk_text": "The sun is a star.",
        "embedding": [0.1, 0.2, 0.3],
    }
    row = row_from_record(d)
    assert row == (
        "MSB_EVS5_CH01_TOP00_000",
        "Maharashtra State Board",
        "5",
        "Environmental Studies",
        "Environmental Studies (Part One), Standard Five",
        1,
        "Our Earth and Our Solar System",
        "Stars",
        "paragraph",
        10,
        10,
        "The sun is a star.",
        "[0.1,0.2,0.3]",
    )


def test_row_from_record_handles_null_topic():
    d = {
        "chunk_id": "MSB_SCI8_CH01_TOP00_000",
        "board": "Maharashtra State Board",
        "class": "8",
        "subject": "Science",
        "book_title": "General Science, Standard Eight",
        "chapter_number": 1,
        "chapter_name": "Living World and Classification of Microbes",
        "topic": None,
        "chunk_type": "exercise",
        "page_start": 14,
        "page_end": 14,
        "chunk_text": "What is the hierarchy for classification?",
        "embedding": [0.5, 0.5],
    }
    row = row_from_record(d)
    assert row[7] is None
