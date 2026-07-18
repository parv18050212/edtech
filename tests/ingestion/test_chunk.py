from ingestion.chunk import (
    BookMeta,
    build_chunk_records,
    estimate_tokens,
    finalize_chunk_text,
    make_chunk_id,
    split_paragraph,
)
from ingestion.structure import Block, TopicSegment


def test_estimate_tokens_uses_word_based_heuristic():
    text = " ".join(["word"] * 75)
    assert estimate_tokens(text) == 100


def test_estimate_tokens_empty_text_is_zero():
    assert estimate_tokens("") == 0


def test_finalize_chunk_text_strips_artifacts():
    raw = "  [[PAGE:10]] Some **bold** text   with   extra space [[PAGE:11]] "
    assert finalize_chunk_text(raw) == "Some bold text with extra space"


def test_split_paragraph_keeps_sentences_intact_and_targets_token_count():
    sentence = "This is one short sentence with about ten words in it total. "
    text = sentence * 10
    chunks = split_paragraph(text, target_tokens=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip().endswith(".")


def test_split_paragraph_single_short_sentence_is_one_chunk():
    chunks = split_paragraph("Short sentence.", target_tokens=250)
    assert chunks == ["Short sentence."]


def test_make_chunk_id_format():
    assert make_chunk_id("EVS", "5", 1, 0, 3) == "MSB_EVS5_CH01_TOP00_003"
    assert make_chunk_id("SCI", "8", 10, 2, 15) == "MSB_SCI8_CH10_TOP02_015"


def test_build_chunk_records_splits_paragraphs_and_keeps_special_blocks_whole():
    meta = BookMeta(
        board="Maharashtra State Board",
        class_="5",
        subject="Environmental Studies",
        subject_code="EVS",
        book_title="Environmental Studies (Part One), Standard Five",
    )
    segments = [
        TopicSegment(
            topic="Stars",
            blocks=[
                Block("paragraph", "The sun is a star. It is very hot.", 10, 10),
                Block("activity", "Name two stars you can see at night.", 10, 10),
            ],
        )
    ]
    records = build_chunk_records(
        meta,
        chapter_number=1,
        chapter_name="Our Earth and Our Solar System",
        topic_segments=segments,
    )

    assert len(records) == 2
    assert records[0].chunk_type == "paragraph"
    assert records[0].topic == "Stars"
    assert records[0].chapter_number == 1
    assert records[0].class_ == "5"
    assert records[1].chunk_type == "activity"
    assert records[1].chunk_text == "Name two stars you can see at night."
    assert records[0].chunk_id == "MSB_EVS5_CH01_TOP00_000"
    assert records[1].chunk_id == "MSB_EVS5_CH01_TOP00_001"
