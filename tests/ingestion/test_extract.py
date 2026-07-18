from pathlib import Path

from ingestion.extract import extract_pages

EVS_PDF = str(Path("data/raw/Class 5 EVS.pdf"))
SCIENCE_PDF = str(Path("data/raw/Science Class 8.pdf"))


def test_extract_pages_inserts_page_markers():
    text = extract_pages(EVS_PDF, 10, 10)
    assert "[[PAGE:10]]" in text


def test_extract_pages_chapter_heading_appears_before_body_text():
    # The chapter heading sits at the top of the page (smallest y-coordinate),
    # so a reading-order-aware sort must place it before the body paragraph
    # that visually starts partway down the page.
    text = extract_pages(EVS_PDF, 10, 10)
    heading_pos = text.find("Our Earth and Our Solar System")
    body_pos = text.find("When we look up from an open ground")
    assert heading_pos != -1
    assert body_pos != -1
    assert heading_pos < body_pos


def test_extract_pages_wraps_bold_topic_labels():
    text = extract_pages(EVS_PDF, 10, 12)
    assert "**Stars :**" in text


def test_extract_pages_wraps_bold_numbered_topic_in_science_book():
    text = extract_pages(SCIENCE_PDF, 10, 10)
    assert "**1.1 Five Kingdom system of classification**" in text


def test_extract_pages_wraps_bold_block_markers():
    text = extract_pages(SCIENCE_PDF, 10, 12)
    assert "**Can you recall?**" in text
    assert "**Try this**" in text
