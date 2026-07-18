from pathlib import Path

from ingestion.clean import clean_text
from ingestion.extract import extract_pages
from ingestion.structure import parse_contents, split_into_chapters

EVS_PDF = str(Path("data/raw/Class 5 EVS.pdf"))
SCIENCE_PDF = str(Path("data/raw/Science Class 8.pdf"))

TOC_PAGE = 9
BODY_FIRST_PAGE = 10
BODY_LAST_PAGE = 147


def _load_chapters(pdf_path: str):
    toc_text = extract_pages(pdf_path, TOC_PAGE, TOC_PAGE)
    toc = parse_contents(toc_text)
    raw = extract_pages(pdf_path, BODY_FIRST_PAGE, BODY_LAST_PAGE)
    cleaned = clean_text(raw, boilerplate_phrases=[])
    chapters = split_into_chapters(cleaned, toc)
    return toc, chapters


def test_evs_full_book_splits_into_all_25_chapters():
    toc, chapters = _load_chapters(EVS_PDF)
    assert len(toc) == 25
    assert set(chapters.keys()) == {t.chapter_number for t in toc}
    for chapter_number, text in chapters.items():
        assert text.strip(), f"Chapter {chapter_number} body is empty"


def test_science_full_book_splits_into_all_19_chapters():
    toc, chapters = _load_chapters(SCIENCE_PDF)
    assert len(toc) == 19
    assert set(chapters.keys()) == {t.chapter_number for t in toc}
    for chapter_number, text in chapters.items():
        assert text.strip(), f"Chapter {chapter_number} body is empty"
