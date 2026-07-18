import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingestion.chunk import BookMeta, build_chunk_records
from ingestion.clean import clean_text
from ingestion.extract import extract_pages
from ingestion.structure import parse_contents, segment_chapter, split_into_chapters

BOILERPLATE_PHRASES = [
    "Maharashtra State Bureau of Textbook Production and Curriculum Research, Pune.",
    "and Curriculum Research, Pune - 411 004.",
]

TOC_PAGE = 9
CHAPTER_1_PAGE_RANGE = (10, 14)  # 0-indexed, covers chapter 1 fully in both books

BOOKS = [
    {
        "pdf_path": "data/raw/Class 5 EVS.pdf",
        "out_name": "EVS_5_ch01.jsonl",
        "meta": BookMeta(
            board="Maharashtra State Board",
            class_="5",
            subject="Environmental Studies",
            subject_code="EVS",
            book_title="Environmental Studies (Part One), Standard Five",
        ),
    },
    {
        "pdf_path": "data/raw/Science Class 8.pdf",
        "out_name": "SCI_8_ch01.jsonl",
        "meta": BookMeta(
            board="Maharashtra State Board",
            class_="8",
            subject="Science",
            subject_code="SCI",
            book_title="General Science, Standard Eight",
        ),
    },
]

VALID_CHUNK_TYPES = {"paragraph", "activity", "recall", "info_box", "exercise"}


def run_book(book: dict, out_dir: Path) -> list:
    toc_text = extract_pages(book["pdf_path"], TOC_PAGE, TOC_PAGE)
    toc = parse_contents(toc_text)

    chapter_number = 1
    first, last = CHAPTER_1_PAGE_RANGE
    raw = extract_pages(book["pdf_path"], first, last)
    cleaned = clean_text(raw, BOILERPLATE_PHRASES)

    # Only the extracted page range's own chapter heading is present in
    # `cleaned` (pages 10-14 cover chapter 1 only, not chapter 2's heading
    # on page 15), so split_into_chapters must only be asked to find that
    # one heading -- passing the full toc would fail on every other chapter.
    chapter_toc = [t for t in toc if t.chapter_number == chapter_number]
    chapters = split_into_chapters(cleaned, chapter_toc)

    chapter_name = chapter_toc[0].chapter_name
    chapter_text = chapters[chapter_number]
    segments = segment_chapter(chapter_text)
    records = build_chunk_records(
        book["meta"], chapter_number, chapter_name, segments
    )

    out_path = out_dir / book["out_name"]
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    return records


def main():
    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)

    for book in BOOKS:
        records = run_book(book, out_dir)
        assert len(records) > 0, f"No chunks produced for {book['pdf_path']}"
        for record in records:
            assert record.chunk_type in VALID_CHUNK_TYPES, (
                f"Invalid chunk_type {record.chunk_type!r} in {record.chunk_id}"
            )
            assert "[[PAGE:" not in record.chunk_text, (
                f"Leaked page marker in {record.chunk_id}"
            )
            assert "**" not in record.chunk_text, (
                f"Leaked bold marker in {record.chunk_id}"
            )
        print(
            f"{book['pdf_path']}: wrote {len(records)} chunks to "
            f"{out_dir / book['out_name']}"
        )


if __name__ == "__main__":
    main()
