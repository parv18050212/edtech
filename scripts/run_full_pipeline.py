import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingestion.chunk import BookMeta, build_chunk_records
from ingestion.clean import clean_text
from ingestion.embed import embed_texts
from ingestion.extract import extract_pages
from ingestion.structure import parse_contents, segment_chapter, split_into_chapters

BOILERPLATE_PHRASES = [
    "Maharashtra State Bureau of Textbook Production and Curriculum Research, Pune.",
    "and Curriculum Research, Pune - 411 004.",
]

TOC_PAGE = 9
BODY_PAGE_RANGE = (10, 147)

BOOKS = [
    {
        "pdf_path": "data/raw/Class 5 EVS.pdf",
        "out_name": "EVS_5_full.jsonl",
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
        "out_name": "SCI_8_full.jsonl",
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


def run_book(book: dict, out_dir: Path) -> int:
    toc_text = extract_pages(book["pdf_path"], TOC_PAGE, TOC_PAGE)
    toc = parse_contents(toc_text)

    first, last = BODY_PAGE_RANGE
    raw = extract_pages(book["pdf_path"], first, last)
    cleaned = clean_text(raw, BOILERPLATE_PHRASES)
    chapters = split_into_chapters(cleaned, toc)

    out_path = out_dir / book["out_name"]
    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for entry in toc:
            chapter_text = chapters[entry.chapter_number]
            segments = segment_chapter(chapter_text)
            records = build_chunk_records(
                book["meta"], entry.chapter_number, entry.chapter_name, segments
            )
            if not records:
                continue

            titles = [r.topic or r.chapter_name for r in records]
            texts = [r.chunk_text for r in records]
            embeddings = embed_texts(texts, titles)

            for record, embedding in zip(records, embeddings):
                assert record.chunk_type in VALID_CHUNK_TYPES, (
                    f"Invalid chunk_type {record.chunk_type!r} in {record.chunk_id}"
                )
                assert len(embedding) == 768, (
                    f"Unexpected embedding size {len(embedding)} for {record.chunk_id}"
                )
                record.embedding = embedding
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                total += 1

            print(
                f"  chapter {entry.chapter_number} "
                f"({entry.chapter_name}): {len(records)} chunks"
            )

    return total


def main():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for book in BOOKS:
        print(f"Processing {book['pdf_path']}...")
        total = run_book(book, out_dir)
        print(f"{book['pdf_path']}: wrote {total} chunks to {out_dir / book['out_name']}")


if __name__ == "__main__":
    main()
