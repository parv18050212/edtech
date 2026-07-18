# Textbook Ingestion: Cleaning & Chunking Pipeline — Design

## Context

This is the first buildable slice of the AI-Powered Chapter-wise RAG Learning
Platform described in `RAG_Learning_Platform_Study.pdf` (Sections 6-8 in
particular). That study scoped a full production architecture and a POC
roadmap; this design covers only the **document ingestion pipeline's
cleaning and chunking stages** — turning the two source textbook PDFs into
structured, chunk-ready JSONL records. Embedding generation and pgvector
storage are explicitly out of scope for this slice and will be designed
separately once chunk output quality is validated.

Two source textbooks are in scope, both Maharashtra State Board (Balbharati)
textbooks, not CBSE as the reference study assumed:

- `Class 5 EVS.pdf` — Environmental Studies (Part One), Standard Five, 143
  pages, 25 chapters.
- `Science Class 8.pdf` — General Science, Standard Eight, 145 pages, 19
  chapters.

Both are digitally created PDFs with cleanly extractable text — no OCR is
required. However, raw extraction has a real reading-order problem: these
textbooks use multi-column layouts with embedded images, captions, and
sidebar boxes ("Try this", "Can you tell?"), and a naive linear text dump
(confirmed via `pdftotext`) interleaves these elements mid-paragraph,
producing garbled chunks if not corrected.

## Scope of this slice

Produce clean, hierarchically-chunked, metadata-tagged text for both
textbooks, as JSONL records ready for a later embedding step. This includes:

- PDF text extraction with correct reading order.
- Removal of extraction noise (page numbers, footnotes, repeated
  boilerplate).
- Structural detection: chapter boundaries, topic headings, and special
  block types (activities, recall questions, info boxes, exercises).
- Hierarchical chunking with token-size targets, never splitting mid-block.
- A pilot run against a subset of chapters, manually reviewed for noise and
  structural accuracy, before running across all 44 chapters.

Explicitly **out of scope**: embedding generation, pgvector schema/loading,
OCR (not needed — both PDFs are digitally created), the LMS/API/quiz layers
described elsewhere in the reference study.

## Extraction approach

Extraction uses **PyMuPDF (`fitz`)**, reading text as positioned blocks
(with bounding boxes) per page, then sorting blocks by column and vertical
position to reconstruct correct reading order. This directly targets the
jumbling problem found during investigation — image captions and sidebar
boxes breaking mid-paragraph under naive linear extraction. This also
matches the source study's own recommendation (Section 7: "PyMuPDF or
pdfplumber for text extraction from digital PDFs").

Cleaning, applied per page after extraction:

- Strip isolated numeric lines (page numbers).
- Strip repeated header/footer boilerplate (e.g. publisher notices that
  repeat verbatim across pages).
- Normalize whitespace and hyphenation artifacts from line-wrapped text.

## Structural detection & chunking strategy

**Chapter detection**: Both books have a Contents page listing exact chapter
numbers, titles, and order — already extracted as ground truth (25 entries
for EVS, 19 for Science). Extracted in-body headings are matched against
this table of contents rather than inferred from regex alone.

**Topic detection**: Two patterns are detected, since the two books use
different conventions:

1. Numbered subheadings (e.g. "1.1 Five Kingdom system of classification"),
   present in Science Class 8.
2. Bold, colon-terminated label headers (e.g. "Stars :", "Gravity",
   "Dwarf planets :") followed by their own explanatory paragraph — the
   pattern observed in EVS Class 5, which does not number its subtopics.

Both patterns feed the same `topic` field. Where neither pattern is
detected, `topic` is left null rather than duplicating the chapter name.
The pilot run specifically validates how reliably pattern 2 (the heuristic
one) is detected, since it is more error-prone than the numbered form.

**Block-type detection**: A fixed marker-phrase list identifies recurring
special sections common to both books, mapped to a small `chunk_type` set:

| chunk_type | Marker phrases |
|---|---|
| `activity` | "Try this", "Let's try this", "Use your brain power!", "Can you tell?" |
| `recall` | "Can you recall?", "Let's recall" |
| `info_box` | "Do you know?", "Research", "Find out", "In History......", "Always remember" |
| `exercise` | End-of-chapter question sections ("Fill in the blanks", "Answer the following questions", "Answer in one sentence", "True or false?", "What's the solution?") |
| `paragraph` | Remaining narrative prose |

Remaining prose is split into paragraph chunks targeting ~250 tokens,
splitting only at sentence boundaries. Token count is estimated with a
simple word-based heuristic (word_count / 0.75), consistent with the
reference study's own definition of a token as "roughly three-quarters of a
word" — no external tokenizer dependency is needed for this estimate.
Formula-specific chunking (Section 8 of the reference study) is not built
in this slice — these are general-science chapters, not formula-heavy, and
any formulas that appear stay inside their surrounding paragraph chunk.
This can be revisited if the pilot shows it matters.

## Data model / output

One JSONL file per book, one record per chunk:

```json
{
  "chunk_id": "MSB_EVS5_CH01_TOP00_003",
  "board": "Maharashtra State Board",
  "class": "5",
  "subject": "Environmental Studies",
  "book_title": "Environmental Studies (Part One), Standard Five",
  "chapter_number": 1,
  "chapter_name": "Our Earth and Our Solar System",
  "topic": null,
  "chunk_type": "paragraph",
  "page_start": 9,
  "page_end": 9,
  "chunk_text": "..."
}
```

`board`, `class`, and `subject` are set per-book from known values
(Maharashtra State Board; class 5 or 8; Environmental Studies or Science),
not inferred from PDF content.

## Project structure

```
edtech/
  data/raw/              # the two source PDFs (moved here)
  data/interim/           # pilot output for inspection
  src/ingestion/
    extract.py            # PyMuPDF block extraction + reading-order reconstruction
    clean.py              # boilerplate/page-number/footer stripping
    structure.py          # chapter/topic/block-type detection
    chunk.py               # hierarchical chunking + token sizing
    schema.py               # chunk record dataclass/Pydantic model
  requirements.txt          # pymupdf, pytest
  .gitignore
```

Git is initialized at `D:\Coding\edtech`, with a Python virtual environment
for dependency isolation.

## Pilot plan

Before running the pipeline across all 44 chapters, run it against **2
chapters per book**:

- EVS Chapter 1 ("Our Earth and Our Solar System") — narrative-heavy, tests
  the bold-label topic heuristic.
- Science Chapter 1 ("Living World and Classification of Microbes") — the
  most visually complex chapter observed during investigation (dense with
  diagrams and sidebar boxes), stress-tests reading-order reconstruction.
- One additional chapter per book, chosen after reviewing the first pair.

Output goes to `data/interim/` as JSONL and is reviewed chunk-by-chunk for:

- Correct reading order (no interleaved captions/sidebar text mid-sentence).
- No stray page numbers or footnote text leaking into `chunk_text`.
- Correct `chunk_type` classification against the marker-phrase table above.
- Topic detection accuracy, particularly the EVS bold-label heuristic.

Only after this review passes does the pipeline run across all chapters of
both books.

## Testing strategy

- Unit tests for the chapter/topic/block-type detectors against known
  extracted text samples (including the exact jumbled-order and bold-label
  cases found during investigation).
- Unit tests confirming chunks are never split mid-sentence and that
  `activity`/`exercise`/`recall`/`info_box` blocks are never split across
  chunk boundaries.
- The pilot review itself serves as the end-to-end validation for this
  slice, standing in for the full-platform end-to-end test described in the
  reference study (which requires the LMS/quiz layers not yet built).
