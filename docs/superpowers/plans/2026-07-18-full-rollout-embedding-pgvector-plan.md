# Full Rollout, Embedding, and pgvector Load — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing ingestion pipeline across all chapters of both textbooks, embed every chunk with `embeddinggemma` via local Ollama, and load the result into a pgvector-backed `chunks` table in Supabase, per `docs/superpowers/specs/2026-07-18-full-rollout-embedding-pgvector-design.md`.

**Architecture:** `extract.py`/`clean.py`/`structure.py`/`chunk.py` are reused unchanged. TOC parsing (page 9) and body extraction (pages 10-147) stay as separate calls to avoid the TOC-page/body-heading regex collision described in the design. A new `embed.py` batches chunk texts per chapter through Ollama's `/api/embed` endpoint using the model's documented `"title: ... | text: ..."` prompt format. A new `db.py` builds pgvector-ready row tuples; `scripts/run_full_pipeline.py` and `scripts/load_to_supabase.py` are thin orchestration wrappers, consistent with `scripts/run_pilot.py`.

**Tech Stack:** Python 3.12, PyMuPDF, requests, psycopg2-binary, python-dotenv, pytest. Ollama 0.32.1 (already installed) running `embeddinggemma` locally. Supabase Postgres 17.6 with pgvector 0.8.2.

## Global Constraints

- Ollama server must be running locally at `http://localhost:11434` with the `embeddinggemma` model pulled (already done: `ollama pull embeddinggemma`, confirmed 621 MB, returns 768-dim vectors).
- Document encoding prompt format (from the model's Hugging Face card): `"title: {title | "none"} | text: {content}"`.
- Both books have 148 pages total (0-indexed 0-147); table of contents is on 0-indexed page 9 in both; chapter body content spans 0-indexed pages 10-147 in both.
- EVS has 25 chapters, Science has 19 chapters (confirmed from each book's table of contents).
- Supabase project id: `aedsgktxvmddarqurbhi`. `vector` extension available but not yet installed at the start of this plan.
- `DATABASE_URL` is read from a `.env` file via `python-dotenv`; the file itself is gitignored and never printed, logged, or committed. Only `.env.example` (with a placeholder) is committed.
- New pinned dependencies (versions confirmed installed in this environment): `requests==2.34.2`, `psycopg2-binary==2.9.12`, `python-dotenv==1.2.2`.
- `class` is the JSON/DB column name for what `ChunkRecord` stores as `class_` in Python — this translation already happens in `ChunkRecord.to_dict()` (existing code, unchanged).

---

## File Structure

```
edtech/
  src/ingestion/
    schema.py          # MODIFY: add `embedding` field to ChunkRecord
    embed.py            # CREATE: build_document_prompt, embed_texts (Ollama client)
    db.py                # CREATE: row_from_record, INSERT_SQL, INSERT_TEMPLATE
  scripts/
    run_full_pipeline.py  # CREATE: rollout + embedding orchestration, all chapters, both books
    load_to_supabase.py    # CREATE: psycopg2 bulk loader
  supabase/
    migrations/
      0001_create_chunks_table.sql  # CREATE: pgvector extension + chunks table + index
  tests/ingestion/
    test_schema.py       # MODIFY: embedding field round-trips through to_dict()
    test_embed.py         # CREATE: build_document_prompt (pure) + embed_texts (real Ollama)
    test_db.py             # CREATE: row_from_record (pure, no DB)
    test_full_book_extraction.py  # CREATE: chapter-count regression guard, both books
  data/
    processed/            # CREATE: final JSONL output (gitignored)
      .gitkeep
  requirements.txt          # MODIFY: add requests, psycopg2-binary, python-dotenv
  .gitignore                 # MODIFY: add .env, data/processed/
  .env.example                # CREATE: documents DATABASE_URL with a placeholder
  .env                         # CREATE locally by the user, never committed
```

---

### Task 1: Dependencies and environment scaffolding

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `.env.example`
- Create: `data/processed/.gitkeep`

**Interfaces:**
- Produces: `requests`, `psycopg2-binary`, and `python-dotenv` installed in `.venv` and available to every later task; `data/processed/` exists as the output directory for Task 5.

- [x] **Step 1: Add the new dependencies to `requirements.txt`**

```
pymupdf==1.28.0
pytest==8.3.3
requests==2.34.2
psycopg2-binary==2.9.12
python-dotenv==1.2.2
```

- [x] **Step 2: Install and verify**

```bash
"D:/Coding/edtech/.venv/Scripts/pip" install -r requirements.txt
"D:/Coding/edtech/.venv/Scripts/python" -c "import requests, psycopg2, dotenv; print('ok')"
```

Expected: prints `ok` with no import errors.

- [x] **Step 3: Create `.env.example`**

```
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.aedsgktxvmddarqurbhi.supabase.co:5432/postgres
```

- [x] **Step 4: Add `.env` and `data/processed/` to `.gitignore`**

Append to the existing `.gitignore`:

```
.env
data/processed/
```

- [x] **Step 5: Create `data/processed/.gitkeep`**

Empty file, so the directory structure is visible even though its contents are gitignored.

- [x] **Step 6: Commit**

```bash
git add requirements.txt .gitignore .env.example data/processed/.gitkeep
git commit -m "chore: add embedding/db dependencies and .env scaffolding"
```

Note: do NOT create the real `.env` file as part of this commit — that happens locally, outside git, whenever the user fills in their actual `DATABASE_URL`.

---

### Task 2: Add `embedding` field to `ChunkRecord`

**Files:**
- Modify: `src/ingestion/schema.py`
- Modify: `tests/ingestion/test_schema.py`

**Interfaces:**
- Consumes: existing `ChunkRecord` (Task 2 of the previous plan).
- Produces: `ChunkRecord.embedding: Optional[list[float]] = None`, included in `to_dict()` output under the key `"embedding"`. Used by `run_full_pipeline.py` (Task 5) to attach vectors after chunking, and by `db.py` (Task 7) to read them back out via `to_dict()`.

- [x] **Step 1: Write the failing test**

Add to `tests/ingestion/test_schema.py`:

```python
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
```

- [x] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_schema.py -v
```

Expected: FAIL — `record.embedding = [0.1, 0.2, 0.3]` raises no error (dataclasses allow arbitrary attribute assignment by default... actually it will raise `AttributeError`-free assignment but the field won't exist, so `to_dict()` (which uses `dataclasses.asdict`) will not include it, and `d["embedding"]` raises `KeyError`).

- [x] **Step 3: Add the field**

In `src/ingestion/schema.py`, add `embedding: Optional[list] = None` as the last field of `ChunkRecord` (after `chunk_text`):

```python
@dataclass
class ChunkRecord:
    chunk_id: str
    board: str
    class_: str
    subject: str
    book_title: str
    chapter_number: int
    chapter_name: str
    topic: Optional[str]
    chunk_type: str
    page_start: int
    page_end: int
    chunk_text: str
    embedding: Optional[list] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("class_")
        return d
```

- [x] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_schema.py -v
```

Expected: 4 passed (2 existing + 2 new).

- [x] **Step 5: Run the full suite to confirm no regressions**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" -v
```

Expected: all previously-passing tests (54 as of the last plan) still pass, plus the 2 new ones.

- [x] **Step 6: Commit**

```bash
git add src/ingestion/schema.py tests/ingestion/test_schema.py
git commit -m "feat: add optional embedding field to ChunkRecord"
```

---

### Task 3: `embed.py` — Ollama embedding client

**Files:**
- Create: `src/ingestion/embed.py`
- Create: `tests/ingestion/test_embed.py`

**Interfaces:**
- Consumes: nothing from earlier tasks in this plan (standalone HTTP client).
- Produces: `build_document_prompt(text: str, title: Optional[str]) -> str` and `embed_texts(texts: list[str], titles: list[Optional[str]], model: str = "embeddinggemma", host: str = "http://localhost:11434") -> list[list[float]]`. Used by `scripts/run_full_pipeline.py` (Task 5).

**Note:** the `embed_texts` tests call the real local Ollama server (consistent with how `test_extract.py` calls the real PDFs) — Ollama must be running with `embeddinggemma` pulled, which is already true in this environment.

- [x] **Step 1: Write the failing tests**

```python
# tests/ingestion/test_embed.py
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
```

- [x] **Step 2: Run tests to verify they fail**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_embed.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.embed'`.

- [x] **Step 3: Write the implementation**

```python
# src/ingestion/embed.py
from typing import Optional

import requests

DEFAULT_MODEL = "embeddinggemma"
DEFAULT_HOST = "http://localhost:11434"


def build_document_prompt(text: str, title: Optional[str]) -> str:
    title_value = title if title else "none"
    return f"title: {title_value} | text: {text}"


def embed_texts(
    texts: list[str],
    titles: list[Optional[str]],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
) -> list[list[float]]:
    prompts = [
        build_document_prompt(text, title) for text, title in zip(texts, titles)
    ]
    response = requests.post(
        f"{host}/api/embed",
        json={"model": model, "input": prompts},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"]
```

- [x] **Step 4: Run tests to verify they pass**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_embed.py -v
```

Expected: 5 passed. If any test fails with a connection error, confirm Ollama is running: `"/c/Users/parva/AppData/Local/Programs/Ollama/ollama.exe" list` should show `embeddinggemma`.

- [x] **Step 5: Commit**

```bash
git add src/ingestion/embed.py tests/ingestion/test_embed.py
git commit -m "feat: add Ollama embedding client for embeddinggemma"
```

---

### Task 4: Full-book chapter-count regression guard

**Files:**
- Create: `tests/ingestion/test_full_book_extraction.py`

**Interfaces:**
- Consumes: `extract_pages` (existing), `clean_text` (existing), `parse_contents`/`split_into_chapters` (existing).
- Produces: no new production code — this validates that extracting the *entire* body (pages 10-147) against the *full* table of contents (not just chapter 1, as the pilot did) correctly locates every chapter's heading without the TOC/body regex collision described in the design doc, before `run_full_pipeline.py` (Task 5) is built on top of that assumption.

**Note:** this test may pass immediately without any code change — that is expected and fine. It exists to catch a real risk (the TOC/body collision) at full-book scale before Task 5 depends on it, not to drive new behavior. If it fails, that is real signal requiring a fix to `structure.py` before proceeding.

- [x] **Step 1: Write the test**

```python
# tests/ingestion/test_full_book_extraction.py
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
```

- [x] **Step 2: Run it**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_full_book_extraction.py -v
```

Expected: 2 passed. This may take 10-20 seconds since it extracts ~138 pages per book via PyMuPDF.

**If it fails:** read the `ValueError` message from `split_into_chapters` (it names which chapter's heading couldn't be found) and inspect that chapter's extracted text directly (e.g. via a throwaway `python -c` script calling `extract_pages` on a narrower page range around where that chapter should be) to find the actual formatting difference before changing `structure.py`. Do not guess at a fix — confirm the actual extracted text first, the same way earlier bugs in this project were diagnosed.

- [x] **Step 3: Commit**

```bash
git add tests/ingestion/test_full_book_extraction.py
git commit -m "test: add full-book chapter-count regression guard"
```

---

### Task 5: `run_full_pipeline.py` — rollout and embedding orchestration

**Files:**
- Create: `scripts/run_full_pipeline.py`

**Interfaces:**
- Consumes: `extract_pages` (existing), `clean_text` (existing), `parse_contents`/`split_into_chapters`/`segment_chapter` (existing), `BookMeta`/`build_chunk_records` (existing), `embed_texts` (Task 3), `ChunkRecord.embedding`/`to_dict()` (Task 2).
- Produces: `data/processed/EVS_5_full.jsonl` and `data/processed/SCI_8_full.jsonl`, each one JSON object per line (chunk fields + `embedding`). Used by `load_to_supabase.py` (Task 7).

- [x] **Step 1: Write `scripts/run_full_pipeline.py`**

```python
# scripts/run_full_pipeline.py
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
```

- [x] **Step 2: Run it**

```bash
"D:/Coding/edtech/.venv/Scripts/python" scripts/run_full_pipeline.py
```

Expected: prints per-chapter chunk counts for all 25 EVS chapters and all 19 Science chapters, then a final total per book, with no `AssertionError`. This will take several minutes (CPU-bound Ollama inference across ~1,500+ chunks) — let it run to completion.

- [x] **Step 3: Spot-check the output**

```bash
"D:/Coding/edtech/.venv/Scripts/python" -c "
import json
for name in ['EVS_5_full.jsonl', 'SCI_8_full.jsonl']:
    with open(f'data/processed/{name}', encoding='utf-8') as f:
        records = [json.loads(l) for l in f]
    chapters = {r['chapter_number'] for r in records}
    print(name, 'chunks:', len(records), 'chapters covered:', len(chapters))
    assert all(len(r['embedding']) == 768 for r in records)
"
```

Expected: `EVS_5_full.jsonl chunks: <N> chapters covered: 25` and `SCI_8_full.jsonl chunks: <M> chapters covered: 19`, with no `AssertionError` from the embedding-length check.

- [x] **Step 4: Commit**

```bash
git add scripts/run_full_pipeline.py
git commit -m "feat: add full-rollout pipeline with per-chapter embedding"
```

Note: `data/processed/*.jsonl` stays untracked (gitignored per Task 1) — only the script is committed.

---

### Task 6: Supabase migration — enable pgvector and create the `chunks` table

**Files:**
- Create: `supabase/migrations/0001_create_chunks_table.sql` (kept as a local record of the migration; applied via the Supabase MCP tool, not `psql`)

**Interfaces:**
- Produces: a `chunks` table in the `aedsgktxvmddarqurbhi` Supabase project with an HNSW index on `embedding`. Used by `load_to_supabase.py` (Task 7).

- [x] **Step 1: Write the migration file**

```sql
-- supabase/migrations/0001_create_chunks_table.sql
create extension if not exists vector;

create table chunks (
  id bigint generated always as identity primary key,
  chunk_id text not null unique,
  board text not null,
  class text not null,
  subject text not null,
  book_title text not null,
  chapter_number int not null,
  chapter_name text not null,
  topic text,
  chunk_type text not null,
  page_start int not null,
  page_end int not null,
  chunk_text text not null,
  embedding vector(768) not null
);

create index on chunks using hnsw (embedding vector_cosine_ops);
```

- [x] **Step 2: Apply the migration**

Use the Supabase MCP tool `apply_migration` with `project_id="aedsgktxvmddarqurbhi"`, `name="create_chunks_table"`, and the SQL contents above.

- [x] **Step 3: Verify**

Use the Supabase MCP tool `list_tables` with `project_id="aedsgktxvmddarqurbhi"`, `schemas=["public"]`, `verbose=true`. Expected: `chunks` table listed with all 13 columns above, `embedding` typed as `vector`.

Use the Supabase MCP tool `list_extensions` with `project_id="aedsgktxvmddarqurbhi"`. Expected: the `vector` extension entry now has a non-null `installed_version`.

- [x] **Step 4: Commit**

```bash
git add supabase/migrations/0001_create_chunks_table.sql
git commit -m "feat: add pgvector chunks table migration"
```

---

### Task 7: `db.py` and `load_to_supabase.py` — bulk load into pgvector

**Files:**
- Create: `src/ingestion/db.py`
- Create: `tests/ingestion/test_db.py`
- Create: `scripts/load_to_supabase.py`

**Interfaces:**
- Consumes: JSONL records produced by `run_full_pipeline.py` (Task 5), the `chunks` table from Task 6.
- Produces: `row_from_record(d: dict) -> tuple` and the SQL constants `INSERT_SQL`, `INSERT_TEMPLATE`, all in `src/ingestion/db.py`. `scripts/load_to_supabase.py` is the orchestration entrypoint — not imported anywhere, verified by running it for real.

- [x] **Step 1: Write the failing test for `row_from_record`**

```python
# tests/ingestion/test_db.py
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
```

- [x] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_db.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.db'`.

- [x] **Step 3: Write `src/ingestion/db.py`**

```python
# src/ingestion/db.py
INSERT_SQL = """
insert into chunks (
    chunk_id, board, class, subject, book_title, chapter_number,
    chapter_name, topic, chunk_type, page_start, page_end, chunk_text, embedding
) values %s
on conflict (chunk_id) do nothing
"""

INSERT_TEMPLATE = "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)"


def row_from_record(d: dict) -> tuple:
    embedding_literal = "[" + ",".join(str(x) for x in d["embedding"]) + "]"
    return (
        d["chunk_id"],
        d["board"],
        d["class"],
        d["subject"],
        d["book_title"],
        d["chapter_number"],
        d["chapter_name"],
        d["topic"],
        d["chunk_type"],
        d["page_start"],
        d["page_end"],
        d["chunk_text"],
        embedding_literal,
    )
```

- [x] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_db.py -v
```

Expected: 2 passed.

- [x] **Step 5: Write `scripts/load_to_supabase.py`**

```python
# scripts/load_to_supabase.py
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from ingestion.db import INSERT_SQL, INSERT_TEMPLATE, row_from_record

load_dotenv()


def load_file(conn, path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        rows = [row_from_record(json.loads(line)) for line in f]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, INSERT_SQL, rows, template=INSERT_TEMPLATE
        )
    conn.commit()
    return len(rows)


def main():
    database_url = os.environ["DATABASE_URL"]
    processed_dir = Path("data/processed")
    files = sorted(processed_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit("No JSONL files found in data/processed/")

    conn = psycopg2.connect(database_url)
    try:
        total = 0
        for path in files:
            n = load_file(conn, path)
            total += n
            print(f"{path.name}: loaded {n} rows")
        print(f"Total: {total} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
```

- [x] **Step 6: Confirm `.env` is filled in**

Ask the user to confirm their real `DATABASE_URL` is set in their local `.env` file (from Task 1's `.env.example`) before proceeding — this step cannot be automated or verified by reading the file's contents (the value must not be echoed into the conversation).

- [x] **Step 7: Run it**

```bash
"D:/Coding/edtech/.venv/Scripts/python" scripts/load_to_supabase.py
```

Expected: prints `EVS_5_full.jsonl: loaded <N> rows`, `SCI_8_full.jsonl: loaded <M> rows`, and `Total: <N+M> rows`, with no connection or SQL errors.

- [x] **Step 8: Commit**

```bash
git add src/ingestion/db.py tests/ingestion/test_db.py scripts/load_to_supabase.py
git commit -m "feat: add pgvector bulk loader"
```

---

### Task 8: End-to-end verification

**Files:** none (verification only).

**Interfaces:** none — this task confirms Tasks 5-7 together produced a correct end state.

- [x] **Step 1: Verify row count matches the JSONL output**

Use the Supabase MCP tool `execute_sql` with `project_id="aedsgktxvmddarqurbhi"` and query `select count(*) from chunks;`. Compare against the total printed by `load_to_supabase.py` in Task 7 Step 7 — they must match exactly (the `on conflict (chunk_id) do nothing` clause means a partial rerun would under-count, so a mismatch here is real signal, not noise).

- [x] **Step 2: Spot-check a handful of rows**

Use `execute_sql` with:

```sql
select chunk_id, subject, chapter_number, chapter_name, topic, chunk_type, page_start, page_end, left(chunk_text, 80) as text_preview, vector_dims(embedding) as dims
from chunks
order by random()
limit 5;
```

Expected: 5 rows with plausible-looking metadata (real chapter names/topics, `dims` always `768`), across both subjects.

- [x] **Step 3: Verify per-chapter coverage**

```sql
select subject, count(distinct chapter_number) as chapters_covered
from chunks
group by subject;
```

Expected: `Environmental Studies` → 25, `Science` → 19 — matching the chapter counts confirmed in Task 4.

- [x] **Step 4: Report results to the user**

Summarize total row count, per-book chunk counts, and confirm the vector index and chapter coverage are correct. No further commit needed for this task — it's read-only verification.

---

## Self-Review Notes

- **Spec coverage:** Full 44-chapter rollout → Tasks 4-5. Embedding via Ollama/embeddinggemma with the documented prompt format → Task 3. pgvector schema and extension → Task 6. Bulk load via direct psycopg2 connection (not routed through MCP tool calls, per the design's reasoning about data volume) → Task 7. End-to-end verification via Supabase MCP queries (not mocks) → Task 8. Query-time retrieval, LMS/quiz layers, and normalized schema are explicitly out of scope per the design and have no tasks here.
- **Type consistency:** `ChunkRecord.embedding` (Task 2) is read via `record.to_dict()["embedding"]` in `run_full_pipeline.py` (Task 5) and via the raw JSONL `"embedding"` key in `row_from_record` (Task 7) — same key throughout. `BookMeta` fields (`class_`, `subject_code`, etc.) match the existing `chunk.py` module unchanged. `db.py`'s `INSERT_SQL`/`INSERT_TEMPLATE` column order (`chunk_id, board, class, subject, book_title, chapter_number, chapter_name, topic, chunk_type, page_start, page_end, chunk_text, embedding`) matches `row_from_record`'s tuple order exactly, and matches the `chunks` table's column order from Task 6's migration.
- **No placeholders:** every step has runnable code, exact SQL, or an exact MCP tool call with expected output. Task 4 and parts of Tasks 6-8 are intentionally verification-only (no new production code) where the spec calls for validating existing behavior at new scale or confirming live infrastructure state — each still has a concrete pass/fail condition, not an open-ended "check it works."
