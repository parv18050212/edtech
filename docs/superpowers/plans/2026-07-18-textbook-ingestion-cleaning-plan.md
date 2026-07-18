# Textbook Ingestion Cleaning & Chunking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the two source textbook PDFs (`Class 5 EVS.pdf`, `Science Class 8.pdf`) into clean, hierarchically-chunked, metadata-tagged JSONL records, validated against a one-chapter-per-book pilot, per `docs/superpowers/specs/2026-07-18-textbook-ingestion-cleaning-design.md`.

**Architecture:** PyMuPDF extracts each page as position-sorted text blocks, wrapping bold spans as `**text**` and inserting `[[PAGE:N]]` markers (this is how bold-label topics and page ranges survive into plain-string processing). Cleaning strips boilerplate/stray page numbers. Structure detection matches chapter headings against the book's own table of contents, then walks the bold-marked chapter text to split it into topics and typed blocks (paragraph/activity/recall/info_box/exercise). Chunking sub-splits paragraph blocks to a ~250-token target and emits one JSONL record per chunk.

**Tech Stack:** Python 3.12, PyMuPDF (`pymupdf`), pytest. No LLM calls, no OCR, no external tokenizer — token count is a word-based estimate.

## Global Constraints

- Python 3.12.9 (already installed on this machine).
- `board` = `"Maharashtra State Board"` for both books (verbatim, not inferred).
- Book metadata is set explicitly per book, never inferred from PDF content:
  - EVS: `class_="5"`, `subject="Environmental Studies"`, `subject_code="EVS"`, `book_title="Environmental Studies (Part One), Standard Five"`.
  - Science: `class_="8"`, `subject="Science"`, `subject_code="SCI"`, `book_title="General Science, Standard Eight"`.
- PDF page indices used throughout this codebase are **0-indexed**, matching PyMuPDF's native convention (not the 1-indexed convention some CLI tools use).
- Verified page facts (both books share the same front-matter layout from this publisher):
  - Both PDFs have 148 pages total (0-indexed pages 0-147).
  - Table of contents is on 0-indexed page 9 in both books.
  - Chapter 1 heading is on 0-indexed page 10 in both books; Chapter 2 heading is on 0-indexed page 15 in both books. Chapter 1 therefore fully fits in 0-indexed page range 10-14 inclusive.
- Verified bold-detection fact: chapter headings, numbered topic headings (e.g. "1.1 Five Kingdom system of classification"), bold-label topic headers (e.g. "Stars :", "Gravity"), and all block markers ("Try this", "Can you recall?", etc.) render with PyMuPDF span `flags & 16` set (bold) and font `TimesNewRomanPS-BoldMT`. Regular body text is not bold. This is the mechanism topic/block detection relies on.
- No dependency beyond `pymupdf` and `pytest` is needed for this slice.

---

## File Structure

```
edtech/
  data/
    raw/
      Class 5 EVS.pdf              # moved here from repo root
      Science Class 8.pdf          # moved here from repo root
    interim/                       # pilot output (gitignored)
  src/
    ingestion/
      __init__.py
      schema.py                    # ChunkRecord
      clean.py                     # boilerplate/page-number stripping, whitespace normalization
      structure.py                 # TOC parsing, chapter splitting, topic/block segmentation
      chunk.py                     # token estimate, paragraph splitting, chunk_id, ChunkRecord assembly
      extract.py                   # PyMuPDF extraction, reading-order sort, bold/page markers
  scripts/
    run_pilot.py                   # orchestrates the pipeline for the pilot chapters
  tests/
    ingestion/
      __init__.py
      test_schema.py
      test_clean.py
      test_structure.py
      test_chunk.py
      test_extract.py
  requirements.txt
  pyproject.toml
  .gitignore                       # already exists, no change needed
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `src/ingestion/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/ingestion/__init__.py`
- Create: `data/raw/.gitkeep`, `data/interim/.gitkeep`
- Move: `Class 5 EVS.pdf` → `data/raw/Class 5 EVS.pdf`
- Move: `Science Class 8.pdf` → `data/raw/Science Class 8.pdf`
- Move: `RAG_Learning_Platform_Study.pdf` → `data/raw/RAG_Learning_Platform_Study.pdf`

**Interfaces:**
- Produces: a `pytest`-discoverable project where `from ingestion import schema` (etc.) works from `tests/`, and `data/raw/Class 5 EVS.pdf` / `data/raw/Science Class 8.pdf` are the paths every later task's code and tests use.

- [x] **Step 1: Create the virtual environment and directories**

```bash
cd "D:/Coding/edtech"
python -m venv .venv
mkdir -p data/raw data/interim src/ingestion tests/ingestion scripts
```

- [x] **Step 2: Move the source PDFs into `data/raw/`**

```bash
git mv "Class 5 EVS.pdf" "data/raw/Class 5 EVS.pdf"
git mv "Science Class 8.pdf" "data/raw/Science Class 8.pdf"
git mv "RAG_Learning_Platform_Study.pdf" "data/raw/RAG_Learning_Platform_Study.pdf"
```

- [x] **Step 3: Create `requirements.txt`**

```
pymupdf==1.28.0
pytest==8.3.3
```

- [x] **Step 4: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [x] **Step 5: Create empty package markers**

Create `src/ingestion/__init__.py`, `tests/__init__.py`, and `tests/ingestion/__init__.py` (all empty files). `tests/__init__.py` is required, not optional: without it, pytest's default import mode treats `tests/ingestion` itself as the top-level `ingestion` package (since it's the first ancestor directory with an `__init__.py`), which collides with and shadows the real `src/ingestion` package in `sys.modules` — every `from ingestion.xxx import ...` in a test then fails with a confusing `ModuleNotFoundError: No module named 'ingestion.xxx'` even though `ingestion` itself imports fine.

- [x] **Step 6: Install dependencies and verify pytest collects cleanly**

```bash
"D:/Coding/edtech/.venv/Scripts/pip" install -r requirements.txt
"D:/Coding/edtech/.venv/Scripts/pytest" --collect-only
```

Expected: exits with "no tests ran" (or similar) and no import/collection errors — there are no test files yet, so this only proves the pytest/pythonpath configuration is valid.

- [x] **Step 7: Create `data/raw/.gitkeep` and `data/interim/.gitkeep`**

Empty files, so the empty `data/interim/` directory structure is visible even though its contents are gitignored.

- [x] **Step 8: Commit**

```bash
git add data requirements.txt pyproject.toml src tests scripts
git commit -m "chore: scaffold ingestion project structure"
```

---

### Task 2: `schema.py` — ChunkRecord

**Files:**
- Create: `src/ingestion/schema.py`
- Test: `tests/ingestion/test_schema.py`

**Interfaces:**
- Produces: `ChunkRecord` dataclass with fields `chunk_id: str, board: str, class_: str, subject: str, book_title: str, chapter_number: int, chapter_name: str, topic: Optional[str], chunk_type: str, page_start: int, page_end: int, chunk_text: str`, and method `to_dict() -> dict` (serializes `class_` field as JSON key `"class"`). Used by `chunk.py` (Task 8) to build records and by `scripts/run_pilot.py` (Task 10) to serialize to JSONL.

- [x] **Step 1: Write the failing test**

```python
# tests/ingestion/test_schema.py
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
```

- [x] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_schema.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.schema'`.

- [x] **Step 3: Write the implementation**

```python
# src/ingestion/schema.py
from dataclasses import asdict, dataclass
from typing import Optional


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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("class_")
        return d
```

- [x] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_schema.py -v
```

Expected: 2 passed.

- [x] **Step 5: Commit**

```bash
git add src/ingestion/schema.py tests/ingestion/test_schema.py
git commit -m "feat: add ChunkRecord schema"
```

---

### Task 3: `clean.py` — boilerplate and noise stripping

**Files:**
- Create: `src/ingestion/clean.py`
- Test: `tests/ingestion/test_clean.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `clean_text(text: str, boilerplate_phrases: list[str]) -> str`, used by `scripts/run_pilot.py` (Task 10) on the raw output of `extract.py` (Task 9) before structure detection (Task 4-7).

- [x] **Step 1: Write the failing test**

```python
# tests/ingestion/test_clean.py
from ingestion.clean import clean_text


def test_strips_standalone_page_number_lines():
    raw = (
        "The moon is closest to the earth.\n"
        "2\n"
        "That is why, it appears to be so big."
    )
    cleaned = clean_text(raw, boilerplate_phrases=[])
    assert "\n2\n" not in cleaned
    assert "The moon is closest to the earth." in cleaned
    assert "That is why, it appears to be so big." in cleaned


def test_preserves_page_markers():
    raw = "[[PAGE:10]]\nSome body text.\n[[PAGE:11]]\nMore text."
    cleaned = clean_text(raw, boilerplate_phrases=[])
    assert "[[PAGE:10]]" in cleaned
    assert "[[PAGE:11]]" in cleaned


def test_strips_boilerplate_lines():
    raw = (
        "1. Our Earth and Our Solar System\n"
        "Maharashtra State Bureau of Textbook Production and Curriculum Research, Pune.\n"
        "The sun is a star."
    )
    cleaned = clean_text(
        raw,
        boilerplate_phrases=[
            "Maharashtra State Bureau of Textbook Production and Curriculum Research, Pune."
        ],
    )
    assert "Maharashtra State Bureau" not in cleaned
    assert "The sun is a star." in cleaned


def test_normalizes_whitespace_and_hyphenation():
    raw = "This is an exam-\nple of a hyphenated   word   split across a line."
    cleaned = clean_text(raw, boilerplate_phrases=[])
    assert "example of a hyphenated word split across a line." in cleaned
```

- [x] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_clean.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.clean'`.

- [x] **Step 3: Write the implementation**

```python
# src/ingestion/clean.py
import re

PAGE_MARKER_RE = re.compile(r"\[\[PAGE:\d+\]\]")
STANDALONE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def _strip_page_numbers(text: str) -> str:
    lines = text.split("\n")
    kept = [line for line in lines if not STANDALONE_NUMBER_RE.match(line)]
    return "\n".join(kept)


def _strip_boilerplate(text: str, boilerplate_phrases: list[str]) -> str:
    normalized_phrases = {p.strip().lower() for p in boilerplate_phrases}
    lines = text.split("\n")
    kept = [
        line for line in lines if line.strip().lower() not in normalized_phrases
    ]
    return "\n".join(kept)


def _normalize_whitespace(text: str) -> str:
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def clean_text(text: str, boilerplate_phrases: list[str]) -> str:
    text = _strip_boilerplate(text, boilerplate_phrases)
    text = _strip_page_numbers(text)
    text = _normalize_whitespace(text)
    return text
```

- [x] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_clean.py -v
```

Expected: 4 passed.

- [x] **Step 5: Commit**

```bash
git add src/ingestion/clean.py tests/ingestion/test_clean.py
git commit -m "feat: add text cleaning (boilerplate, page numbers, whitespace)"
```

---

### Task 4: `structure.py` Part A — table of contents parsing

**Files:**
- Create: `src/ingestion/structure.py`
- Test: `tests/ingestion/test_structure.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `TocEntry` dataclass (`chapter_number: int, chapter_name: str, start_page: int`) and `parse_contents(contents_text: str) -> list[TocEntry]`. Used by `split_into_chapters` (Task 5) and `scripts/run_pilot.py` (Task 10).

- [x] **Step 1: Write the failing test**

```python
# tests/ingestion/test_structure.py
from ingestion.structure import TocEntry, parse_contents

EVS_TOC_TEXT = (
    "1. Our Earth and Our Solar System .......................................................... 1 "
    "2. Motions of the Earth.............................................................................. 6 "
    "3. The Earth and its Living World .......................................................... 11"
)

SCIENCE_TOC_TEXT = (
    "1. \t Living World and Classification of Microbes....................................................... 1 "
    "2. \t Health and Diseases................................................................................................. 6 "
    "3. \t Force and Pressure ............................................................................................... 14"
)


def test_parse_contents_evs():
    entries = parse_contents(EVS_TOC_TEXT)
    assert entries[0] == TocEntry(1, "Our Earth and Our Solar System", 1)
    assert entries[1] == TocEntry(2, "Motions of the Earth", 6)
    assert entries[2] == TocEntry(3, "The Earth and its Living World", 11)


def test_parse_contents_science_handles_tabs():
    entries = parse_contents(SCIENCE_TOC_TEXT)
    assert entries[0] == TocEntry(1, "Living World and Classification of Microbes", 1)
    assert entries[1] == TocEntry(2, "Health and Diseases", 6)
    assert entries[2] == TocEntry(3, "Force and Pressure", 14)


def test_parse_contents_tolerates_bold_markers():
    text = "**1.** **Our Earth and Our Solar System** .......................... 1"
    entries = parse_contents(text)
    assert entries[0] == TocEntry(1, "Our Earth and Our Solar System", 1)
```

- [x] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_structure.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.structure'`.

- [x] **Step 3: Write the implementation**

```python
# src/ingestion/structure.py
import re
from dataclasses import dataclass
from typing import Optional

TOC_ENTRY_RE = re.compile(
    r"\*{0,2}\s*(\d{1,2})\.\*{0,2}\s*\*{0,2}\s*(.+?)\*{0,2}\s*\.{2,}\s*\*{0,2}\s*(\d{1,3})\*{0,2}"
)


@dataclass
class TocEntry:
    chapter_number: int
    chapter_name: str
    start_page: int


def parse_contents(contents_text: str) -> list[TocEntry]:
    entries = []
    for match in TOC_ENTRY_RE.finditer(contents_text):
        number = int(match.group(1))
        name = re.sub(r"\s+", " ", match.group(2)).strip()
        start_page = int(match.group(3))
        entries.append(TocEntry(number, name, start_page))
    return entries
```

- [x] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_structure.py -v
```

Expected: 3 passed.

- [x] **Step 5: Commit**

```bash
git add src/ingestion/structure.py tests/ingestion/test_structure.py
git commit -m "feat: add table of contents parsing"
```

---

### Task 5: `structure.py` Part B — chapter splitting

**Files:**
- Modify: `src/ingestion/structure.py`
- Modify: `tests/ingestion/test_structure.py`

**Interfaces:**
- Consumes: `TocEntry` (Task 4).
- Produces: `split_into_chapters(full_text: str, toc: list[TocEntry]) -> dict[int, str]`. Used by `scripts/run_pilot.py` (Task 10).

- [x] **Step 1: Write the failing test**

Append to `tests/ingestion/test_structure.py`:

```python
from ingestion.structure import split_into_chapters


def test_split_into_chapters_isolates_each_chapters_body():
    toc = [
        TocEntry(1, "Our Earth and Our Solar System", 1),
        TocEntry(2, "Motions of the Earth", 6),
    ]
    full_text = (
        "**1. Our Earth and Our Solar System**\n"
        "The sun and the moon are close to earth.\n"
        "**2. Motions of the Earth**\n"
        "The earth rotates on its axis.\n"
    )
    chapters = split_into_chapters(full_text, toc)
    assert "The sun and the moon are close to earth." in chapters[1]
    assert "2. Motions of the Earth" not in chapters[1]
    assert "The earth rotates on its axis." in chapters[2]


def test_split_into_chapters_raises_if_heading_not_found():
    toc = [TocEntry(1, "A Chapter That Does Not Exist", 1)]
    with pytest.raises(ValueError):
        split_into_chapters("no matching heading here", toc)
```

Add `import pytest` to the top of `tests/ingestion/test_structure.py`.

- [x] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_structure.py -v
```

Expected: FAIL with `ImportError: cannot import name 'split_into_chapters'`.

- [x] **Step 3: Write the implementation**

Add to `src/ingestion/structure.py`:

```python
def split_into_chapters(full_text: str, toc: list[TocEntry]) -> dict[int, str]:
    matches = []
    for entry in toc:
        pattern = re.compile(
            r"\*{0,2}\s*"
            + re.escape(f"{entry.chapter_number}.")
            + r"\s*\*{0,2}\s*"
            + re.escape(entry.chapter_name)
            + r"\s*\*{0,2}"
        )
        found = pattern.search(full_text)
        if not found:
            raise ValueError(
                f"Could not locate heading for chapter {entry.chapter_number}: "
                f"{entry.chapter_name!r}"
            )
        matches.append((entry.chapter_number, found.start(), found.end()))

    matches.sort(key=lambda m: m[1])
    chapters: dict[int, str] = {}
    for i, (chapter_number, _start, end) in enumerate(matches):
        next_start = matches[i + 1][1] if i + 1 < len(matches) else len(full_text)
        chapters[chapter_number] = full_text[end:next_start]
    return chapters
```

- [x] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_structure.py -v
```

Expected: 5 passed.

- [x] **Step 5: Commit**

```bash
git add src/ingestion/structure.py tests/ingestion/test_structure.py
git commit -m "feat: add chapter splitting via table of contents matching"
```

---

### Task 6: `structure.py` Part C — classifying bold spans

**Files:**
- Modify: `src/ingestion/structure.py`
- Modify: `tests/ingestion/test_structure.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure string classifier).
- Produces: `classify_bold_span(text: str) -> tuple[str, str]`, returning `("topic", value)`, `("marker", chunk_type)`, or `("emphasis", text)`. Used by `segment_chapter` (Task 7).

- [x] **Step 1: Write the failing test**

Append to `tests/ingestion/test_structure.py`:

```python
from ingestion.structure import classify_bold_span


@pytest.mark.parametrize(
    "bold_text,expected",
    [
        ("1.1 Five Kingdom system of classification", ("topic", "1.1 Five Kingdom system of classification")),
        ("Stars :", ("topic", "Stars")),
        ("Gravity", ("topic", "Gravity")),
        ("Dwarf planets", ("topic", "Dwarf planets")),
        ("2. Use your brain power !", ("marker", "activity")),
        ("Try this.", ("marker", "activity")),
        ("Can you tell  ?", ("marker", "activity")),
        ("Can you recall?", ("marker", "recall")),
        ("Do you know  ?", ("marker", "info_box")),
        ("Find out my partner.", ("marker", "info_box")),
        ("In History......", ("marker", "info_box")),
        ("Always remember", ("marker", "info_box")),
        ("5. \tFill in the blanks.", ("marker", "exercise")),
        ("1. Answer the following in your own words.", ("marker", "exercise")),
        ("State true or false.", ("marker", "exercise")),
        (
            "This is a much longer bold phrase used only for testing purposes",
            ("emphasis", "This is a much longer bold phrase used only for testing purposes"),
        ),
    ],
)
def test_classify_bold_span(bold_text, expected):
    assert classify_bold_span(bold_text) == expected
```

- [x] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_structure.py -v
```

Expected: FAIL with `ImportError: cannot import name 'classify_bold_span'`.

- [x] **Step 3: Write the implementation**

Add to `src/ingestion/structure.py`:

```python
NUMBERED_TOPIC_RE = re.compile(r"^(\d{1,2}\.\d{1,2})\s+(.+)$")

BLOCK_MARKER_PREFIXES: dict[str, list[str]] = {
    "activity": ["try this", "let's try this", "use your brain power", "can you tell"],
    "recall": ["can you recall", "let's recall"],
    "info_box": ["do you know", "research", "find out", "in history", "always remember"],
    "exercise": [
        "fill in the blank",
        "state true or false",
        "true or false",
        "answer the following",
        "answer in",
        "what's the solution",
    ],
}


def _normalize_marker(text: str) -> str:
    normalized = text.strip()
    normalized = re.sub(r"^\d+\.\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(".! ?")
    return normalized.lower()


def classify_bold_span(text: str) -> tuple[str, str]:
    stripped = text.strip()

    numbered = NUMBERED_TOPIC_RE.match(stripped)
    if numbered:
        return ("topic", stripped)

    normalized = _normalize_marker(stripped)
    for chunk_type, prefixes in BLOCK_MARKER_PREFIXES.items():
        for prefix in prefixes:
            if normalized.startswith(prefix):
                return ("marker", chunk_type)

    word_count = len(normalized.split())
    if 0 < word_count <= 6:
        label = stripped.rstrip(": ").strip()
        return ("topic", label)

    return ("emphasis", stripped)
```

- [x] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_structure.py -v
```

Expected: 21 passed (5 previous + 16 parametrized cases).

- [x] **Step 5: Commit**

```bash
git add src/ingestion/structure.py tests/ingestion/test_structure.py
git commit -m "feat: classify bold spans into topics, block markers, or emphasis"
```

---

### Task 7: `structure.py` Part D — segmenting a chapter into topics and blocks

**Files:**
- Modify: `src/ingestion/structure.py`
- Modify: `tests/ingestion/test_structure.py`

**Interfaces:**
- Consumes: `classify_bold_span` (Task 6).
- Produces: `Block` dataclass (`chunk_type: str, text: str, page_start: int, page_end: int`), `TopicSegment` dataclass (`topic: Optional[str], blocks: list[Block]`), and `segment_chapter(chapter_text: str) -> list[TopicSegment]`. Used by `build_chunk_records` (Task 8) and `scripts/run_pilot.py` (Task 10).

- [x] **Step 1: Write the failing test**

Append to `tests/ingestion/test_structure.py`:

```python
from ingestion.structure import Block, TopicSegment, segment_chapter


def test_segment_chapter_splits_on_topics_and_markers():
    chapter_text = (
        "[[PAGE:10]]\n"
        "When we look up we see the sky. It has many stars.\n"
        "**Stars :** The heavenly bodies that twinkle are called stars.\n"
        "**2. Use your brain power !**\n"
        "Name two heavenly bodies that do not twinkle.\n"
        "[[PAGE:11]]\n"
        "**Planets :** Planets do not have light of their own.\n"
    )
    segments = segment_chapter(chapter_text)

    assert segments[0].topic is None
    assert segments[0].blocks[0].chunk_type == "paragraph"
    assert "many stars" in segments[0].blocks[0].text
    assert segments[0].blocks[0].page_start == 10
    assert segments[0].blocks[0].page_end == 10

    assert segments[1].topic == "Stars"
    assert segments[1].blocks[0].chunk_type == "paragraph"
    assert "heavenly bodies that twinkle" in segments[1].blocks[0].text
    assert segments[1].blocks[1].chunk_type == "activity"
    assert "do not twinkle" in segments[1].blocks[1].text

    assert segments[2].topic == "Planets"
    assert segments[2].blocks[0].page_start == 11
    assert "light of their own" in segments[2].blocks[0].text


def test_segment_chapter_keeps_emphasis_inline():
    # A bold run longer than 6 words falls through classify_bold_span's
    # topic heuristic to "emphasis", so it must stay inline as plain text
    # rather than starting a new topic segment.
    chapter_text = (
        "[[PAGE:5]]\n"
        "The fox jumped over **a fence that was much too tall for it to clear** easily.\n"
    )
    segments = segment_chapter(chapter_text)
    assert segments[0].blocks[0].text == (
        "The fox jumped over a fence that was much too tall for it to clear easily."
    )
```

- [x] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_structure.py -v
```

Expected: FAIL with `ImportError: cannot import name 'segment_chapter'`.

- [x] **Step 3: Write the implementation**

Add to `src/ingestion/structure.py`:

```python
TOKEN_RE = re.compile(r"\[\[PAGE:(\d+)\]\]|\*\*(.+?)\*\*")


@dataclass
class Block:
    chunk_type: str
    text: str
    page_start: int
    page_end: int


@dataclass
class TopicSegment:
    topic: Optional[str]
    blocks: list


def segment_chapter(chapter_text: str) -> list:
    segments: list[TopicSegment] = []
    current_topic: Optional[str] = None
    current_blocks: list[Block] = []
    current_type = "paragraph"
    current_text_parts: list[str] = []
    current_page = 0
    block_start_page = 0
    block_end_page = 0
    pos = 0

    def flush_block():
        nonlocal current_text_parts, current_type
        text = re.sub(r"\s+", " ", "".join(current_text_parts)).strip()
        if text:
            current_blocks.append(
                Block(current_type, text, block_start_page, block_end_page)
            )
        current_text_parts = []
        current_type = "paragraph"

    def flush_segment():
        nonlocal current_blocks
        if current_blocks:
            segments.append(TopicSegment(current_topic, current_blocks))
        current_blocks = []

    for match in TOKEN_RE.finditer(chapter_text):
        plain_before = chapter_text[pos : match.start()]
        if plain_before.strip():
            if not current_text_parts:
                block_start_page = current_page
            current_text_parts.append(plain_before)
            block_end_page = current_page
        pos = match.end()

        page_group, bold_group = match.group(1), match.group(2)
        if page_group is not None:
            current_page = int(page_group)
            continue

        kind, value = classify_bold_span(bold_group)
        if kind == "topic":
            flush_block()
            flush_segment()
            current_topic = value
            block_start_page = current_page
            block_end_page = current_page
        elif kind == "marker":
            flush_block()
            current_type = value
            block_start_page = current_page
            block_end_page = current_page
        else:  # emphasis: keep the plain text inline
            if not current_text_parts:
                block_start_page = current_page
            current_text_parts.append(value)
            block_end_page = current_page

    trailing = chapter_text[pos:]
    if trailing.strip():
        if not current_text_parts:
            block_start_page = current_page
        current_text_parts.append(trailing)
        block_end_page = current_page

    flush_block()
    flush_segment()
    return segments
```

- [x] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_structure.py -v
```

Expected: 23 passed.

- [x] **Step 5: Commit**

```bash
git add src/ingestion/structure.py tests/ingestion/test_structure.py
git commit -m "feat: segment chapter text into topics and typed blocks"
```

---

### Task 8: `chunk.py` — token estimate, paragraph splitting, chunk assembly

**Files:**
- Create: `src/ingestion/chunk.py`
- Test: `tests/ingestion/test_chunk.py`

**Interfaces:**
- Consumes: `ChunkRecord` (Task 2), `Block` / `TopicSegment` (Task 7).
- Produces: `estimate_tokens(text: str) -> int`, `finalize_chunk_text(text: str) -> str`, `split_paragraph(text: str, target_tokens: int = 250) -> list[str]`, `make_chunk_id(subject_code: str, class_: str, chapter_number: int, topic_index: int, seq: int) -> str`, `BookMeta` dataclass (`board: str, class_: str, subject: str, subject_code: str, book_title: str`), `build_chunk_records(book_meta: BookMeta, chapter_number: int, chapter_name: str, topic_segments: list) -> list[ChunkRecord]`. Used by `scripts/run_pilot.py` (Task 10).

- [x] **Step 1: Write the failing test**

```python
# tests/ingestion/test_chunk.py
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
    records = build_chunk_records(meta, chapter_number=1, chapter_name="Our Earth and Our Solar System", topic_segments=segments)

    assert len(records) == 2
    assert records[0].chunk_type == "paragraph"
    assert records[0].topic == "Stars"
    assert records[0].chapter_number == 1
    assert records[0].class_ == "5"
    assert records[1].chunk_type == "activity"
    assert records[1].chunk_text == "Name two stars you can see at night."
    assert records[0].chunk_id == "MSB_EVS5_CH01_TOP00_000"
    assert records[1].chunk_id == "MSB_EVS5_CH01_TOP00_001"
```

- [x] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_chunk.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.chunk'`.

- [x] **Step 3: Write the implementation**

```python
# src/ingestion/chunk.py
import re
from dataclasses import dataclass

from ingestion.schema import ChunkRecord

ARTIFACT_RE = re.compile(r"\[\[PAGE:\d+\]\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    words = len(text.split())
    if words == 0:
        return 0
    return max(1, round(words / 0.75))


def finalize_chunk_text(text: str) -> str:
    text = ARTIFACT_RE.sub(" ", text)
    text = text.replace("**", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_paragraph(text: str, target_tokens: int = 250) -> list[str]:
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text.strip()) if s]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > target_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def make_chunk_id(
    subject_code: str, class_: str, chapter_number: int, topic_index: int, seq: int
) -> str:
    return (
        f"MSB_{subject_code}{class_}_CH{chapter_number:02d}"
        f"_TOP{topic_index:02d}_{seq:03d}"
    )


@dataclass
class BookMeta:
    board: str
    class_: str
    subject: str
    subject_code: str
    book_title: str


def build_chunk_records(
    book_meta: BookMeta,
    chapter_number: int,
    chapter_name: str,
    topic_segments: list,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for topic_index, segment in enumerate(topic_segments):
        seq = 0
        for block in segment.blocks:
            if block.chunk_type == "paragraph":
                sub_texts = split_paragraph(block.text)
            else:
                cleaned = block.text.strip()
                sub_texts = [cleaned] if cleaned else []

            for sub_text in sub_texts:
                final_text = finalize_chunk_text(sub_text)
                if not final_text:
                    continue
                records.append(
                    ChunkRecord(
                        chunk_id=make_chunk_id(
                            book_meta.subject_code,
                            book_meta.class_,
                            chapter_number,
                            topic_index,
                            seq,
                        ),
                        board=book_meta.board,
                        class_=book_meta.class_,
                        subject=book_meta.subject,
                        book_title=book_meta.book_title,
                        chapter_number=chapter_number,
                        chapter_name=chapter_name,
                        topic=segment.topic,
                        chunk_type=block.chunk_type,
                        page_start=block.page_start,
                        page_end=block.page_end,
                        chunk_text=final_text,
                    )
                )
                seq += 1
    return records
```

- [x] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_chunk.py -v
```

Expected: 7 passed.

- [x] **Step 5: Commit**

```bash
git add src/ingestion/chunk.py tests/ingestion/test_chunk.py
git commit -m "feat: add token estimation, paragraph splitting, and chunk assembly"
```

---

### Task 9: `extract.py` — PyMuPDF extraction with reading-order sort and bold/page markers

**Files:**
- Create: `src/ingestion/extract.py`
- Test: `tests/ingestion/test_extract.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (only PyMuPDF and the real PDFs in `data/raw/`).
- Produces: `extract_pages(pdf_path: str, first_page: int, last_page: int) -> str` (0-indexed, inclusive page range), returning one string with `**bold**`-wrapped bold spans and `[[PAGE:N]]` markers before each page's content. Used by `scripts/run_pilot.py` (Task 10) and consumed downstream by `clean_text` (Task 3) and `parse_contents` / `split_into_chapters` / `segment_chapter` (Tasks 4-7).

**Note:** this task's tests run against the real PDFs in `data/raw/`, so Task 1 (which moves the PDFs there) must be complete first.

- [x] **Step 1: Write the failing test**

```python
# tests/ingestion/test_extract.py
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
```

- [x] **Step 2: Run test to verify it fails**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_extract.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.extract'`.

- [x] **Step 3: Write the implementation**

```python
# src/ingestion/extract.py
import fitz

BOLD_FLAG = 16


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & BOLD_FLAG) or "bold" in span["font"].lower()


def _block_sort_key(block: dict) -> tuple[float, float]:
    x0, y0, _x1, _y1 = block["bbox"]
    return (round(y0 / 10) * 10, x0)


def _block_to_marked_text(block: dict) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line["spans"]:
            text = span["text"]
            if not text:
                continue
            if _is_bold(span):
                parts.append(f"**{text}**")
            else:
                parts.append(text)
        parts.append("\n")
    return "".join(parts)


def extract_pages(pdf_path: str, first_page: int, last_page: int) -> str:
    doc = fitz.open(pdf_path)
    try:
        out: list[str] = []
        for page_number in range(first_page, last_page + 1):
            page = doc[page_number]
            out.append(f"[[PAGE:{page_number}]]")
            blocks = [
                b for b in page.get_text("dict")["blocks"] if "lines" in b
            ]
            blocks.sort(key=_block_sort_key)
            for block in blocks:
                out.append(_block_to_marked_text(block))
        return "\n".join(out)
    finally:
        doc.close()
```

- [x] **Step 4: Run test to verify it passes**

```bash
"D:/Coding/edtech/.venv/Scripts/pytest" tests/ingestion/test_extract.py -v
```

Expected: 5 passed.

- [x] **Step 5: Commit**

```bash
git add src/ingestion/extract.py tests/ingestion/test_extract.py
git commit -m "feat: add PyMuPDF extraction with reading-order sort and bold/page markers"
```

---

### Task 10: Pilot run and manual review

**Files:**
- Create: `scripts/run_pilot.py`
- Create: `data/interim/REVIEW.md` (written during Step 4, findings filled in from actual pilot output)

**Interfaces:**
- Consumes: `extract_pages` (Task 9), `clean_text` (Task 3), `TocEntry`/`parse_contents`/`split_into_chapters`/`segment_chapter` (Tasks 4-7), `BookMeta`/`build_chunk_records` (Task 8), `ChunkRecord.to_dict()` (Task 2).
- Produces: `data/interim/EVS_5_ch01.jsonl`, `data/interim/SCI_8_ch01.jsonl`, and the human-reviewed `data/interim/REVIEW.md`.

- [x] **Step 1: Write `scripts/run_pilot.py`**

```python
# scripts/run_pilot.py
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

    first, last = CHAPTER_1_PAGE_RANGE
    raw = extract_pages(book["pdf_path"], first, last)
    cleaned = clean_text(raw, BOILERPLATE_PHRASES)
    chapters = split_into_chapters(cleaned, toc)

    chapter_number = 1
    chapter_name = next(
        t.chapter_name for t in toc if t.chapter_number == chapter_number
    )
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
        print(f"{book['pdf_path']}: wrote {len(records)} chunks to {out_dir / book['out_name']}")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run the pilot script**

```bash
"D:/Coding/edtech/.venv/Scripts/python" scripts/run_pilot.py
```

Expected: prints `data/raw/Class 5 EVS.pdf: wrote N chunks to data/interim/EVS_5_ch01.jsonl` and the equivalent line for the Science book, with no `AssertionError`. These in-script assertions are the automated half of the pilot check — they confirm every record has a valid `chunk_type` and no `[[PAGE:` or `**` artifacts leaked into `chunk_text` before any human reads the output.

- [x] **Step 3: Manually review the pilot output**

Read `data/interim/EVS_5_ch01.jsonl` and `data/interim/SCI_8_ch01.jsonl` chunk-by-chunk (e.g. `python -c "import json; [print(json.loads(l)) for l in open('data/interim/EVS_5_ch01.jsonl', encoding='utf-8')]"`) and check each chunk against:

1. **Reading order** — does `chunk_text` read as coherent, correctly-ordered prose, or is a sentence fragment from an image caption/sidebar interleaved mid-sentence?
2. **Noise** — any stray page numbers, footnote markers, or publisher boilerplate that slipped through `clean_text`?
3. **`chunk_type` correctness** — do `activity`/`recall`/`info_box`/`exercise` chunks actually contain that kind of content, and does `paragraph` only contain narrative prose?
4. **`topic` accuracy** — for EVS in particular, are the bold-label topics (e.g. "Stars", "Gravity") landing on the right paragraphs, or is the ≤6-word heuristic misfiring on non-topic bold text?

- [x] **Step 4: Write findings to `data/interim/REVIEW.md`**

Document, for each of the two chapters reviewed: chunk count, how many chunks had reading-order problems (with 1-2 concrete examples quoted), how many had noise leakage, whether `chunk_type` classification looked correct, and whether topic detection looked correct — including specific EVS bold-label examples that worked or misfired. End with an explicit go/no-go recommendation: is the pipeline ready to run across all 44 chapters as-is, or does a specific detector (name it) need a fix first?

- [x] **Step 5: Commit**

```bash
git add scripts/run_pilot.py data/interim/REVIEW.md
git commit -m "feat: add pilot run script and record pilot review findings"
```

Note: `data/interim/*.jsonl` itself stays untracked (already covered by `.gitignore`'s `data/interim/` rule) — only the script and the review notes are committed.

---

## Self-Review Notes

- **Spec coverage:** Extraction (PyMuPDF, reading-order) → Task 9. Cleaning (boilerplate/page-number/whitespace) → Task 3. Chapter detection via TOC → Tasks 4-5. Topic detection (numbered + bold-label) → Task 6. Block-type detection (activity/recall/info_box/exercise) → Task 6. Chunking with ~250-token paragraph target, never splitting special blocks → Task 8. Data model matching the spec's JSON shape → Task 2. Pilot on 1 chapter per book with the 4-point review checklist → Task 10. Formula-specific chunking and full 44-chapter rollout are explicitly out of scope per the spec and are not tasks here.
- **Type consistency:** `ChunkRecord.class_` (Task 2) matches the `class_` keyword argument used throughout `chunk.py` (Task 8) and `BookMeta.class_`. `Block`/`TopicSegment` field names from Task 7 (`chunk_type`, `text`, `page_start`, `page_end`, `topic`, `blocks`) are used identically in Task 8's `build_chunk_records`. `classify_bold_span`'s three-tuple return shape (`"topic"`/`"marker"`/`"emphasis"`) from Task 6 is consumed exactly that way in Task 7's `segment_chapter`.
- **No placeholders:** every step has runnable code or an exact command with expected output; the one inherently non-deterministic step (Task 10 Step 4, writing review findings) is scoped with a concrete checklist rather than left open-ended, since its content can only be known after the script actually runs.
