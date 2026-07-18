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
