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
