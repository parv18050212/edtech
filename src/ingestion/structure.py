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
