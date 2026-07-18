import re
from dataclasses import dataclass
from typing import Optional

TOC_ENTRY_RE = re.compile(
    r"[\s*]*(\d{1,2})\.[\s*]*(.+?)[\s*]*\.{2,}[\s*]*(\d{1,3})[\s*]*"
)

PAGE_MARKER_RE = re.compile(r"\[\[PAGE:\d+\]\]")


@dataclass
class TocEntry:
    chapter_number: int
    chapter_name: str
    start_page: int


def parse_contents(contents_text: str) -> list[TocEntry]:
    entries = []
    for match in TOC_ENTRY_RE.finditer(contents_text):
        number = int(match.group(1))
        # The TOC page can render "N." and the title as separate adjacent
        # bold spans (each independently **-wrapped by extract.py), so the
        # lazily-captured name can contain stray '*' runs at the seams
        # between spans (e.g. "**** ****Living World..."). Strip them.
        name = re.sub(r"\*+", "", match.group(2))
        name = re.sub(r"\s+", " ", name).strip()
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
    for i, (chapter_number, start, end) in enumerate(matches):
        next_start = matches[i + 1][1] if i + 1 < len(matches) else len(full_text)
        # The heading's own page marker (e.g. "[[PAGE:10]]") appears BEFORE
        # the heading text in full_text, so slicing from `end` would silently
        # drop it -- leaving segment_chapter with no page context for the
        # chapter's opening content. Recover the most recent page marker
        # before the heading and prepend it to the slice.
        preceding_pages = list(PAGE_MARKER_RE.finditer(full_text[:start]))
        prefix = preceding_pages[-1].group(0) + "\n" if preceding_pages else ""
        chapters[chapter_number] = prefix + full_text[end:next_start]
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
    # These textbooks use a Unicode right single quotation mark (U+2019) as
    # their apostrophe, not ASCII "'" -- normalize to ASCII so prefixes like
    # "what's the solution" match regardless of which one the PDF used.
    normalized = normalized.replace("’", "'")
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
