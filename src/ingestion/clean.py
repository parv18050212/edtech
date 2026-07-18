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
