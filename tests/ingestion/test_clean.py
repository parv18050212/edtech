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
