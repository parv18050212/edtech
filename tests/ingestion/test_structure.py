import pytest

from ingestion.structure import (
    TocEntry,
    classify_bold_span,
    parse_contents,
    segment_chapter,
    split_into_chapters,
)

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


def test_parse_contents_tolerates_fragmented_adjacent_bold_spans():
    # Real extracted text from Science Class 8's TOC page: the "N." prefix,
    # a bold tab-only span, and the title render as three separate adjacent
    # bold spans, producing runs of 4+ asterisks at the seams.
    text = "**1. \t**** ****Living World and Classification of Microbes**............. 1"
    entries = parse_contents(text)
    assert entries[0] == TocEntry(
        1, "Living World and Classification of Microbes", 1
    )


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


def test_split_into_chapters_matches_heading_case_insensitively():
    # Real case found in Science Class 8: the TOC lists "Composition of
    # Matter" but the actual body heading reads "Composition of matter"
    # (lowercase 'm') -- a publisher inconsistency, not something we can
    # assume away.
    toc = [TocEntry(6, "Composition of Matter", 39)]
    full_text = (
        "**6. Composition of matter**\n"
        "What are the various states of matter?\n"
    )
    chapters = split_into_chapters(full_text, toc)
    assert "What are the various states of matter?" in chapters[6]


def test_split_into_chapters_matches_ampersand_for_and():
    # Real case found in Science Class 8: the TOC lists "Introduction to
    # Acid and Base" but the actual body heading reads "Introduction to
    # Acid & Base" -- another publisher inconsistency between the TOC and
    # body text.
    toc = [TocEntry(12, "Introduction to Acid and Base", 83)]
    full_text = (
        "**12. Introduction to Acid & Base**\n"
        "You will notice that some substances have sweet taste.\n"
    )
    chapters = split_into_chapters(full_text, toc)
    assert "You will notice that some substances have sweet taste." in chapters[12]


def test_split_into_chapters_tolerates_fragmented_heading_bold_spans():
    # Real case found in Class 5 EVS: the body heading for chapter 9 renders
    # as three separate adjacent bold spans split around the hyphen --
    # "**9. Maps ****-**** our Companions**" -- the same class of PyMuPDF
    # span-fragmentation issue already handled for TOC parsing, but this
    # time inside a chapter heading itself.
    toc = [TocEntry(9, "Maps - our Companions", 39)]
    full_text = (
        "**9. Maps ****-**** our Companions**\n"
        "The landscape around us is made up of many features.\n"
    )
    chapters = split_into_chapters(full_text, toc)
    assert "The landscape around us is made up of many features." in chapters[9]


def test_split_into_chapters_raises_if_heading_not_found():
    toc = [TocEntry(1, "A Chapter That Does Not Exist", 1)]
    with pytest.raises(ValueError):
        split_into_chapters("no matching heading here", toc)


def test_split_into_chapters_recovers_page_marker_preceding_the_heading():
    # extract_pages() puts the page marker BEFORE that page's blocks, so the
    # heading's own page marker sits before the heading text -- slicing from
    # the end of the heading match must not lose it, or every chunk on the
    # chapter's opening page ends up with no page number at all.
    toc = [TocEntry(1, "Our Earth and Our Solar System", 1)]
    full_text = (
        "[[PAGE:10]]\n"
        "**1. Our Earth and Our Solar System**\n"
        "The sun and the moon are close to earth.\n"
    )
    chapters = split_into_chapters(full_text, toc)
    assert chapters[1].startswith("[[PAGE:10]]")


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


def test_classify_bold_span_handles_curly_apostrophe():
    # The source PDFs use U+2019 (curly apostrophe) as their apostrophe
    # character, not ASCII "'".
    assert classify_bold_span("What’s the solution ?") == ("marker", "exercise")
    assert classify_bold_span("Let’s try this") == ("marker", "activity")


@pytest.mark.parametrize(
    "bold_text",
    [
        "-",  # bare punctuation, no letters at all
        "t",  # single letter, likely a rendering artifact
        "X",  # single letter
        "a man-made satellite",  # lowercase-starting caption fragment
        "taken by Mangalyaan",  # lowercase-starting caption fragment
        "fungi.",  # lowercase-starting sentence-ending fragment
    ],
)
def test_classify_bold_span_rejects_non_topic_fragments(bold_text):
    # These are all real misfires found during the Chapter 1 pilot review
    # (data/interim/REVIEW.md): short bold fragments that satisfy the old
    # <=6-word heuristic but aren't real topic headers. Real headers in
    # these textbooks always start with a capital letter or a digit and
    # contain at least one real word.
    assert classify_bold_span(bold_text) == ("emphasis", bold_text)


@pytest.mark.parametrize(
    "bold_text,expected_label",
    [
        ("Mangalyaan", "Mangalyaan"),
        ("Satellites", "Satellites"),
        ("1. Bacteria", "1. Bacteria"),
        ("3.Fungi-", "3.Fungi-"),
    ],
)
def test_classify_bold_span_still_accepts_real_topic_labels(bold_text, expected_label):
    # Regression guard: the tightened heuristic must not reject legitimate
    # short topic headers, including ones that start with a digit (numbered
    # sub-topic lists like "1. Bacteria", "2. Protozoa" seen in the Science
    # book pilot output).
    assert classify_bold_span(bold_text) == ("topic", expected_label)


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
