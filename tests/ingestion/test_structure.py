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
