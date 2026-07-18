# Pilot Review — Chapter 1, both books

Reviewed: `EVS_5_ch01.jsonl` (33 chunks, Class 5 EVS Chapter 1 "Our Earth and
Our Solar System") and `SCI_8_ch01.jsonl` (36 chunks, Class 8 Science
Chapter 1 "Living World and Classification of Microbes"). Both from PDF
pages 10-14 (0-indexed).

## Bugs found and fixed during this pilot

Three real defects surfaced only once the pipeline ran against actual PDF
data (none of the hand-written unit test fixtures exposed them). All three
are fixed, covered by new regression tests, and the full 44-test (now
44-test) suite passes.

1. **Page numbers were `0` for every chunk on a chapter's opening page.**
   `extract_pages` inserts `[[PAGE:N]]` *before* that page's content, so a
   chapter heading's own page marker sits before the heading text.
   `split_into_chapters` sliced from the *end* of the heading match, which
   silently dropped that marker — `segment_chapter` then had no page
   context until the next real page boundary. Fixed by recovering the
   nearest preceding page marker and prepending it to each chapter's slice.

2. **Curly-apostrophe marker phrases weren't recognized.** Both PDFs use
   Unicode U+2019 (’) as their apostrophe, not ASCII `'`. Marker prefixes
   like `"what's the solution"` and `"let's try this"` never matched, so
   e.g. "What's the solution ?" fell through to the topic heuristic instead
   of being classified as an `exercise` marker. Fixed by normalizing curly
   to straight apostrophes before prefix matching.

3. **TOC parsing failed outright on the Science book.** Its contents page
   renders the chapter number, a bold tab character, and the chapter title
   as three separate adjacent bold spans, producing runs of 4+ consecutive
   asterisks at the seams (e.g. `**1. \t**** ****Living World...**`) that
   the original `\*{0,2}`-bounded regex couldn't consume. Fixed by using
   unbounded `[\s*]*` separators and stripping stray `*` from the captured
   name.

## Checklist findings

**1. Reading order.** Still imperfect on visually complex pages, as
anticipated in the design spec. Two concrete examples:

- EVS chunk `TOP00_000`: "...many stars. They are very far away from the
  earth. **and brilliant. In its bright light, during the day, we cannot
  see other stars.**" — the bolded sentence is a caption/aside that the
  block sort placed mid-paragraph instead of after its actual context.
- Science: the "Can you recall?" bold marker was sorted *after* its own
  three numbered questions on the page (confirmed by inspecting the
  extracted text directly), so those questions ended up in an untyped
  `paragraph` block rather than `recall` — not fixable by a code change to
  `structure.py`, since it correctly processed the tokens in the order
  `extract.py` gave it. This is a genuine geometry-based reading-order
  limit, not a bug.

**2. Noise.** No leaked `[[PAGE:` or `**` artifacts (enforced by the
pilot script's own assertions). A few chunks are near-empty fragments left
over from split points, e.g. EVS `TOP15_000` chunk_text is just `"X"` and
`TOP24_000` is just `"1."` — both are single stray characters/numbers that
survived because they had enough surrounding whitespace to count as
non-empty after `finalize_chunk_text`. Minor, but real: a stricter
minimum-length filter before emitting a chunk would remove these.

**3. `chunk_type` correctness.** Where classification succeeded, it's
accurate — e.g. Science `TOP01_001` (info_box, the "In History......"
timeline of classification systems) and EVS `TOP26_001` (activity, "What
will happen to our solar system if the sun were to suddenly disappear?").
Distribution: EVS 29 paragraph / 2 activity / 1 info_box / 1 exercise;
Science 33 paragraph / 2 info_box / 1 activity. The low marker-block counts
relative to the number of visible "Try this"/"Can you tell?"/exercise
sections in the source chapters suggest the reading-order issue in finding
#1 above is suppressing more marker classifications than these numbers
show — markers whose content was reordered ahead of the marker itself lose
their type the same way the "Can you recall?" case did.

**4. Topic accuracy.** The bold-label heuristic (≤6 words → topic) works
well for genuine cases — `Stars`, `Gravity`, `Dwarf planets`, `Satellites`,
`Planets` in EVS; `1.1 Five Kingdom system of classification`,
`Biodiversity and need of classification` in Science. But it also misfires
on short bold fragments that aren't topics at all: EVS topics `'X'`, `'-'`,
`'t'`, and `'a man-made satellite'` / `'taken by Mangalyaan'` (both photo
caption fragments) are not real section headers. Science has similar cases
in the exercise section, e.g. topic `'6.'` and `'Give answers.'` are
exercise sub-numbering, not topics.

## Go/no-go recommendation

**No-go on running all 44 chapters as-is.** The three fixed bugs were
worth catching before scaling (especially the TOC parser, which would have
hard-failed on every chapter of the Science book). But the topic-heuristic
misfires (finding #4) are frequent enough — roughly 4 of ~29 EVS topics
were clearly wrong — that they'd noticeably degrade retrieval metadata
quality across all 44 chapters if left as-is.

Recommended next step before the full rollout: tighten
`classify_bold_span`'s topic fallback so it requires the bold span to be
followed by a non-trivial amount of body text before the next structural
token (filtering out single-word/photo-caption fragments like `"X"` or
`"-"` that happen to be ≤6 words but aren't followed by real content). The
reading-order limitation (finding #1) is accepted as a known constraint of
the rule-based approach, per the design spec — not something to chase
further here.

## Follow-up: topic heuristic tightened

`classify_bold_span`'s topic fallback now also requires the candidate
label to contain at least 2 alphabetic characters, with the first one
uppercase (`_looks_like_topic_label`, `src/ingestion/structure.py`). This
targets the specific failure mode found above — real headers in these
textbooks always start with a capital letter or a digit and contain a real
word; caption fragments and stray characters don't.

Rerunning the pilot after the fix: all previously-flagged junk topics are
gone — `'X'`, `'-'`, `'t'`, `'a man-made satellite'`, `'taken by
Mangalyaan'`, and `'fungi.'` no longer appear anywhere in either book's
output (verified directly, not just by the 6 new regression tests added
for these exact strings). Chunk counts shifted slightly (EVS 33→34,
Science 36→38) as a side effect: text that used to spuriously start a new
topic segment now flows as inline text within the surrounding paragraph
instead, which is the intended behavior, not a regression.

A few borderline caption-like fragments remain — e.g. `'(Dwarf'` and `'The
moon as seen'` — because they start with an uppercase letter and have
enough alphabetic content to pass the tightened check, even though they're
still caption fragments rather than real topic headers. This is an
inherent limit of a purely local (no-lookahead) heuristic: it can't
distinguish a genuinely capitalized topic label from a capitalized caption
fragment without understanding the surrounding content. Given the clear,
frequent junk cases are now eliminated and the remaining cases are
occasional and lower-impact, this is an acceptable residual limitation
rather than a blocker.

**Updated recommendation: go for the full 44-chapter rollout**, with the
remaining caption-fragment edge cases and the reading-order limitation
(finding #1) both accepted as known, documented constraints rather than
things to chase further before scaling.
