# Full 44-Chapter Rollout, Embedding, and pgvector Load — Design

## Context

The previous slice (`docs/superpowers/specs/2026-07-18-textbook-ingestion-cleaning-design.md`)
built and pilot-validated a cleaning/chunking pipeline against Chapter 1 of
both textbooks, with a "go" recommendation recorded in
`data/interim/REVIEW.md` after tightening the topic-detection heuristic.
This slice extends that pipeline three ways:

1. Run it across all chapters of both books (25 for EVS, 19 for Science),
   not just Chapter 1.
2. Generate embeddings for every chunk using `embeddinggemma` via a local
   Ollama server.
3. Load the chunks and their embeddings into a `pgvector`-backed table in
   the user's Supabase Postgres project.

Embedding generation and pgvector storage were explicitly out of scope for
the previous slice; this is where that boundary gets crossed.

## Verified facts

These were confirmed directly against the running tools/services before
this design was written, not assumed:

- **Ollama**: installed at version 0.32.1 (satisfies the model's
  `>=0.11.10` requirement). The `embeddinggemma` model (621 MB) is pulled
  and running locally.
- **Embedding dimension**: confirmed empirically via a live call to
  `http://localhost:11434/api/embed` — returns 768-dimensional vectors
  (the model's default; Matryoshka truncation to 512/256/128 exists but is
  not used here).
- **Document encoding prompt format** (from the model's Hugging Face card):
  `"title: {title | "none"} | text: {content}"`. This is the format for
  content being indexed for retrieval (as opposed to the `"task: search
  result | query: {content}"` format used for a user's search query at
  retrieval time — out of scope here, since this slice only indexes
  content).
- **Ollama's `/api/embed` endpoint** accepts either a single string or an
  array of strings as `"input"`, returning one embedding per input in the
  same order — batching per chapter (not per chunk) is both correct and
  efficient.
- **Supabase project** `aedsgktxvmddarqurbhi` (Postgres 17.6.1,
  ap-southeast-1) currently has zero tables in the `public` schema. The
  `vector` extension (pgvector 0.8.2) is available but not yet installed
  (`installed_version: null`).

## Scope of this slice

- Generalize the existing pilot script to process every chapter of both
  books (not just Chapter 1), reusing `extract.py`, `clean.py`,
  `structure.py`, and `chunk.py` unchanged.
- Add an embedding stage that calls Ollama and attaches a 768-element
  `embedding` field to every chunk record.
- Enable `pgvector` and create a `chunks` table in Supabase via a
  migration.
- Load all chunks (with embeddings) into that table via a direct
  `psycopg2` connection.

Explicitly **out of scope**: query-time retrieval logic (the
`"task: search result | query: ..."` prompt format, similarity search
queries, an API layer), the LMS/quiz layers, and any change to the
chunking/cleaning logic itself (it is reused as-is).

## Chapter-boundary extraction: avoiding a TOC/body collision

`split_into_chapters` finds each chapter's heading by regex-searching for
its number and title. If the searched text includes the table of contents
page (which lists every chapter's number and title, followed by a leader
of dots and a page number), the regex could match the *TOC entry* instead
of the *real body heading* — the TOC entry appears earlier in the document
and nothing in the current pattern rules it out.

This is avoided by keeping TOC parsing and body extraction as two separate
`extract_pages` calls, as the Chapter 1 pilot already did:

- TOC: `extract_pages(pdf_path, 9, 9)` → `parse_contents(...)` → the full
  list of `TocEntry` for the book.
- Body: `extract_pages(pdf_path, 10, 147)` → `clean_text(...)` →
  `split_into_chapters(cleaned, full_toc)` — since the TOC page itself is
  outside this range, only real headings can match.

Both books have 148 pages total (0-indexed 0-147), confirmed during the
previous slice, so page range `10-147` covers the entire body of both
books in one extraction call per book.

## Embedding stage (`src/ingestion/embed.py`)

```
def embed_texts(texts: list[str], titles: list[str], model: str = "embeddinggemma", host: str = "http://localhost:11434") -> list[list[float]]
```

Builds one `"title: {title} | text: {text}"` prompt per input (falling
back to `"title: none | text: {text}"` when there is no title), POSTs the
batch to `{host}/api/embed`, and returns the `embeddings` list in the same
order as the input. `title` per chunk is `topic` when present, else
`chapter_name` — giving the more specific label priority when one exists.

Called once per chapter (not once per chunk) to keep the number of HTTP
round-trips proportional to chapter count (~44) rather than chunk count
(~1,500+).

## Database schema

A single flat table, denormalized (matches the current `ChunkRecord`
shape exactly, plus the embedding column) rather than the fully normalized
Books/Chapters/Topics schema from the original architecture study — there
is no LMS/user layer yet to justify that normalization, and it can be
introduced later without touching this table's core columns.

```sql
create extension if not exists vector;

create table chunks (
  id bigint generated always as identity primary key,
  chunk_id text not null unique,
  board text not null,
  class text not null,
  subject text not null,
  book_title text not null,
  chapter_number int not null,
  chapter_name text not null,
  topic text,
  chunk_type text not null,
  page_start int not null,
  page_end int not null,
  chunk_text text not null,
  embedding vector(768) not null
);

create index on chunks using hnsw (embedding vector_cosine_ops);
```

The `hnsw` index on `embedding` is added upfront since the table is
write-once for this slice (bulk load, no incremental inserts to worry
about reindexing around), and cosine distance matches the similarity
metric `embeddinggemma`'s documentation recommends for retrieval.

`class` is a reserved word in Python but not in SQL, so the column is
named `class` directly (no `class_` translation needed at the DB layer —
that translation already happens in `ChunkRecord.to_dict()` before this
point).

## Load script (`scripts/load_to_supabase.py`)

Reads `data/processed/*.jsonl` (produced by the rollout+embedding stage),
connects via `psycopg2` using `DATABASE_URL` read from a `.env` file
(via `python-dotenv`, never hardcoded or logged), and batch-inserts rows
using `psycopg2.extras.execute_values` for efficiency at ~1,500+ rows.
`.env` is added to `.gitignore`; `.env.example` documents the expected
variable name with a placeholder value.

## Pipeline flow (end to end)

Steps 1-6 live in `scripts/run_full_pipeline.py` (generalizes
`scripts/run_pilot.py` to all chapters and adds the embedding call); steps
7-8 are the migration and `scripts/load_to_supabase.py` described above.

```
For each book (EVS, Science):
  1. Extract TOC (page 9) -> parse_contents -> full TocEntry list
  2. Extract body (pages 10-147) -> clean_text -> split_into_chapters(full toc)
  3. For each chapter in the book:
       segment_chapter -> build_chunk_records -> list[ChunkRecord]
  4. Collect all chapters' records into one list for the book
  5. embed_texts() on all chunk_texts (batched per chapter) -> attach `embedding`
  6. Write data/processed/{SUBJECT_CODE}_{class}_full.jsonl (chunk dict + embedding)

Then, once both books are written:
  7. Apply the pgvector migration (enable extension, create table+index)
  8. load_to_supabase.py reads both JSONL files and bulk-inserts into `chunks`
```

## Testing strategy

- `embed.py`: unit tests against the real local Ollama server (consistent
  with how `extract.py` is tested against real PDFs in this project) —
  verify a known input returns a 768-length vector, verify batch input
  order is preserved, verify the title-fallback behavior (topic vs
  chapter_name vs "none").
- Rollout generalization: a test verifying the full book extraction +
  chapter split produces the expected chapter count (25 for EVS, 19 for
  Science) with no `ValueError` from `split_into_chapters` — this is the
  regression test for the TOC/body collision risk described above.
- `load_to_supabase.py` is verified by running it against the real
  Supabase project and checking row counts via `mcp__plugin_supabase_supabase__execute_sql`
  (`select count(*) from chunks`), not a mocked DB — consistent with this
  project's existing preference for testing against real systems over
  mocks.

## Risks accepted for this slice

- **CPU embedding throughput**: ~1,500+ chunks through local CPU inference
  will take some minutes; no batching/parallelism beyond per-chapter
  batching is built for this slice given the one-time nature of this load.
- **Last-chapter trailing content**: extracting through page 147 may pull
  in back-matter after the final chapter's actual content into that
  chapter's last chunks. Accepted as a minor, bounded imperfection
  consistent with the rule-based-only approach already accepted in the
  previous slice.
