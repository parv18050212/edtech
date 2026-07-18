import re
from dataclasses import dataclass

from ingestion.schema import ChunkRecord

ARTIFACT_RE = re.compile(r"\[\[PAGE:\d+\]\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    words = len(text.split())
    if words == 0:
        return 0
    return max(1, round(words / 0.75))


def finalize_chunk_text(text: str) -> str:
    text = ARTIFACT_RE.sub(" ", text)
    text = text.replace("**", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_paragraph(text: str, target_tokens: int = 250) -> list[str]:
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text.strip()) if s]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > target_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def make_chunk_id(
    subject_code: str, class_: str, chapter_number: int, topic_index: int, seq: int
) -> str:
    return (
        f"MSB_{subject_code}{class_}_CH{chapter_number:02d}"
        f"_TOP{topic_index:02d}_{seq:03d}"
    )


@dataclass
class BookMeta:
    board: str
    class_: str
    subject: str
    subject_code: str
    book_title: str


def build_chunk_records(
    book_meta: BookMeta,
    chapter_number: int,
    chapter_name: str,
    topic_segments: list,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for topic_index, segment in enumerate(topic_segments):
        seq = 0
        for block in segment.blocks:
            if block.chunk_type == "paragraph":
                sub_texts = split_paragraph(block.text)
            else:
                cleaned = block.text.strip()
                sub_texts = [cleaned] if cleaned else []

            for sub_text in sub_texts:
                final_text = finalize_chunk_text(sub_text)
                if not final_text:
                    continue
                records.append(
                    ChunkRecord(
                        chunk_id=make_chunk_id(
                            book_meta.subject_code,
                            book_meta.class_,
                            chapter_number,
                            topic_index,
                            seq,
                        ),
                        board=book_meta.board,
                        class_=book_meta.class_,
                        subject=book_meta.subject,
                        book_title=book_meta.book_title,
                        chapter_number=chapter_number,
                        chapter_name=chapter_name,
                        topic=segment.topic,
                        chunk_type=block.chunk_type,
                        page_start=block.page_start,
                        page_end=block.page_end,
                        chunk_text=final_text,
                    )
                )
                seq += 1
    return records
