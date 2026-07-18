from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ChunkRecord:
    chunk_id: str
    board: str
    class_: str
    subject: str
    book_title: str
    chapter_number: int
    chapter_name: str
    topic: Optional[str]
    chunk_type: str
    page_start: int
    page_end: int
    chunk_text: str
    embedding: Optional[list] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("class_")
        return d
