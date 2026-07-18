import fitz

BOLD_FLAG = 16


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & BOLD_FLAG) or "bold" in span["font"].lower()


def _block_sort_key(block: dict) -> tuple[float, float]:
    x0, y0, _x1, _y1 = block["bbox"]
    return (round(y0 / 10) * 10, x0)


def _block_to_marked_text(block: dict) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line["spans"]:
            text = span["text"]
            if not text:
                continue
            if _is_bold(span):
                parts.append(f"**{text}**")
            else:
                parts.append(text)
        parts.append("\n")
    return "".join(parts)


def extract_pages(pdf_path: str, first_page: int, last_page: int) -> str:
    doc = fitz.open(pdf_path)
    try:
        out: list[str] = []
        for page_number in range(first_page, last_page + 1):
            page = doc[page_number]
            out.append(f"[[PAGE:{page_number}]]")
            blocks = [
                b for b in page.get_text("dict")["blocks"] if "lines" in b
            ]
            blocks.sort(key=_block_sort_key)
            for block in blocks:
                out.append(_block_to_marked_text(block))
        return "\n".join(out)
    finally:
        doc.close()
