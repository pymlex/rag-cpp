from pathlib import Path

from rag_mcp.config import CONFIG


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def chunk_file(path: Path) -> list[str]:
    text = read_text_file(path)
    return split_text(
        text,
        CONFIG["chunk_size_chars"],
        CONFIG["chunk_overlap_chars"],
    )
