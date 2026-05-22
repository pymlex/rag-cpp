from pathlib import Path

from transformers import AutoTokenizer

from config.settings import (
    CHUNK_TOKEN_OVERLAP,
    CHUNK_TOKEN_SIZE,
    EMBEDDING_MODEL_ID,
)


_TOKENIZER_MAX_LENGTH = 1024

_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID)
_tokenizer.model_max_length = _TOKENIZER_MAX_LENGTH
_tokenizer.clean_up_tokenization_spaces = False


def split_text_tokens(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    ids = _tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= chunk_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(ids):
        end = start + chunk_tokens
        piece = ids[start:end]
        chunks.append(_tokenizer.decode(piece, skip_special_tokens=True))
        if end >= len(ids):
            break
        start = end - overlap_tokens
    return chunks


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_file(path: Path, chunk_tokens: int, overlap_tokens: int) -> list[tuple[str, int]]:
    text = read_text_file(path)
    parts = split_text_tokens(text, chunk_tokens, overlap_tokens)
    return [(part, idx) for idx, part in enumerate(parts)]
