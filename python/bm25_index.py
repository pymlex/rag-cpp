import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[а-яёa-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class Bm25Index:
    def __init__(self):
        self.chunk_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []
        self.model: BM25Okapi | None = None

    def build(self, corpus: list[dict]) -> None:
        self.chunk_ids = [row["chunk_id"] for row in corpus]
        self.corpus_tokens = [tokenize(row["text"]) for row in corpus]
        self.model = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self.model is None:
            return []
        scores = self.model.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[int(i)], float(scores[int(i)])) for i in order]

    def save(self, path: Path) -> None:
        payload = {
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load_or_build(self, path: Path, corpus: list[dict]) -> None:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.chunk_ids = data["chunk_ids"]
            self.corpus_tokens = data["corpus_tokens"]
            self.model = BM25Okapi(self.corpus_tokens)
            return
        self.build(corpus)
        self.save(path)
