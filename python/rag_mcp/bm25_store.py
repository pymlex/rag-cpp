import json
import math
import re
from collections import Counter
from pathlib import Path

from rag_mcp.config import CONFIG


TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class Bm25Store:
    def __init__(self):
        self.chunk_ids: list[str] = []
        self.doc_freq: Counter[str] = Counter()
        self.term_freq: list[Counter[str]] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.k1 = 1.5
        self.b = 0.75

    def add(self, chunk_id: str, text: str) -> None:
        tokens = tokenize(text)
        self.chunk_ids.append(chunk_id)
        tf = Counter(tokens)
        self.term_freq.append(tf)
        self.doc_len.append(len(tokens))
        for term in tf.keys():
            self.doc_freq[term] += 1
        n = len(self.doc_len)
        self.avgdl = sum(self.doc_len) / n if n else 0.0

    def remove(self, chunk_id: str) -> None:
        if chunk_id not in self.chunk_ids:
            return
        idx = self.chunk_ids.index(chunk_id)
        tf = self.term_freq[idx]
        for term in tf.keys():
            self.doc_freq[term] -= 1
            if self.doc_freq[term] <= 0:
                del self.doc_freq[term]
        del self.chunk_ids[idx]
        del self.term_freq[idx]
        del self.doc_len[idx]
        n = len(self.doc_len)
        self.avgdl = sum(self.doc_len) / n if n else 0.0

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        n = len(self.chunk_ids)
        if n == 0:
            return []
        scores = []
        for i in range(n):
            score = 0.0
            dl = self.doc_len[i]
            tf = self.term_freq[i]
            for term in q_tokens:
                if term not in self.doc_freq:
                    continue
                df = self.doc_freq[term]
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                freq = tf.get(term, 0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                score += idf * (freq * (self.k1 + 1.0)) / denom
            scores.append((self.chunk_ids[i], score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def save(self, path: Path) -> None:
        payload = {
            "chunk_ids": self.chunk_ids,
            "term_freq": [dict(counter) for counter in self.term_freq],
            "doc_len": self.doc_len,
            "doc_freq": dict(self.doc_freq),
            "avgdl": self.avgdl,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.chunk_ids = payload["chunk_ids"]
        self.term_freq = [Counter(item) for item in payload["term_freq"]]
        self.doc_len = payload["doc_len"]
        self.doc_freq = Counter(payload["doc_freq"])
        self.avgdl = payload["avgdl"]
