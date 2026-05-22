import sys
from pathlib import Path

import numpy as np

from rag_mcp.config import CONFIG, ROOT


def _add_native_path() -> None:
    candidates = [
        ROOT / "cpp" / "build" / "Release",
        ROOT / "cpp" / "build",
        ROOT / "python" / "rag_mcp",
    ]
    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path))


_add_native_path()

import rag_db_native


class NativeVectorStore:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index = rag_db_native.HnswIndex(
            CONFIG["embedding_dim"],
            CONFIG["hnsw_max_elements"],
            CONFIG["hnsw_m"],
            CONFIG["hnsw_ef_construction"],
            True,
        )
        if index_path.exists():
            self.index.load(str(index_path))

    def add(self, vector: np.ndarray) -> int:
        vec = np.asarray(vector, dtype=np.float32)
        return int(self.index.add(vec))

    def mark_deleted(self, label: int) -> None:
        self.index.mark_deleted(label)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        vec = np.asarray(vector, dtype=np.float32)
        labels, distances = self.index.search(vec, k, CONFIG["hnsw_ef_search"])
        return list(zip([int(x) for x in labels], [float(x) for x in distances]))

    def persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(str(self.index_path))
