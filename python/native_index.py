from pathlib import Path

import numpy as np

from config.settings import (
    EMBEDDING_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    MAX_ELEMENTS,
)

import ragdb_native


def make_params():
    params = ragdb_native.HnswParams()
    params.dim = EMBEDDING_DIM
    params.max_elements = MAX_ELEMENTS
    params.M = HNSW_M
    params.M_max0 = HNSW_M * 2
    params.ef_construction = HNSW_EF_CONSTRUCTION
    params.ef_search = HNSW_EF_SEARCH
    return params


class NativeVectorIndex:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.params = make_params()
        self.index = ragdb_native.HnswIndex(self.params)
        if index_path.exists():
            self.index.load(str(index_path))

    def add(self, vector: np.ndarray) -> int:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        return int(self.index.add(vec))

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        hits = self.index.search(vec, int(top_k))
        return [(int(h.id), float(h.distance)) for h in hits]

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(str(self.index_path))

    def size(self) -> int:
        return int(self.index.size())
