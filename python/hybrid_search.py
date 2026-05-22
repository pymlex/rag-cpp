from collections import defaultdict

import numpy as np

from config.settings import HYBRID_BM25_WEIGHT, HYBRID_VECTOR_WEIGHT
from python.bm25_index import Bm25Index
from python.metadata_store import ChunkRecord, MetadataStore
from python.native_index import NativeVectorIndex


def min_max_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float64)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridSearcher:
    def __init__(
        self,
        vector_index: NativeVectorIndex,
        bm25: Bm25Index,
        metadata: MetadataStore,
        vector_id_to_chunk: dict[int, str],
    ):
        self.vector_index = vector_index
        self.bm25 = bm25
        self.metadata = metadata
        self.vector_id_to_chunk = vector_id_to_chunk

    def search(self, query_vector: np.ndarray, query_text: str, top_k: int) -> list[ChunkRecord]:
        vec_hits = self.vector_index.search(query_vector, top_k * 3)
        vec_scores: dict[str, float] = {}
        for vid, dist in vec_hits:
            cid = self.vector_id_to_chunk.get(vid)
            if cid is None:
                continue
            vec_scores[cid] = 1.0 - float(dist)
        bm25_hits = self.bm25.search(query_text, top_k * 3)
        bm25_scores = {cid: score for cid, score in bm25_hits}
        vec_n = min_max_norm(vec_scores)
        bm25_n = min_max_norm(bm25_scores)
        merged: dict[str, float] = defaultdict(float)
        for cid, s in vec_n.items():
            merged[cid] += HYBRID_VECTOR_WEIGHT * s
        for cid, s in bm25_n.items():
            merged[cid] += HYBRID_BM25_WEIGHT * s
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        chunk_ids = [cid for cid, _ in ranked]
        records = self.metadata.list_chunks()
        by_id = {r.chunk_id: r for r in records}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]
