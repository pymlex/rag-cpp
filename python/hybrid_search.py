from collections import defaultdict

import numpy as np

from config.settings import (
    HYBRID_BM25_WEIGHT,
    HYBRID_CANDIDATE_MULT,
    HYBRID_RRF_K,
    HYBRID_VECTOR_WEIGHT,
    MAX_CHUNKS_PER_FILE,
    RERANK_OVERLAP_WEIGHT,
)
from python.bm25_index import Bm25Index, tokenize
from python.metadata_store import ChunkRecord, MetadataStore
from python.native_index import NativeVectorIndex
from python.path_utils import canonical_file_key


def min_max_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float64)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def rrf_contribution(rank: int, k: int) -> float:
    return 1.0 / (k + rank + 1)


def query_token_overlap(query_text: str, chunk_text: str) -> float:
    query_tokens = set(tokenize(query_text))
    chunk_tokens = set(tokenize(chunk_text))
    if not query_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / len(query_tokens)


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
        pool = max(top_k * HYBRID_CANDIDATE_MULT, top_k + 8)
        vec_hits = self.vector_index.search(query_vector, pool)
        vec_rank: dict[str, int] = {}
        vec_scores: dict[str, float] = {}
        for rank, (vid, dist) in enumerate(vec_hits):
            cid = self.vector_id_to_chunk.get(vid)
            if cid is None or cid in vec_rank:
                continue
            vec_rank[cid] = rank
            vec_scores[cid] = 1.0 - float(dist)

        bm25_hits = self.bm25.search(query_text, pool)
        bm25_rank: dict[str, int] = {}
        bm25_scores: dict[str, float] = {}
        for rank, (cid, score) in enumerate(bm25_hits):
            if cid in bm25_rank:
                continue
            bm25_rank[cid] = rank
            bm25_scores[cid] = score

        fused: dict[str, float] = defaultdict(float)
        for cid, rank in vec_rank.items():
            fused[cid] += HYBRID_VECTOR_WEIGHT * rrf_contribution(rank, HYBRID_RRF_K)
        for cid, rank in bm25_rank.items():
            fused[cid] += HYBRID_BM25_WEIGHT * rrf_contribution(rank, HYBRID_RRF_K)

        vec_n = min_max_norm(vec_scores)
        bm25_n = min_max_norm(bm25_scores)
        for cid in set(vec_n) | set(bm25_n):
            fused[cid] += 0.15 * vec_n.get(cid, 0.0) + 0.15 * bm25_n.get(cid, 0.0)

        candidate_ids = [cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[: pool]]
        records = self.metadata.list_chunks()
        by_id = {r.chunk_id: r for r in records}
        rerank_pool = [by_id[cid] for cid in candidate_ids if cid in by_id]

        scored: list[tuple[float, ChunkRecord]] = []
        for record in rerank_pool:
            base = fused[record.chunk_id]
            overlap = query_token_overlap(query_text, record.text)
            scored.append((base + RERANK_OVERLAP_WEIGHT * overlap, record))
        scored.sort(key=lambda x: x[0], reverse=True)

        selected: list[ChunkRecord] = []
        per_file: dict[str, int] = defaultdict(int)
        for _, record in scored:
            file_key = canonical_file_key(record.file_path)
            if per_file[file_key] >= MAX_CHUNKS_PER_FILE:
                continue
            selected.append(record)
            per_file[file_key] += 1
            if len(selected) >= top_k:
                break
        return selected
