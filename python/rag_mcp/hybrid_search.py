from pathlib import Path

import numpy as np

from rag_mcp.bm25_store import Bm25Store
from rag_mcp.config import CONFIG
from rag_mcp.embeddings_client import EmbeddingsClient
from rag_mcp.manifest import ManifestStore
from rag_mcp.native_store import NativeVectorStore
from rag_mcp.schemas import SearchResult


class HybridSearcher:
    def __init__(self, corpus_root: Path):
        self.corpus_root = corpus_root.resolve()
        self.index_dir = self.corpus_root / CONFIG["index_subdir"]
        self.manifest = ManifestStore(self.index_dir / CONFIG["manifest_name"])
        self.bm25 = Bm25Store()
        self.bm25.load(self.index_dir / CONFIG["bm25_name"])
        self.vector_store = NativeVectorStore(self.index_dir / CONFIG["hnsw_name"])
        self.embedder = EmbeddingsClient()

    def search(self, query: str) -> list[SearchResult]:
        query_vec = self.embedder.embed([query])[0]
        vector_hits = self.vector_store.search(query_vec, CONFIG["vector_top_k"])
        bm25_hits = self.bm25.search(query, CONFIG["bm25_top_k"])
        fused = self._rrf(vector_hits, bm25_hits)
        results = []
        for chunk_id, fused_score, vector_score, bm25_score in fused[: CONFIG["fusion_top_k"]]:
            record = self.manifest.chunks[chunk_id]
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    file_path=record.file_path,
                    text=record.text,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    fused_score=fused_score,
                    modified_at=record.modified_at,
                )
            )
        return results

    def _rrf(
        self,
        vector_hits: list[tuple[int, float]],
        bm25_hits: list[tuple[str, float]],
    ) -> list[tuple[str, float, float | None, float | None]]:
        scores: dict[str, float] = {}
        vector_map: dict[str, float] = {}
        bm25_map: dict[str, float] = {}
        k = CONFIG["rrf_k"]
        for rank, (label, dist) in enumerate(vector_hits):
            chunk_id = self.manifest.label_to_chunk.get(label)
            if chunk_id is None:
                continue
            vector_map[chunk_id] = dist
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, (chunk_id, bm25_score) in enumerate(bm25_hits):
            bm25_map[chunk_id] = bm25_score
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        output = []
        for chunk_id, fused in ordered:
            output.append(
                (
                    chunk_id,
                    fused,
                    vector_map.get(chunk_id),
                    bm25_map.get(chunk_id),
                )
            )
        return output
