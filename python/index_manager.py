from datetime import datetime
from pathlib import Path

import numpy as np

from config.settings import (
    BM25_FILE,
    CHUNK_TOKEN_OVERLAP,
    CHUNK_TOKEN_SIZE,
    EMBEDDING_DIM,
    HNSW_FILE,
    INDEX_SUBDIR,
    TEXT_EXTENSIONS,
)
from python.bm25_index import Bm25Index
from python.chunking import chunk_file
from python.corpus_paths import get_corpus_root
from python.embeddings_client import EmbeddingsClient
from python.file_scanner import file_hash, file_times, iter_text_files
from python.metadata_store import ChunkRecord, MetadataStore
from python.native_index import NativeVectorIndex


class IndexManager:
    def __init__(self, corpus_root: Path | None = None):
        self.corpus_root = (corpus_root or get_corpus_root()).resolve()
        self.index_dir = self.corpus_root / INDEX_SUBDIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.meta = MetadataStore(self.index_dir / META_DB)
        self.vector_index = NativeVectorIndex(self.index_dir / HNSW_FILE)
        self.bm25 = Bm25Index()
        self.embedder = EmbeddingsClient()
        self.vector_id_to_chunk: dict[int, str] = {}
        corpus = self.meta.export_corpus()
        if corpus:
            self.bm25.load_or_build(self.index_dir / BM25_FILE, corpus)
        self._refresh_mapping()

    def _chunk_id(self, rel_path: str, chunk_index: int) -> str:
        return f"{rel_path}::{chunk_index}"

    def _refresh_mapping(self) -> None:
        self.vector_id_to_chunk = {}
        for row in self.meta.list_chunks():
            if row.vector_id is not None:
                self.vector_id_to_chunk[int(row.vector_id)] = row.chunk_id

    def sync(self) -> dict[str, int]:
        discovered = {
            str(p.relative_to(self.corpus_root)): p
            for p in iter_text_files(self.corpus_root, TEXT_EXTENSIONS)
            if INDEX_SUBDIR not in p.parts
        }
        indexed_paths = self.meta.indexed_file_paths()
        stats = {"added": 0, "updated": 0, "removed": 0, "rebuilt": 0}
        for stale in indexed_paths - set(discovered.keys()):
            stats["removed"] += self._purge_file_meta(stale)
        for rel, path in discovered.items():
            digest = file_hash(path)
            prev = self.meta.get_file_hash(rel)
            if prev == digest:
                continue
            if prev is not None:
                stats["updated"] += self._purge_file_meta(rel)
            stats["added"] += self._index_file(rel, path, digest)
        if stats["removed"] > 0 or stats["updated"] > 0:
            self._rebuild_hnsw_hard()
            stats["rebuilt"] = 1
        corpus = self.meta.export_corpus()
        self.bm25.build(corpus)
        self.bm25.save(self.index_dir / BM25_FILE)
        self.vector_index.save()
        self._refresh_mapping()
        return stats

    def _purge_file_meta(self, rel_path: str) -> int:
        return self.meta.delete_file_chunks(rel_path)

    def _rebuild_hnsw_hard(self) -> None:
        hnsw_path = self.index_dir / HNSW_FILE
        if hnsw_path.exists():
            hnsw_path.unlink()
        self.vector_index = NativeVectorIndex(hnsw_path)
        records = self.meta.list_chunks_with_vectors()
        for record in records:
            vec = np.frombuffer(record.vector_blob, dtype=np.float32).reshape(EMBEDDING_DIM)
            vector_id = self.vector_index.add(vec)
            record.vector_id = vector_id
            self.meta.upsert_chunk(record)

    def _index_file(self, rel_path: str, path: Path, digest: str) -> int:
        created, modified = file_times(path)
        self.meta.set_file_hash(rel_path, digest, created, modified)
        chunks = chunk_file(path, CHUNK_TOKEN_SIZE, CHUNK_TOKEN_OVERLAP)
        texts = [c[0] for c in chunks]
        vectors = self.embedder.embed(texts)
        now = datetime.utcnow().isoformat()
        count = 0
        for (text, chunk_index), vec in zip(chunks, vectors):
            chunk_id = self._chunk_id(rel_path, chunk_index)
            vector_id = self.vector_index.add(vec)
            blob = np.asarray(vec, dtype=np.float32).tobytes()
            record = ChunkRecord(
                chunk_id=chunk_id,
                file_path=rel_path,
                chunk_index=chunk_index,
                text=text,
                vector_id=vector_id,
                vector_blob=blob,
                created_at=now,
                modified_at=now,
                file_created_at=created,
                file_modified_at=modified,
                content_hash=digest,
            )
            self.meta.upsert_chunk(record)
            self.vector_id_to_chunk[vector_id] = chunk_id
            count += 1
        return count

    def close(self) -> None:
        self.meta.close()
