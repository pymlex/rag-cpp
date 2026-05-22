import uuid
from datetime import datetime, timezone
from pathlib import Path

from rag_mcp.bm25_store import Bm25Store
from rag_mcp.chunking import chunk_file
from rag_mcp.config import CONFIG
from rag_mcp.embeddings_client import EmbeddingsClient
from rag_mcp.file_scanner import current_file_state
from rag_mcp.manifest import ManifestStore, file_hash, utc_from_timestamp
from rag_mcp.native_store import NativeVectorStore
from rag_mcp.schemas import ChunkRecord, FileMeta


class RagIndexer:
    def __init__(self, corpus_root: Path):
        self.corpus_root = corpus_root.resolve()
        self.index_dir = self.corpus_root / CONFIG["index_subdir"]
        self.manifest = ManifestStore(self.index_dir / CONFIG["manifest_name"])
        self.bm25 = Bm25Store()
        bm25_path = self.index_dir / CONFIG["bm25_name"]
        if bm25_path.exists():
            self.bm25.load(bm25_path)
        self.vector_store = NativeVectorStore(self.index_dir / CONFIG["hnsw_name"])
        self.embedder = EmbeddingsClient()

    def sync(self) -> dict[str, int]:
        stats = {"added": 0, "updated": 0, "removed": 0}
        disk_state = current_file_state(self.corpus_root)
        known_paths = set(self.manifest.files.keys())
        disk_paths = set(disk_state.keys())
        for removed in known_paths - disk_paths:
            stats["removed"] += self._purge_file(removed)
        for rel, meta in disk_state.items():
            path = self.corpus_root / rel
            digest = meta["content_hash"]
            existing = self.manifest.files.get(rel)
            if existing is None:
                stats["added"] += self._index_file(rel, path, digest, meta)
            elif existing.content_hash != digest:
                stats["updated"] += self._reindex_file(rel, path, digest, meta)
        self._persist()
        return stats

    def _purge_file(self, rel: str) -> int:
        meta = self.manifest.files.get(rel)
        if meta is None:
            return 0
        labels = []
        for chunk_id in meta.chunk_ids:
            record = self.manifest.chunks.get(chunk_id)
            if record is not None:
                labels.append((chunk_id, record.label))
        chunk_ids = self.manifest.remove_file(rel)
        for chunk_id, label in labels:
            self.vector_store.mark_deleted(label)
            self.bm25.remove(chunk_id)
        return len(chunk_ids)

    def _reindex_file(self, rel: str, path: Path, digest: str, meta: dict) -> int:
        self._purge_file(rel)
        return self._index_file(rel, path, digest, meta)

    def _index_file(self, rel: str, path: Path, digest: str, meta: dict) -> int:
        chunks = chunk_file(path)
        if len(chunks) == 0:
            return 0
        vectors = self.embedder.embed(chunks)
        chunk_ids = []
        created = utc_from_timestamp(meta["created_at"])
        modified = utc_from_timestamp(meta["modified_at"])
        for idx, text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            label = self.vector_store.add(vectors[idx])
            record = ChunkRecord(
                chunk_id=chunk_id,
                label=label,
                file_path=rel,
                chunk_index=idx,
                text=text,
                content_hash=digest,
                created_at=created,
                modified_at=modified,
            )
            self.manifest.register_chunk(record)
            self.bm25.add(chunk_id, text)
            chunk_ids.append(chunk_id)
        self.manifest.files[rel] = FileMeta(
            file_path=rel,
            content_hash=digest,
            created_at=created,
            modified_at=modified,
            chunk_ids=chunk_ids,
        )
        return len(chunk_ids)

    def _persist(self) -> None:
        self.manifest.save()
        self.bm25.save(self.index_dir / CONFIG["bm25_name"])
        self.vector_store.persist()
