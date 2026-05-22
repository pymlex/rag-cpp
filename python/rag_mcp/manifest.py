import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_mcp.schemas import ChunkRecord, FileMeta


def utc_from_timestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def file_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


class ManifestStore:
    def __init__(self, path: Path):
        self.path = path
        self.files: dict[str, FileMeta] = {}
        self.chunks: dict[str, ChunkRecord] = {}
        self.label_to_chunk: dict[int, str] = {}
        if path.exists():
            self.load()

    def load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.files = {
            key: FileMeta.model_validate(value) for key, value in payload["files"].items()
        }
        self.chunks = {
            key: ChunkRecord.model_validate(value) for key, value in payload["chunks"].items()
        }
        self.label_to_chunk = {
            int(key): value for key, value in payload["label_to_chunk"].items()
        }

    def save(self) -> None:
        payload = {
            "files": {k: v.model_dump(mode="json") for k, v in self.files.items()},
            "chunks": {k: v.model_dump(mode="json") for k, v in self.chunks.items()},
            "label_to_chunk": {str(k): v for k, v in self.label_to_chunk.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def register_chunk(self, record: ChunkRecord) -> None:
        self.chunks[record.chunk_id] = record
        self.label_to_chunk[record.label] = record.chunk_id

    def remove_file(self, file_path: str) -> list[str]:
        meta = self.files.pop(file_path, None)
        if meta is None:
            return []
        removed = []
        for chunk_id in meta.chunk_ids:
            record = self.chunks.pop(chunk_id, None)
            if record is not None:
                self.label_to_chunk.pop(record.label, None)
                removed.append(chunk_id)
        return removed
