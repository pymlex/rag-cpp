import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ChunkRecord:
    chunk_id: str
    file_path: str
    chunk_index: int
    text: str
    vector_id: int | None
    vector_blob: bytes | None
    created_at: str
    modified_at: str
    file_created_at: str
    file_modified_at: str
    content_hash: str


class MetadataStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector_id INTEGER,
                vector_blob BLOB,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                file_created_at TEXT NOT NULL,
                file_modified_at TEXT NOT NULL,
                content_hash TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                file_created_at TEXT NOT NULL,
                file_modified_at TEXT NOT NULL
            )
            """
        )
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(chunks)").fetchall()}
        if "vector_blob" not in columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN vector_blob BLOB")
        self.conn.commit()

    def upsert_chunk(self, record: ChunkRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, file_path, chunk_index, text, vector_id, vector_blob,
                created_at, modified_at, file_created_at, file_modified_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                text=excluded.text,
                vector_id=excluded.vector_id,
                vector_blob=excluded.vector_blob,
                modified_at=excluded.modified_at,
                file_created_at=excluded.file_created_at,
                file_modified_at=excluded.file_modified_at,
                content_hash=excluded.content_hash
            """,
            (
                record.chunk_id,
                record.file_path,
                record.chunk_index,
                record.text,
                record.vector_id,
                record.vector_blob,
                record.created_at,
                record.modified_at,
                record.file_created_at,
                record.file_modified_at,
                record.content_hash,
            ),
        )
        self.conn.commit()

    def delete_file_chunks(self, file_path: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        count = int(row[0])
        self.conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self.conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
        self.conn.commit()
        return count

    def _row_to_record(self, row: tuple) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row[0],
            file_path=row[1],
            chunk_index=row[2],
            text=row[3],
            vector_id=row[4],
            vector_blob=row[5],
            created_at=row[6],
            modified_at=row[7],
            file_created_at=row[8],
            file_modified_at=row[9],
            content_hash=row[10],
        )

    def list_chunks(self) -> list[ChunkRecord]:
        rows = self.conn.execute(
            "SELECT * FROM chunks ORDER BY file_path, chunk_index"
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_chunks_with_vectors(self) -> list[ChunkRecord]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE vector_blob IS NOT NULL ORDER BY file_path, chunk_index"
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def set_file_hash(self, file_path: str, content_hash: str, created: str, modified: str) -> None:
        self.conn.execute(
            """
            INSERT INTO files (file_path, content_hash, file_created_at, file_modified_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                content_hash=excluded.content_hash,
                file_modified_at=excluded.file_modified_at
            """,
            (file_path, content_hash, created, modified),
        )
        self.conn.commit()

    def get_file_hash(self, file_path: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_hash FROM files WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        return row[0] if row else None

    def export_corpus(self) -> list[dict]:
        rows = self.conn.execute("SELECT chunk_id, text FROM chunks").fetchall()
        return [{"chunk_id": r[0], "text": r[1]} for r in rows]

    def indexed_file_paths(self) -> set[str]:
        rows = self.conn.execute("SELECT DISTINCT file_path FROM chunks").fetchall()
        return {r[0] for r in rows}

    def close(self) -> None:
        self.conn.close()
