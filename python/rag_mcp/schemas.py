from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    chunk_id: str
    label: int
    file_path: str
    chunk_index: int
    text: str
    content_hash: str
    created_at: datetime
    modified_at: datetime


class FileMeta(BaseModel):
    file_path: str
    content_hash: str
    created_at: datetime
    modified_at: datetime
    chunk_ids: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    chunk_id: str
    file_path: str
    text: str
    vector_score: float | None = None
    bm25_score: float | None = None
    fused_score: float
    modified_at: datetime | None = None


class RagAnswer(BaseModel):
    answer: str
    sources: list[SearchResult]
    found: bool


class BenchmarkCase(BaseModel):
    query: str
    expected_chunk_ids: list[str]
    expected_keywords: list[str] = Field(default_factory=list)
    domain: str = "knowledge"


class BenchmarkRunResult(BaseModel):
    case_id: str
    recall_at_k: float
    mrr: float
    keyword_hit: float
    answer_found: bool
    latency_ms: float
    extra: dict[str, Any] = Field(default_factory=dict)
