import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".cpp", ".hpp", ".h", ".c", ".cc",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".rst", ".tex", ".csv", ".tsv", ".sql", ".sh", ".ps1",
    ".bat", ".xml", ".html", ".css", ".js", ".ts", ".java",
    ".go", ".rs", ".rb", ".php", ".r", ".ipynb",
}

CHUNK_TOKEN_SIZE = 320
CHUNK_TOKEN_OVERLAP = 80
CHUNK_INDEX_FINGERPRINT = f"{CHUNK_TOKEN_SIZE}:{CHUNK_TOKEN_OVERLAP}:heading-v1"
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "mlsa-iai-msu-lab/sci-rus-tiny")
EMBEDDING_DIM = 312
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64
MAX_ELEMENTS = 200000

HYBRID_VECTOR_WEIGHT = 0.65
HYBRID_BM25_WEIGHT = 0.35
RETRIEVAL_TOP_K = 16
FINAL_TOP_K = 8
HYBRID_CANDIDATE_MULT = 4
HYBRID_RRF_K = 60
RERANK_OVERLAP_WEIGHT = 0.2
MAX_CHUNKS_PER_FILE = 2

EMBEDDING_URL = "http://127.0.0.1:8000/"
EMBED_BATCH_SIZE = 8
EMBED_REQUEST_TIMEOUT = 300
ZVENO_BASE_URL = "https://api.zveno.ai/v1"
ZVENO_MODEL = "openai/gpt-oss-120b"
HYDE_MAX_TOKENS = 180
LLM_MAX_OUTPUT_TOKENS = 900
LLM_TEMPERATURE = 0.2

INDEX_SUBDIR = "rag_index"
HNSW_FILE = "hnsw.bin"
META_DB = "chunks.sqlite"
BM25_FILE = "bm25_corpus.json"
