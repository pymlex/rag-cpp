from pathlib import Path

from rag_mcp.config import CONFIG
from rag_mcp.manifest import file_hash


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in CONFIG["text_extensions"]


def scan_corpus(root: Path) -> dict[str, Path]:
    files = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not is_text_candidate(path):
            continue
        rel = str(path.resolve().relative_to(root.resolve()))
        files[rel] = path
    return files


def build_file_meta(path: Path, rel: str, digest: str) -> dict:
    stat = path.stat()
    return {
        "file_path": rel,
        "content_hash": digest,
        "created_at": stat.st_ctime,
        "modified_at": stat.st_mtime,
    }


def current_file_state(root: Path) -> dict[str, dict]:
    scanned = scan_corpus(root)
    state = {}
    for rel, path in scanned.items():
        state[rel] = build_file_meta(path, rel, file_hash(path))
    return state
