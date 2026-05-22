import hashlib
from datetime import datetime
from pathlib import Path

from config.settings import TEXT_EXTENSIONS


def file_times(path: Path) -> tuple[str, str]:
    stat = path.stat()
    created = datetime.utcfromtimestamp(stat.st_ctime).isoformat()
    modified = datetime.utcfromtimestamp(stat.st_mtime).isoformat()
    return created, modified


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def iter_text_files(root: Path, extensions: set[str]) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in extensions:
            files.append(path)
    return files
