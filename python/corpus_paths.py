import os
from pathlib import Path


def get_corpus_root() -> Path:
    value = os.environ["RAG_CORPUS_ROOT"]
    return Path(value).resolve()
