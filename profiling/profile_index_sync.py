import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from python.corpus_paths import get_corpus_root
from python.index_manager import IndexManager


def main() -> None:
    load_dotenv(ROOT / ".env")
    manager = IndexManager(get_corpus_root())
    manager.sync()
    manager.close()


if __name__ == "__main__":
    main()
