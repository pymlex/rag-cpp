import sys

from dotenv import load_dotenv

from python.corpus_paths import get_corpus_root
from python.index_manager import IndexManager
from python.rag_pipeline import RagPipeline


def main() -> None:
    load_dotenv()
    if len(sys.argv) < 2:
        print("usage: python main.py <query>")
        sys.exit(1)
    query = sys.argv[1]
    manager = IndexManager(get_corpus_root())
    manager.sync()
    pipeline = RagPipeline(manager)
    hyde_text, answer = pipeline.answer(query, use_hyde=True)
    print(hyde_text)
    print(answer)
    manager.close()


if __name__ == "__main__":
    main()
