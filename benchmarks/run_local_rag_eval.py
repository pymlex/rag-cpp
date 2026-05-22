import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.corpus_paths import get_corpus_root
from python.embeddings_client import EmbeddingsClient
from python.hybrid_search import HybridSearcher
from python.hyde import expand_query
from python.index_manager import IndexManager
from python.llm_client import ZvenoClient
from python.rag_pipeline import RagPipeline


load_dotenv(ROOT / ".env")


def answer_hit(answer: str, must_contain: list[str]) -> bool:
    low = answer.lower()
    hits = sum(1 for token in must_contain if token.lower() in low)
    return hits >= max(1, len(must_contain) // 2)


def retrieval_hit(hits_files: list[str], source_file: str) -> bool:
    target = source_file.replace("\\", "/").lower()
    for path in hits_files:
        if target in path.replace("\\", "/").lower():
            return True
    return False


def main() -> None:
    dataset_path = ROOT / "benchmarks" / "generated_qa.json"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing {dataset_path}. Run: python benchmarks/generate_qa_dataset.py"
        )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    out_dir = ROOT / "benchmarks" / "results" / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    manager = IndexManager(get_corpus_root())
    manager.sync()
    pipeline = RagPipeline(manager)
    llm = ZvenoClient()
    embedder = EmbeddingsClient()
    searcher = HybridSearcher(
        manager.vector_index,
        manager.bm25,
        manager.meta,
        manager.vector_id_to_chunk,
    )

    retrieval_scores = []
    answer_scores = []
    rows = []

    for item in dataset:
        question = item["question"]
        hyde_text = expand_query(llm, question)
        vector = embedder.embed([hyde_text])[0]
        chunks = searcher.search(vector, hyde_text, top_k=8)
        hit_files = [c.file_path for c in chunks]
        retr = retrieval_hit(hit_files, item["source_file"])
        _, answer = pipeline.answer(question, use_hyde=True)
        ans = answer_hit(answer, item["must_contain"])
        retrieval_scores.append(1.0 if retr else 0.0)
        answer_scores.append(1.0 if ans else 0.0)
        rows.append(
            {
                "question": question,
                "source_file": item["source_file"],
                "retrieval_pass": retr,
                "answer_pass": ans,
                "retrieved_files": hit_files[:5],
                "answer_head": answer[:350],
            }
        )

    manager.close()
    metrics = {
        "n": len(dataset),
        "retrieval_pass_rate": float(np.mean(retrieval_scores)),
        "answer_pass_rate": float(np.mean(answer_scores)),
        "combined_pass_rate": float(np.mean(np.array(retrieval_scores) * np.array(answer_scores))),
        "rows": rows,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    x = np.arange(len(dataset))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, retrieval_scores, width=width, label="retrieval")
    plt.bar(x + width / 2, answer_scores, width=width, label="answer")
    plt.ylim(0, 1.05)
    plt.xlabel("case")
    plt.ylabel("pass")
    plt.legend()
    plt.title("rag_eval")
    plt.tight_layout()
    plt.savefig(out_dir / "pass_rates.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(
        ["retrieval", "answer", "combined"],
        [
            metrics["retrieval_pass_rate"],
            metrics["answer_pass_rate"],
            metrics["combined_pass_rate"],
        ],
    )
    plt.ylim(0, 1.05)
    plt.title("aggregate")
    plt.tight_layout()
    plt.savefig(out_dir / "aggregate.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
