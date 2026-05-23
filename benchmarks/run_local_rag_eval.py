import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.eval_metrics import (
    answer_grounded,
    answer_hit_loose,
    answer_hit_strict,
    retrieval_hit,
)
from benchmarks.plot_eval import plot_aggregate, plot_outcomes
from config.settings import RETRIEVAL_TOP_K
from python.corpus_paths import get_corpus_root
from python.embeddings_client import EmbeddingsClient
from python.hybrid_search import HybridSearcher
from python.hyde import expand_query
from python.index_manager import IndexManager
from python.llm_client import ZvenoClient
from python.rag_pipeline import RagPipeline


load_dotenv(ROOT / ".env")


def main() -> None:
    dataset_path = ROOT / "benchmarks" / "generated_qa.json"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing {dataset_path}. Run: python benchmarks/generate_qa_dataset.py"
        )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    out_dir = ROOT / "benchmarks" / "results" / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
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
    answer_loose_scores = []
    answer_strict_scores = []
    answer_grounded_scores = []
    combined_scores = []
    rows = []

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        print(f"eval {index}/{len(dataset)}", flush=True)
        hyde_text = expand_query(llm, question)
        vector = embedder.embed([hyde_text])[0]
        chunks = searcher.search(vector, hyde_text, top_k=RETRIEVAL_TOP_K)
        hit_files = [c.file_path for c in chunks]
        retr = retrieval_hit(hit_files, item["source_file"])
        _, answer = pipeline.answer(question, use_hyde=True)
        ans_loose = answer_hit_loose(answer, item["must_contain"])
        ans_strict = answer_hit_strict(answer, item["must_contain"])
        ans_grounded = answer_grounded(answer)
        retrieval_scores.append(1.0 if retr else 0.0)
        answer_loose_scores.append(1.0 if ans_loose else 0.0)
        answer_strict_scores.append(1.0 if ans_strict else 0.0)
        answer_grounded_scores.append(1.0 if ans_grounded else 0.0)
        combined_scores.append(1.0 if retr and ans_loose else 0.0)
        rows.append(
            {
                "question": question,
                "source_file": item["source_file"],
                "retrieval_pass": retr,
                "answer_pass": ans_loose,
                "answer_strict_pass": ans_strict,
                "answer_grounded": ans_grounded,
                "retrieved_files": hit_files[:6],
                "answer_head": answer[:350],
            }
        )

    manager.close()
    metrics = {
        "n": len(dataset),
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "retrieval_pass_rate": float(np.mean(retrieval_scores)),
        "answer_pass_rate": float(np.mean(answer_loose_scores)),
        "answer_strict_pass_rate": float(np.mean(answer_strict_scores)),
        "answer_grounded_rate": float(np.mean(answer_grounded_scores)),
        "combined_pass_rate": float(np.mean(combined_scores)),
        "rows": rows,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_aggregate(metrics, out_dir / "aggregate.png")
    plot_outcomes(metrics, out_dir / "outcomes.png")

    report_dir = ROOT / "benchmarks" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "aggregate.png", "outcomes.png"):
        target = report_dir / name
        target.write_bytes((out_dir / name).read_bytes())
    (report_dir / "eval_run.json").write_text(
        json.dumps(
            {
                "run_id": out_dir.name,
                "cases": metrics["n"],
                "retrieval_top_k": RETRIEVAL_TOP_K,
                "retrieval_pass_rate": metrics["retrieval_pass_rate"],
                "answer_pass_rate": metrics["answer_pass_rate"],
                "answer_strict_pass_rate": metrics["answer_strict_pass_rate"],
                "answer_grounded_rate": metrics["answer_grounded_rate"],
                "combined_pass_rate": metrics["combined_pass_rate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metrics: {out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
