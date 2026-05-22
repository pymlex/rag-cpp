import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rag_mcp.config import CONFIG, ROOT
from rag_mcp.pipeline import RagPipeline
from rag_mcp.schemas import BenchmarkCase, BenchmarkRunResult


DATASET_PATH = ROOT / "benchmarks" / "data" / "tau2_knowledge_ru.json"
RESULTS_DIR = ROOT / "benchmarks" / "results"


def load_cases() -> list[BenchmarkCase]:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return [BenchmarkCase.model_validate(item) for item in payload]


def recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    top = set(retrieved[:k])
    if len(expected) == 0:
        return 0.0
    hit = len(top.intersection(expected))
    return hit / len(expected)


def mrr(expected: list[str], retrieved: list[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def keyword_hit(case: BenchmarkCase, answer: str) -> float:
    if len(case.expected_keywords) == 0:
        return 1.0
    lower = answer.lower()
    hits = sum(1 for kw in case.expected_keywords if kw.lower() in lower)
    return hits / len(case.expected_keywords)


def run_benchmark(corpus: Path) -> dict:
    pipeline = RagPipeline(corpus)
    pipeline.sync_index()
    cases = load_cases()
    rows: list[BenchmarkRunResult] = []
    for idx, case in enumerate(cases):
        t0 = time.perf_counter()
        hits = pipeline.searcher.search(case.query)
        retrieved = [hit.chunk_id for hit in hits]
        result = pipeline.query(case.query, use_hyde=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        row = BenchmarkRunResult(
            case_id=f"case_{idx:03d}",
            recall_at_k=recall_at_k(case.expected_chunk_ids, retrieved, CONFIG["fusion_top_k"]),
            mrr=mrr(case.expected_chunk_ids, retrieved),
            keyword_hit=keyword_hit(case, result.answer),
            answer_found=result.found,
            latency_ms=latency_ms,
            extra={"domain": case.domain},
        )
        rows.append(row)
    return {"rows": rows, "cases": len(cases)}


def save_outputs(report: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = report["rows"]
    payload = [row.model_dump(mode="json") for row in rows]
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    recall = [row.recall_at_k for row in rows]
    mrr_vals = [row.mrr for row in rows]
    latency = [row.latency_ms for row in rows]
    summary = {
        "cases": report["cases"],
        "mean_recall_at_k": float(np.mean(recall)),
        "mean_mrr": float(np.mean(mrr_vals)),
        "mean_keyword_hit": float(np.mean([row.keyword_hit for row in rows])),
        "answer_found_rate": float(np.mean([1.0 if row.answer_found else 0.0 for row in rows])),
        "mean_latency_ms": float(np.mean(latency)),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(recall, bins=10)
    axes[0].set_title("Recall@k")
    axes[1].hist(mrr_vals, bins=10)
    axes[1].set_title("MRR")
    axes[2].hist(latency, bins=10)
    axes[2].set_title("Latency ms")
    fig.tight_layout()
    fig.savefig(out_dir / "tau2_rag_metrics.png", dpi=160)
    plt.close(fig)
    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    report = run_benchmark(Path(args.corpus))
    out = save_outputs(report)
    print(out)
