import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.generate_qa_dataset import (
    OUT_PATH,
    flush_dataset,
    load_dataset,
    merge_batch,
    parse_json_array,
)
from config.settings import BENCHMARK_QA_BATCH_SIZE


def main() -> None:
    raw_dir = ROOT / "benchmarks" / "datasets" / "raw_batches"
    parsed_dir = ROOT / "benchmarks" / "datasets" / "parsed_batches"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    dataset, seen = load_dataset()
    paths = sorted(raw_dir.glob("batch_*.txt"))
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        batch = parse_json_array(raw, BENCHMARK_QA_BATCH_SIZE)
        parsed_path = parsed_dir / f"{path.stem}.json"
        parsed_path.write_text(
            json.dumps(batch, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        added = merge_batch(dataset, seen, batch)
        flush_dataset(dataset)
        print(f"{path.stem}: parsed {len(batch)} new {added} total {len(dataset)}", flush=True)
    flush_dataset(dataset)
    print(f"done {len(dataset)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
