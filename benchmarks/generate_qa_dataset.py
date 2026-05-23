import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import BENCHMARK_QA_BATCH_SIZE, BENCHMARK_QA_COUNT, TEXT_EXTENSIONS
from python.corpus_paths import get_corpus_root
from python.file_scanner import iter_text_files
from python.llm_client import ZvenoClient


QA_SYSTEM_TEMPLATE = (
    "Ты составляешь набор вопросов и ответов для оценки RAG по корпусу документов. "
    "Используй только факты из переданного контекста. "
    "Верни строго JSON-массив из {count} объектов с полями: "
    "question (строка), answer (строка), source_file (строка, относительный путь), "
    "must_contain (массив из 2-4 коротких фраз из answer). "
    "В source_file только прямые слэши, без обратных слэшей. Пример: docs/note.md. "
    "Вопросы на русском, без повторов, разные файлы и темы. "
    "Без markdown-обёртки, только JSON."
)

CONTEXT_CHAR_LIMIT = 32_000
FILE_SNIPPET_CHARS = 2800


def collect_file_blocks(root: Path) -> list[tuple[str, str]]:
    blocks = []
    for path in iter_text_files(root, TEXT_EXTENSIONS):
        if "rag_index" in path.parts:
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        text = path.read_text(encoding="utf-8", errors="ignore")
        blocks.append((rel, text[:FILE_SNIPPET_CHARS]))
    return blocks


def build_context(blocks: list[tuple[str, str]], start_index: int) -> str:
    pieces = []
    total = 0
    n = len(blocks)
    for offset in range(n):
        rel, text = blocks[(start_index + offset) % n]
        piece = f"### {rel}\n{text}\n"
        if total + len(piece) > CONTEXT_CHAR_LIMIT:
            break
        pieces.append(piece)
        total += len(piece)
    return "\n".join(pieces)


def repair_json_payload(payload: str) -> str:
    return re.sub(
        r'(?<!\\)\\(?![\\"/bfnrtu])',
        "/",
        payload,
    )


def parse_json_array(raw: str, limit: int) -> list[dict]:
    match = re.search(r"\[[\s\S]*\]", raw)
    payload = match.group(0) if match else raw
    payload = repair_json_payload(payload)
    data = json.loads(payload)
    out = []
    for item in data[:limit]:
        source = str(item["source_file"]).replace("\\", "/")
        out.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "source_file": source,
                "must_contain": item["must_contain"],
            }
        )
    return out


def main() -> None:
    load_dotenv(ROOT / ".env")
    root = get_corpus_root()
    blocks = collect_file_blocks(root)
    if not blocks:
        raise FileNotFoundError(f"No text files under {root}")
    client = ZvenoClient()
    dataset: list[dict] = []
    seen_questions: set[str] = set()
    batch_index = 0
    raw_dir = ROOT / "benchmarks" / "datasets" / "raw_batches"
    raw_dir.mkdir(parents=True, exist_ok=True)

    while len(dataset) < BENCHMARK_QA_COUNT:
        need = min(BENCHMARK_QA_BATCH_SIZE, BENCHMARK_QA_COUNT - len(dataset))
        context = build_context(blocks, batch_index * 3)
        system = QA_SYSTEM_TEMPLATE.format(count=need)
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
            max_tokens=12_000,
            temperature=0.2,
        )
        batch_path = raw_dir / f"batch_{batch_index:03d}.txt"
        batch_path.write_text(raw, encoding="utf-8")
        batch = parse_json_array(raw, need)
        for item in batch:
            key = item["question"].strip().lower()
            if key in seen_questions:
                continue
            seen_questions.add(key)
            dataset.append(item)
            if len(dataset) >= BENCHMARK_QA_COUNT:
                break
        print(f"batch {batch_index}: +{len(batch)} total {len(dataset)}", flush=True)
        batch_index += 1
        if batch_index > 30:
            break

    dataset = dataset[:BENCHMARK_QA_COUNT]
    out_path = ROOT / "benchmarks" / "generated_qa.json"
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    archive = ROOT / "benchmarks" / "datasets" / f"qa_{stamp}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"saved {len(dataset)} items to {out_path}")


if __name__ == "__main__":
    main()
