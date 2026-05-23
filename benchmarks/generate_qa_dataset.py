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
OUT_PATH = ROOT / "benchmarks" / "generated_qa.json"


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


def salvage_array_items(raw: str) -> list[dict]:
    payload = repair_json_payload(raw)
    start = payload.find("[")
    if start < 0:
        return []
    s = payload[start:]
    items: list[dict] = []
    i = 1
    n = len(s)
    while i < n:
        while i < n and s[i] in " \t\n\r,":
            i += 1
        if i >= n or s[i] == "]":
            break
        if s[i] != "{":
            break
        depth = 0
        in_str = False
        esc = False
        j = i
        closed = False
        while j < n:
            c = s[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[i : j + 1]
                    items.append(json.loads(chunk))
                    i = j + 1
                    closed = True
                    break
            j += 1
        if not closed:
            break
    return items


def parse_json_array(raw: str, limit: int) -> list[dict]:
    items = salvage_array_items(raw)
    if not items:
        match = re.search(r"\[[\s\S]*\]", raw)
        payload = match.group(0) if match else raw
        payload = repair_json_payload(payload)
        items = json.loads(payload)
    out = []
    for item in items[:limit]:
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


def load_dataset() -> tuple[list[dict], set[str]]:
    if not OUT_PATH.exists():
        return [], set()
    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    seen = {item["question"].strip().lower() for item in data}
    return data, seen


def flush_dataset(dataset: list[dict]) -> None:
    OUT_PATH.write_text(
        json.dumps(dataset[:BENCHMARK_QA_COUNT], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def merge_batch(dataset: list[dict], seen: set[str], batch: list[dict]) -> int:
    added = 0
    for item in batch:
        key = item["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        dataset.append(item)
        added += 1
        if len(dataset) >= BENCHMARK_QA_COUNT:
            break
    return added


def main() -> None:
    load_dotenv(ROOT / ".env")
    root = get_corpus_root()
    blocks = collect_file_blocks(root)
    if not blocks:
        raise FileNotFoundError(f"No text files under {root}")
    client = ZvenoClient()
    dataset, seen_questions = load_dataset()
    batch_index = 0
    raw_dir = ROOT / "benchmarks" / "datasets" / "raw_batches"
    parsed_dir = ROOT / "benchmarks" / "datasets" / "parsed_batches"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    if dataset:
        print(f"resume: {len(dataset)} items in {OUT_PATH}", flush=True)

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
        parsed_path = parsed_dir / f"batch_{batch_index:03d}.json"
        parsed_path.write_text(
            json.dumps(batch, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        added = merge_batch(dataset, seen_questions, batch)
        flush_dataset(dataset)
        print(
            f"batch {batch_index}: parsed {len(batch)} new {added} total {len(dataset)}",
            flush=True,
        )
        if len(batch) == 0:
            print(f"batch {batch_index}: empty parse, raw kept at {batch_path}", flush=True)
        batch_index += 1
        if batch_index > 40:
            break

    flush_dataset(dataset)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    archive = ROOT / "benchmarks" / "datasets" / f"qa_{stamp}.json"
    archive.write_text(OUT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"saved {len(dataset)} items to {OUT_PATH}")


if __name__ == "__main__":
    main()
