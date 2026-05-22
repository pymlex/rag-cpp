import json
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import TEXT_EXTENSIONS
from python.corpus_paths import get_corpus_root
from python.file_scanner import iter_text_files
from python.llm_client import ZvenoClient


QA_SYSTEM = (
    "Ты составляешь набор вопросов и ответов для оценки RAG по корпусу документов. "
    "Используй только факты из переданного контекста. "
    "Верни строго JSON-массив из 20 объектов с полями: "
    "question (строка), answer (строка), source_file (строка, относительный путь), "
    "must_contain (массив из 2-4 коротких фраз из answer). "
    "В source_file только прямые слэши, без обратных слэшей. Пример: docs/note.md. "
    "Вопросы на русском, разные аспекты корпуса. Без markdown-обёртки, только JSON."
)


def collect_context(root: Path, max_chars: int = 28000) -> str:
    blocks = []
    total = 0
    for path in iter_text_files(root, TEXT_EXTENSIONS):
        if "rag_index" in path.parts:
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        text = path.read_text(encoding="utf-8", errors="ignore")
        piece = f"### {rel}\n{text[:2500]}\n"
        if total + len(piece) > max_chars:
            break
        blocks.append(piece)
        total += len(piece)
    return "\n".join(blocks)


def repair_json_payload(payload: str) -> str:
    return re.sub(
        r'(?<!\\)\\(?![\\"/bfnrtu])',
        "/",
        payload,
    )


def parse_json_array(raw: str) -> list[dict]:
    match = re.search(r"\[[\s\S]*\]", raw)
    payload = match.group(0) if match else raw
    payload = repair_json_payload(payload)
    data = json.loads(payload)
    out = []
    for item in data[:20]:
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
    context = collect_context(root)
    client = ZvenoClient()
    raw = client.chat(
        messages=[
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user", "content": context},
        ],
        max_tokens=4000,
        temperature=0.2,
    )
    raw_path = ROOT / "benchmarks" / "last_qa_raw.txt"
    raw_path.write_text(raw, encoding="utf-8")
    dataset = parse_json_array(raw)
    out_path = ROOT / "benchmarks" / "generated_qa.json"
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive = ROOT / "benchmarks" / "datasets" / f"qa_{stamp}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"saved {len(dataset)} items to {out_path}")


if __name__ == "__main__":
    main()
