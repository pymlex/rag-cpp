import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def main() -> None:
    out_dir = ROOT / "benchmarks" / "results" / f"tau2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = os.getenv("ZVENO_MODEL", "openai/gpt-oss-120b")
    cmd = [
        "tau2",
        "run",
        "--domain",
        "banking_knowledge",
        "--retrieval-config",
        "bm25",
        "--agent-llm",
        model,
        "--user-llm",
        model,
        "--num-trials",
        "1",
        "--num-tasks",
        "5",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    (out_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    summary = {
        "returncode": proc.returncode,
        "command": cmd,
        "model": model,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
