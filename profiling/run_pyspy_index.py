import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    corpus = os.environ.get("RAG_PROFILE_CORPUS", os.environ["RAG_CORPUS_ROOT"])
    out_dir = ROOT / "profiling" / "results" / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "py-spy",
        "record",
        "-o",
        str(out_dir / "index_sync.svg"),
        "--format",
        "speedscope",
        "--",
        sys.executable,
        "-c",
        (
            f"import os; os.environ['RAG_CORPUS_ROOT']='{corpus}'; "
            f"from python.index_manager import IndexManager; "
            f"from pathlib import Path; "
            f"m=IndexManager(Path(r'{corpus}')); m.sync(); m.close()"
        ),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    main()
