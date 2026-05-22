import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out_dir = ROOT / "profiling" / "results" / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = os.getenv("PYSPY_DURATION", "30")
    cmd = [
        "py-spy",
        "record",
        "-o",
        str(out_dir / "gradio.svg"),
        "--format",
        "speedscope",
        "-d",
        duration,
        "--",
        sys.executable,
        str(ROOT / "gradio_app.py"),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
