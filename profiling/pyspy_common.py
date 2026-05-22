import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def load_profile_env() -> None:
    load_dotenv(ROOT / ".env")


def apply_profile_corpus_override() -> None:
    override = os.environ.get("RAG_PROFILE_CORPUS")
    if override:
        os.environ["RAG_CORPUS_ROOT"] = override


def results_subdir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out = ROOT / "profiling" / "results" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def pyspy_executable() -> str:
    if sys.platform == "win32":
        candidate = ROOT / ".venv" / "Scripts" / "py-spy.exe"
        if candidate.is_file():
            return str(candidate)
    return "py-spy"


def pyspy_record_prefix(out_file: Path, duration: str | None = None) -> list[str]:
    cmd = [
        pyspy_executable(),
        "record",
        "-o",
        str(out_file),
        "--format",
        "speedscope",
    ]
    if sys.platform == "win32":
        cmd.append("--subprocesses")
    if duration is not None:
        cmd.extend(["-d", duration])
    return cmd


def runner_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return env
