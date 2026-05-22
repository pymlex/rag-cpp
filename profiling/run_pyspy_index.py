import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiling.pyspy_common import (
    ROOT,
    apply_profile_corpus_override,
    load_profile_env,
    pyspy_record_prefix,
    results_subdir,
    runner_env,
)


def main() -> None:
    load_profile_env()
    apply_profile_corpus_override()
    out_dir = results_subdir()
    cmd = pyspy_record_prefix(out_dir / "index_sync.svg")
    cmd.extend(["--", sys.executable, str(ROOT / "profiling" / "profile_index_sync.py")])
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=runner_env())


if __name__ == "__main__":
    main()
