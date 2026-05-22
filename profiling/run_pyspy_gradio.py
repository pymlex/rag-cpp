import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiling.pyspy_common import (
    ROOT,
    load_profile_env,
    pyspy_record_prefix,
    results_subdir,
    runner_env,
)


def main() -> None:
    load_profile_env()
    out_dir = results_subdir()
    duration = os.getenv("PYSPY_DURATION", "30")
    cmd = pyspy_record_prefix(out_dir / "gradio.svg", duration=duration)
    cmd.extend(["--", sys.executable, str(ROOT / "gradio_app.py")])
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=runner_env())


if __name__ == "__main__":
    main()
