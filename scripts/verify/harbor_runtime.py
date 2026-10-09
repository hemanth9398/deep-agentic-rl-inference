"""Verify the Harbor task runtime inside your sandbox pool.

Usage: python scripts/verify/harbor_runtime.py TASK_DIR

Builds the task image, then runs Harbor's own entry points in containers with the network off:
  - tests/test.sh on the buggy repo must write reward 0
  - solution/solve.sh then tests/test.sh must write reward 1
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from swarmrl.sandbox.docker import DockerSandboxFactory
from swarmrl.tasks.harbor import HarborTask


def run(image: str, task: Path, solve: bool) -> str:
    logs = Path(tempfile.mkdtemp())
    script = ("bash /solution/solve.sh && " if solve else "") + "bash /tests/test.sh; cat /logs/verifier/reward.txt"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        f"{task / 'tests'}:/tests:ro",
        "-v",
        f"{task / 'solution'}:/solution:ro",
        "-v",
        f"{logs}:/logs/verifier",
        image,
        "bash",
        "-c",
        script,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return (p.stdout.strip().splitlines() or ["?"])[-1]


def main() -> int:
    task = HarborTask.load(sys.argv[1])
    image = DockerSandboxFactory().image_for(task)
    before, after = run(image, task.task_dir, False), run(image, task.task_dir, True)
    print(f"reward before fix: {before}; after oracle: {after}")
    ok = before == "0" and after == "1"
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
