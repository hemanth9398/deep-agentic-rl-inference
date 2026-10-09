"""Sandbox base classes: one fresh, isolated repo per episode.

Backends (LocalSandbox, DockerSandbox, KubernetesSandbox) implement four primitives:
`exec`, `write_file`, `copy_in` and `_teardown`. Everything else (baseline commit,
diff extraction, patch application, file reads, context management) is shared here.
"""

from __future__ import annotations

import abc
import base64
import logging
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

    from swarmrl.tasks.harbor import HarborTask

log = logging.getLogger(__name__)

GIT_EXCLUDES = "__pycache__/\n*.pyc\n.pytest_cache/\n.swarmrl/\n"
BASELINE_TAG = "swarmrl-baseline"
_GIT = "git -c user.email=swarm@local -c user.name=swarm"


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def output(self) -> str:
        if self.stderr:
            return f"{self.stdout}\n{self.stderr}" if self.stdout else self.stderr
        return self.stdout


class InfraError(RuntimeError):
    """The sandbox or a service failed (container died, image missing). Never scored against the model."""


class Sandbox(abc.ABC):
    """A live, isolated working copy of a task repository."""

    workdir: str

    def __init__(self) -> None:
        self._closed = False

    # --- backend primitives -------------------------------------------------------

    @abc.abstractmethod
    def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        """Run a bash command in the workdir. Must enforce `timeout` and never raise for a non-zero exit."""

    @abc.abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Write a text file (relative to the workdir), creating parent directories."""

    @abc.abstractmethod
    def copy_in(self, src: str | PathLike[str], dest: str = ".") -> None:
        """Overlay a host directory onto `dest` inside the sandbox (hidden tests are copied in this way)."""

    @abc.abstractmethod
    def _teardown(self) -> None:
        """Release backend resources. Called once by `close`."""

    # --- shared behaviour -------------------------------------------------------------

    def read_file(self, path: str) -> str:
        r = self.exec(f"base64 -- {shlex.quote(path)}", timeout=60)
        if r.returncode != 0:
            raise FileNotFoundError(path)
        return base64.b64decode(r.stdout).decode(errors="replace")

    def exists(self, path: str) -> bool:
        return self.exec(f"test -e {shlex.quote(path)}", timeout=30).returncode == 0

    def init_baseline(self) -> None:
        """Commit the starting repo state so the final diff is exactly the agent's change."""
        script = (
            "git init -q 2>/dev/null; "
            f"mkdir -p .git/info && printf {shlex.quote(GIT_EXCLUDES)} >> .git/info/exclude; "
            f"{_GIT} add -A && {_GIT} commit -q --allow-empty -m baseline && "
            f"git tag -f {BASELINE_TAG} >/dev/null"
        )
        r = self.exec(script, timeout=120)
        if r.returncode != 0:
            raise InfraError(f"baseline commit failed: {r.output}")

    def diff(self) -> str:
        """The agent's full change against the baseline, including new and binary files."""
        r = self.exec(f"git add -A && git diff --cached --binary {BASELINE_TAG}", timeout=120)
        if r.returncode != 0:
            raise InfraError(f"diff extraction failed: {r.output}")
        return r.stdout

    def apply_patch(self, patch: str) -> ExecResult:
        if not patch.strip():
            return ExecResult(0, "", "")
        self.write_file(".swarmrl/agent.patch", patch)
        return self.exec("git apply --whitespace=nowarn .swarmrl/agent.patch", timeout=120)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._teardown()
        except Exception:
            log.exception("sandbox teardown failed")

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # last-resort leak guard; explicit close() is the contract
        if not getattr(self, "_closed", True):
            log.warning("%s garbage-collected without close(); tearing down", type(self).__name__)
            self.close()


class SandboxFactory(abc.ABC):
    """Creates a fresh sandbox for a task: one per episode, and a separate clean one per grading."""

    @abc.abstractmethod
    def create(self, task: HarborTask) -> Sandbox:
        """Start a sandbox with the task's repo at the buggy commit and a baseline commit taken."""
