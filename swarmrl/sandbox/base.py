"""Sandbox interface: one fresh, isolated repo per episode."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Protocol

GIT_EXCLUDES = "__pycache__/\n*.pyc\n.pytest_cache/\n.swarmrl/\n"
BASELINE_TAG = "swarmrl-baseline"


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
    """The sandbox itself failed (container died, image missing). Never scored against the model."""


class Sandbox(Protocol):
    workdir: str

    def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult: ...

    def read_file(self, path: str) -> str: ...

    def write_file(self, path: str, content: str) -> None: ...

    def close(self) -> None: ...


class SandboxFactory(Protocol):
    def create(self, task: "object") -> Sandbox: ...


def init_baseline(sb: Sandbox) -> None:
    """Commit the starting repo state so the final diff is exactly the agent's change."""
    script = (
        "git init -q 2>/dev/null; "
        f"mkdir -p .git/info && printf {shlex.quote(GIT_EXCLUDES)} >> .git/info/exclude; "
        "git -c user.email=swarm@local -c user.name=swarm add -A && "
        "git -c user.email=swarm@local -c user.name=swarm commit -q --allow-empty -m baseline && "
        f"git tag -f {BASELINE_TAG} >/dev/null"
    )
    r = sb.exec(script, timeout=120)
    if r.returncode != 0:
        raise InfraError(f"baseline commit failed: {r.output}")


def extract_diff(sb: Sandbox) -> str:
    """The agent's full change against the baseline, including new files."""
    r = sb.exec(f"git add -A && git diff --cached --binary {BASELINE_TAG}", timeout=120)
    if r.returncode != 0:
        raise InfraError(f"diff extraction failed: {r.output}")
    return r.stdout


def apply_diff(sb: Sandbox, diff: str) -> ExecResult:
    if not diff.strip():
        return ExecResult(0, "", "")
    sb.write_file(".swarmrl/agent.patch", diff)
    return sb.exec("git apply --whitespace=nowarn .swarmrl/agent.patch", timeout=120)
