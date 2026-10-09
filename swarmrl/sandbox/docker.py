"""Docker sandbox: one fresh container per episode, network off, resource limits.

Talks to the docker CLI so it needs no Python client. The Kubernetes pool follows the
same interface (exec / read_file / write_file / copy_in / close).
"""

from __future__ import annotations

import base64
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass

from swarmrl.sandbox.base import ExecResult, InfraError, init_baseline


@dataclass(frozen=True)
class Limits:
    memory: str = "4g"
    cpus: float = 2.0
    pids: int = 512


def _docker(*args: str, timeout: float = 300.0, input: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout, input=input, errors="replace"
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise InfraError(f"docker {args[0]} failed: {e}") from e


class DockerSandbox:
    def __init__(self, image: str, workdir: str = "/workspace", limits: Limits = Limits()):
        self.workdir = workdir
        self.name = f"swarmrl-{uuid.uuid4().hex[:12]}"
        r = _docker(
            "run", "-d", "--rm", "--name", self.name,
            "--network", "none",
            "--memory", limits.memory, "--cpus", str(limits.cpus), "--pids-limit", str(limits.pids),
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-w", workdir, image, "sleep", "infinity",
        )  # fmt: skip
        if r.returncode != 0:
            raise InfraError(f"container start failed: {r.stderr.strip()}")
        init_baseline(self)

    def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        # `timeout` inside the container kills the process tree; the outer timeout is a backstop.
        inner = f"timeout --kill-after=5 {int(timeout)} bash -c {shlex.quote(cmd)}"
        try:
            p = subprocess.run(
                ["docker", "exec", "-w", self.workdir, self.name, "bash", "-c", inner],
                capture_output=True, text=True, timeout=timeout + 30, errors="replace",
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", f"command timed out after {timeout}s", timed_out=True)
        if p.returncode in (125, 126) and "No such container" in p.stderr:
            raise InfraError(p.stderr.strip())
        timed_out = p.returncode == 124
        return ExecResult(p.returncode, p.stdout, p.stderr, timed_out=timed_out)

    def read_file(self, path: str) -> str:
        r = self.exec(f"cat -- {shlex.quote(path)}")
        if r.returncode != 0:
            raise FileNotFoundError(path)
        return r.stdout

    def write_file(self, path: str, content: str) -> None:
        b64 = base64.b64encode(content.encode()).decode()
        q = shlex.quote(path)
        r = self.exec(f'mkdir -p "$(dirname {q})" && echo {b64} | base64 -d > {q}')
        if r.returncode != 0:
            raise OSError(f"write failed: {r.output}")

    def copy_in(self, src: str | os.PathLike[str], dest: str = ".") -> None:
        dest_abs = dest if dest.startswith("/") else os.path.join(self.workdir, dest)
        r = _docker("cp", f"{os.fspath(src)}/.", f"{self.name}:{dest_abs}")
        if r.returncode != 0:
            raise InfraError(f"docker cp failed: {r.stderr.strip()}")

    def close(self) -> None:
        _docker("rm", "-f", self.name, timeout=60)


class DockerSandboxFactory:
    def __init__(self, limits: Limits = Limits()):
        self.limits = limits
        self._built: dict[str, str] = {}

    def image_for(self, task) -> str:
        if task.docker_image:
            return task.docker_image
        if task.task_id in self._built:
            return self._built[task.task_id]
        tag = f"swarmrl-task-{task.content_hash()[:16]}"
        r = _docker("build", "-q", "-t", tag, os.fspath(task.environment_dir), timeout=task.build_timeout_s)
        if r.returncode != 0:
            raise InfraError(f"image build failed for {task.task_id}: {r.stderr.strip()[-2000:]}")
        self._built[task.task_id] = tag
        return tag

    def create(self, task) -> DockerSandbox:
        return DockerSandbox(self.image_for(task), task.workdir, self.limits)
