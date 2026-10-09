"""Docker sandbox: one fresh container per episode, network off, resource limits.

Uses the docker CLI, so it needs no Python client and works with any compatible runtime
(Docker, Podman via `docker` alias). Images are built once per task content hash.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
import uuid
from dataclasses import dataclass

from swarmrl.sandbox.base import ExecResult, InfraError, Sandbox, SandboxFactory

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Limits:
    memory: str = "4g"
    cpus: float = 2.0
    pids: int = 512


def _cli(binary: str, *args: str, timeout: float = 300.0, input: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [binary, *args], capture_output=True, text=True, timeout=timeout, input=input, errors="replace"
        )
    except FileNotFoundError as e:
        raise InfraError(f"{binary} CLI not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise InfraError(f"{binary} {args[0]} timed out after {timeout}s") from e


class DockerSandbox(Sandbox):
    binary = "docker"

    def __init__(self, image: str, workdir: str = "/workspace", limits: Limits = Limits()):
        super().__init__()
        self.workdir = workdir
        self.name = f"swarmrl-{uuid.uuid4().hex[:12]}"
        r = _cli(
            self.binary, "run", "-d", "--rm", "--name", self.name,
            "--network", "none",
            "--memory", limits.memory, "--memory-swap", limits.memory,
            "--cpus", str(limits.cpus), "--pids-limit", str(limits.pids),
            "--security-opt", "no-new-privileges",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-w", workdir, image, "sleep", "infinity",
        )  # fmt: skip
        if r.returncode != 0:
            self._closed = True
            raise InfraError(f"container start failed: {r.stderr.strip()}")
        try:
            self.init_baseline()
        except BaseException:
            self.close()
            raise

    def _run(self, cmd: str, timeout: float, input: str | None = None) -> ExecResult:
        # `timeout` inside the container kills the process tree; the outer timeout is a backstop.
        inner = f"timeout --kill-after=5 {max(int(timeout), 1)} bash -c {shlex.quote(cmd)}"
        args = ["exec", *(["-i"] if input is not None else []), "-w", self.workdir, self.name, "bash", "-c", inner]
        try:
            p = subprocess.run(
                [self.binary, *args],
                capture_output=True,
                text=True,
                timeout=timeout + 30,
                input=input,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", f"command timed out after {timeout}s", timed_out=True)
        if p.returncode in (125, 126, 127) and ("No such container" in p.stderr or "is not running" in p.stderr):
            raise InfraError(p.stderr.strip())
        return ExecResult(p.returncode, p.stdout, p.stderr, timed_out=p.returncode == 124)

    def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        return self._run(cmd, timeout)

    def write_file(self, path: str, content: str) -> None:
        # Content goes over stdin: a single argv entry is capped at 128 KB by the kernel.
        q = shlex.quote(path)
        r = self._run(f'mkdir -p "$(dirname {q})" && cat > {q}', timeout=120, input=content)
        if r.returncode != 0:
            raise OSError(f"write {path} failed: {r.output}")

    def copy_in(self, src: str | os.PathLike[str], dest: str = ".") -> None:
        dest_abs = dest if dest.startswith("/") else os.path.normpath(os.path.join(self.workdir, dest))
        r = _cli(self.binary, "cp", f"{os.fspath(src)}/.", f"{self.name}:{dest_abs}")
        if r.returncode != 0:
            raise InfraError(f"{self.binary} cp failed: {r.stderr.strip()}")

    def _teardown(self) -> None:
        _cli(self.binary, "rm", "-f", self.name, timeout=60)


class DockerSandboxFactory(SandboxFactory):
    def __init__(self, limits: Limits = Limits()):
        self.limits = limits
        self._built: dict[str, str] = {}
        self._lock = threading.Lock()

    def image_for(self, task) -> str:
        if task.docker_image:
            return task.docker_image
        tag = f"swarmrl-task-{task.content_hash()[:16]}"
        with self._lock:  # one build per task even when many episodes start at once
            if tag in self._built.values():
                return tag
            if _cli("docker", "image", "inspect", tag, timeout=60).returncode != 0:
                log.info("building %s for %s", tag, task.task_id)
                r = _cli(
                    "docker", "build", "-q", "-t", tag, os.fspath(task.environment_dir), timeout=task.build_timeout_s
                )
                if r.returncode != 0:
                    raise InfraError(f"image build failed for {task.task_id}: {r.stderr.strip()[-2000:]}")
            self._built[task.task_id] = tag
        return tag

    def create(self, task) -> DockerSandbox:
        return DockerSandbox(self.image_for(task), task.workdir, self.limits)
