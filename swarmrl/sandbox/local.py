"""Local sandbox: a temp-dir copy of the repo and subprocesses.

For development, tests and the oracle-agent check. It gives a fresh repo per episode but NOT
network, memory, CPU or pid isolation: never run model output at scale with it; use
DockerSandbox or KubernetesSandbox.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from swarmrl.sandbox.base import ExecResult, InfraError, Sandbox, SandboxFactory


class LocalSandbox(Sandbox):
    def __init__(self, repo_dir: str | os.PathLike[str]):
        super().__init__()
        src = Path(repo_dir)
        if not src.is_dir():
            self._closed = True
            raise InfraError(f"repo dir not found: {src}")
        self._tmp = tempfile.mkdtemp(prefix="swarmrl-")
        self.workdir = os.path.join(self._tmp, "workspace")
        try:
            shutil.copytree(src, self.workdir, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
            self.init_baseline()
        except BaseException:
            self.close()
            raise

    def _resolve(self, path: str) -> Path:
        root = Path(self.workdir).resolve()
        p = (root / path).resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"path escapes the workspace: {path}")
        return p

    def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": self._tmp,
            "PYTHONDONTWRITEBYTECODE": "1",
            "LANG": "C.UTF-8",
        }
        try:
            p = subprocess.run(
                ["bash", "-c", cmd],
                cwd=self.workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
                start_new_session=True,  # so a timeout kills the whole process group
            )
            return ExecResult(p.returncode, p.stdout, p.stderr)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return ExecResult(124, out, f"command timed out after {timeout}s", timed_out=True)

    def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise FileNotFoundError(path)
        return p.read_text(errors="replace")

    def write_file(self, path: str, content: str) -> None:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def copy_in(self, src: str | os.PathLike[str], dest: str = ".") -> None:
        shutil.copytree(src, self._resolve(dest), dirs_exist_ok=True)

    def _teardown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


class LocalSandboxFactory(SandboxFactory):
    def create(self, task) -> LocalSandbox:
        if task.repo_dir is None:
            raise InfraError(f"task {task.task_id} has no environment/repo for local sandboxes")
        return LocalSandbox(task.repo_dir)
