"""Harbor task directories.

Layout (Harbor schema 1.4, plus the swarm conventions under [metadata]):

    task.toml                 [metadata] split, source, repo, pytest_targets; [environment]; [verifier]
    instruction.md            issue text: symptoms, not the fix          (seen by the agent)
    environment/Dockerfile    repo at the buggy commit, deps, pytest     (the agent works inside it)
    environment/repo/         the same repo as a directory, for LocalSandbox
    tests/test.sh             Harbor verifier entry: overlays tests, runs pytest, writes reward.txt
    tests/overlay/            hidden FAIL_TO_PASS / PASS_TO_PASS tests  (copied in only at grading)
    solution/solve.patch      oracle fix                                 (validation only)
    solution/solve.sh         Harbor oracle entry: git apply /solution/solve.patch

The same directory runs unchanged in SkyRL-Agent, Verifiers, NeMo Gym and Harbor itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPLITS = ("train", "dev", "heldout", "external")

TEST_SH = """#!/bin/bash
# Harbor verifier entry: overlay hidden tests, run pytest, write a 0/1 reward.
set -u
cd "${WORKDIR:-/workspace}"
cp -r /tests/overlay/. .
mkdir -p /logs/verifier
if python -m pytest -q -p no:cacheprovider {targets}; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

SOLVE_SH = """#!/bin/bash
set -e
cd "${WORKDIR:-/workspace}"
git apply /solution/solve.patch
"""


@dataclass
class HarborTask:
    task_dir: Path
    task_id: str
    instruction: str
    split: str
    source: str
    repo: str
    pytest_targets: list[str]
    workdir: str = "/workspace"
    docker_image: str | None = None
    build_timeout_s: float = 600.0
    verifier_timeout_s: float = 600.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def environment_dir(self) -> Path:
        return self.task_dir / "environment"

    @property
    def repo_dir(self) -> Path | None:
        p = self.environment_dir / "repo"
        return p if p.is_dir() else None

    @property
    def test_overlay_dir(self) -> Path:
        return self.task_dir / "tests" / "overlay"

    @property
    def oracle_patch(self) -> str:
        p = self.task_dir / "solution" / "solve.patch"
        return p.read_text() if p.exists() else ""

    @property
    def min_turns_to_solve(self) -> int | None:
        v = self.metadata.get("min_turns_to_solve")
        return int(v) if v is not None else None

    def content_hash(self) -> str:
        """Hash of every file in the task; validation results are cached under it."""
        h = hashlib.sha256()
        for p in sorted(self.task_dir.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".validation.json":
                h.update(str(p.relative_to(self.task_dir)).encode())
                h.update(b"\0")
                h.update(p.read_bytes())
        return h.hexdigest()

    @classmethod
    def load(cls, task_dir: str | os.PathLike[str]) -> "HarborTask":
        d = Path(task_dir).resolve()
        cfg = tomllib.loads((d / "task.toml").read_text())
        meta = dict(cfg.get("metadata", {}))
        env = cfg.get("environment", {})
        ver = cfg.get("verifier", {})
        split = meta.get("split", "train")
        if split not in SPLITS:
            raise ValueError(f"{d.name}: unknown split {split!r}")
        return cls(
            task_dir=d,
            task_id=meta.get("task_id", d.name),
            instruction=(d / "instruction.md").read_text(),
            split=split,
            source=meta.get("source", "unknown"),
            repo=meta.get("repo", d.name),
            pytest_targets=list(meta.get("pytest_targets", [])),
            workdir=env.get("workdir", "/workspace"),
            docker_image=env.get("docker_image"),
            build_timeout_s=float(env.get("build_timeout_sec", 600.0)),
            verifier_timeout_s=float(ver.get("timeout_sec", 600.0)),
            metadata=meta,
        )


def load_tasks(root: str | os.PathLike[str], split: str | None = None) -> list[HarborTask]:
    tasks = [HarborTask.load(p.parent) for p in sorted(Path(root).rglob("task.toml"))]
    return [t for t in tasks if split is None or t.split == split]


def _toml_value(v: Any) -> str:
    return json.dumps(v)  # JSON strings, numbers, bools and lists of them are valid TOML


def write_task(
    task_dir: str | os.PathLike[str],
    *,
    instruction: str,
    repo_src: str | os.PathLike[str],
    test_overlay_src: str | os.PathLike[str],
    oracle_patch: str,
    pytest_targets: list[str],
    split: str,
    source: str,
    repo: str,
    extra_metadata: dict[str, Any] | None = None,
    base_image: str = "python:3.12-slim",
) -> Path:
    """Write a complete Harbor task directory (used by the mutant generator and adapters)."""
    import shutil

    d = Path(task_dir)
    (d / "environment").mkdir(parents=True, exist_ok=True)
    (d / "tests").mkdir(exist_ok=True)
    (d / "solution").mkdir(exist_ok=True)
    meta = {"split": split, "source": source, "repo": repo, "pytest_targets": pytest_targets, **(extra_metadata or {})}
    lines = ['schema_version = "1.4"', "", "[metadata]"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in meta.items()]
    lines += [
        "",
        "[environment]",
        'workdir = "/workspace"',
        "allow_internet = false",
        "cpus = 2",
        "memory_mb = 4096",
        "",
        "[verifier]",
        "timeout_sec = 600.0",
        "",
        "[agent]",
        "timeout_sec = 1800.0",
        "",
    ]
    (d / "task.toml").write_text("\n".join(lines))
    (d / "instruction.md").write_text(instruction)
    shutil.copytree(repo_src, d / "environment" / "repo", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))  # fmt: skip
    (d / "environment" / "Dockerfile").write_text(
        f"FROM {base_image}\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*\n"
        "RUN pip install --no-cache-dir pytest\n"
        "COPY repo/ /workspace/\nWORKDIR /workspace\n"
        "RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi\n"
    )
    shutil.copytree(test_overlay_src, d / "tests" / "overlay", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))  # fmt: skip
    (d / "tests" / "test.sh").write_text(TEST_SH.replace("{targets}", " ".join(pytest_targets)))
    (d / "solution" / "solve.patch").write_text(oracle_patch)
    (d / "solution" / "solve.sh").write_text(SOLVE_SH)
    return d
