"""Validation, before a task enters any split.

1. Run the hidden tests on the buggy repo, then with the oracle applied.
2. Keep the task only if the oracle passes every test and at least one test fails before the fix.
3. Record FAIL_TO_PASS and PASS_TO_PASS, cached by the task's content hash.
4. (swarmrl.tasks.oracle_agent) a scripted oracle agent must resolve it through the real tools.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from swarmrl.grading.pytest_runner import run_pytest
from swarmrl.sandbox.base import SandboxFactory


@dataclass
class ValidationRecord:
    task_id: str
    content_hash: str
    valid: bool
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    reason: str = ""
    oracle_agent_resolved: bool | None = None


class ValidationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, dict] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, content_hash: str) -> ValidationRecord | None:
        d = self._data.get(content_hash)
        return ValidationRecord(**d) if d else None

    def put(self, rec: ValidationRecord) -> None:
        self._data[rec.content_hash] = asdict(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1, sort_keys=True))
        tmp.replace(self.path)


class TaskValidator:
    def __init__(self, factory: SandboxFactory, cache: ValidationCache | None = None):
        self.factory = factory
        self.cache = cache

    def _run(self, task, patch: str | None):
        sb = self.factory.create(task)
        try:
            if patch is not None:
                r = sb.apply_patch(patch)
                if r.returncode != 0:
                    return None, f"oracle patch does not apply: {r.output[-500:]}"
            sb.copy_in(task.test_overlay_dir)
            return run_pytest(sb, task.pytest_targets, timeout=task.verifier_timeout_s), ""
        finally:
            sb.close()

    def validate(self, task) -> ValidationRecord:
        h = task.content_hash()
        if self.cache and (rec := self.cache.get(h)) is not None:
            return rec
        rec = self._validate(task, h)
        if self.cache:
            self.cache.put(rec)
        return rec

    def _validate(self, task, h: str) -> ValidationRecord:
        def bad(reason: str) -> ValidationRecord:
            return ValidationRecord(task.task_id, h, False, reason=reason)

        if not task.oracle_patch.strip():
            return bad("no oracle patch")
        if not task.pytest_targets:
            return bad("no pytest_targets in task.toml")
        before, err = self._run(task, None)
        if before is None:
            return bad(err)
        after, err = self._run(task, task.oracle_patch)
        if after is None:
            return bad(err)
        if after.collection_error or not after.outcomes:
            return bad("tests did not run with the oracle applied")
        if after.failed:
            return bad(f"oracle fails {len(after.failed)} test(s): {sorted(after.failed)[:5]}")
        f2p = sorted(t for t in after.passed if not before.outcomes.get(t, False))
        p2p = sorted(t for t in after.passed if before.outcomes.get(t, False))
        if not f2p:
            return bad("no test fails before the fix")
        return ValidationRecord(task.task_id, h, True, f2p, p2p)
