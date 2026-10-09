"""Clean-room grading.

Only the agent's source diff moves into a second, fresh container. Edits to protected
paths are dropped and logged as hack attempts; then the hidden tests are copied in and run.
Nothing the agent wrote can grade itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from swarmrl.grading.protected import filter_diff
from swarmrl.grading.pytest_runner import run_pytest
from swarmrl.sandbox.base import SandboxFactory
from swarmrl.tasks.validate import ValidationRecord


@dataclass
class GradeResult:
    resolved: bool
    f2p_passed: int
    f2p_total: int
    p2p_passed: int
    p2p_total: int
    protected_edits: list[str] = field(default_factory=list)
    apply_failed: bool = False
    failed_tests: list[str] = field(default_factory=list)
    log: str = ""


class CleanRoomGrader:
    def __init__(self, factory: SandboxFactory, protected_patterns: tuple[str, ...]):
        self.factory = factory
        self.protected_patterns = protected_patterns

    def grade(self, task, diff: str, record: ValidationRecord) -> GradeResult:
        kept, dropped = filter_diff(diff, self.protected_patterns)
        sb = self.factory.create(task)
        try:
            applied = sb.apply_patch(kept)
            if applied.returncode != 0:
                return GradeResult(
                    False, 0, len(record.fail_to_pass), 0, len(record.pass_to_pass),
                    dropped, apply_failed=True, log=applied.output[-4000:],
                )  # fmt: skip
            sb.copy_in(task.test_overlay_dir)
            run = run_pytest(sb, task.pytest_targets, timeout=task.verifier_timeout_s)
        finally:
            sb.close()
        f2p_ok = [t for t in record.fail_to_pass if run.outcomes.get(t, False)]
        p2p_ok = [t for t in record.pass_to_pass if run.outcomes.get(t, False)]
        failed = [t for t in record.fail_to_pass + record.pass_to_pass if not run.outcomes.get(t, False)]
        resolved = not failed and len(f2p_ok) == len(record.fail_to_pass) and len(record.fail_to_pass) > 0
        log = run.exec_result.output[-4000:] if run.exec_result else ""
        return GradeResult(
            resolved, len(f2p_ok), len(record.fail_to_pass), len(p2p_ok), len(record.pass_to_pass),
            dropped, failed_tests=failed, log=log,
        )  # fmt: skip
