"""Run pytest inside a sandbox and read per-test outcomes from JUnit XML."""

from __future__ import annotations

import shlex
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from swarmrl.sandbox.base import ExecResult, Sandbox

JUNIT_PATH = ".swarmrl/junit.xml"


@dataclass
class TestRun:
    outcomes: dict[str, bool] = field(default_factory=dict)  # test id -> passed
    exec_result: ExecResult | None = None
    collection_error: bool = False

    @property
    def passed(self) -> set[str]:
        return {k for k, v in self.outcomes.items() if v}

    @property
    def failed(self) -> set[str]:
        return {k for k, v in self.outcomes.items() if not v}


def parse_junit(xml_text: str) -> dict[str, bool]:
    root = ET.fromstring(xml_text)
    out: dict[str, bool] = {}
    for case in root.iter("testcase"):
        tid = f"{case.get('classname', '')}::{case.get('name', '')}"
        bad = any(child.tag in ("failure", "error", "skipped") for child in case)
        out[tid] = not bad
    return out


def run_pytest(sb: Sandbox, targets: list[str], timeout: float = 600.0, extra_args: str = "") -> TestRun:
    sb.exec(f"rm -f {JUNIT_PATH} && mkdir -p .swarmrl", timeout=30)
    tgt = " ".join(shlex.quote(t) for t in targets)
    cmd = f"python -m pytest -q -p no:cacheprovider --junitxml={JUNIT_PATH} {extra_args} {tgt}"
    r = sb.exec(cmd, timeout=timeout)
    try:
        outcomes = parse_junit(sb.read_file(JUNIT_PATH))
    except (FileNotFoundError, OSError, ET.ParseError):
        outcomes = {}
    # Exit code 2/4 = interrupted / usage or collection error; tests did not run as specified.
    return TestRun(outcomes, r, collection_error=r.returncode in (2, 3, 4) or (not outcomes and r.returncode != 5))
