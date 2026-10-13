"""What one v2 episode records: every model call (for training) and every outcome (for credit)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from swarmrl.v2.router import GateDecision


@dataclass
class Call:
    """One generation by one role's model; exactly what that model's trainer will score."""

    role: str  # "planner" | "coder" | "router"
    input_ids: list[int]
    output_ids: list[int]
    rollout_logprobs: list[float]
    closing_ids: list[int] = field(default_factory=list)
    text: str = ""
    finish_reason: str = "stop"


@dataclass
class Verdict:
    p: float  # router's P(solved | candidate)
    reason: str
    valid: bool  # first output token was YES or NO
    call: Call | None = None


@dataclass
class PatchRecord:
    files: dict[str, str]
    call: Call
    valid: bool
    visible: dict[str, Any] = field(default_factory=dict)  # probe re-run + visible tests after applying
    verdict: Verdict | None = None
    hidden_resolved: bool | None = None  # clean-room hidden tests (None = not graded)
    diff: str = ""

    @property
    def y(self) -> float:
        return 1.0 if self.hidden_resolved else 0.0


@dataclass
class PlanRecord:
    text: str
    call: Call
    valid: bool
    hypothesis: str = ""
    files: list[str] = field(default_factory=list)
    probe: str = ""
    expect: str = "fail"
    probe_result: dict[str, Any] = field(default_factory=dict)
    verdict: Verdict | None = None
    patches: list[PatchRecord] = field(default_factory=list)

    @property
    def value(self) -> float | None:
        """Monte-Carlo value of the plan: fraction of its graded patches that solve the task."""
        graded = [p.y for p in self.patches if p.hidden_resolved is not None]
        return sum(graded) / len(graded) if graded else None


@dataclass
class Round:
    plans: list[PlanRecord]
    gate: GateDecision


@dataclass
class TriEpisode:
    episode_id: str
    task_id: str
    group_id: str
    seed: int
    rounds: list[Round] = field(default_factory=list)
    final: tuple[int, int, int] | None = None  # (round, plan, patch) of the submitted patch
    reward: float = 0.0
    error: str | None = None  # infra error: never scored
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def scored(self) -> bool:
        return self.error is None

    def final_patch(self) -> PatchRecord | None:
        if self.final is None:
            return None
        r, k, j = self.final
        return self.rounds[r].plans[k].patches[j]

    def main_plans(self) -> list[tuple[int, int]]:
        return [(r, k) for r, rd in enumerate(self.rounds) for k in rd.gate.main]
