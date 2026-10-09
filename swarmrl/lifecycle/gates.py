"""End-to-end lifecycle: 8 stages, 7 gates, each checked with eval path A on heldout tasks.

    Environments  --[oracle agent resolves 100% of tasks]-->
    Harness/evals --[two changes in a row move nothing: freeze the swarm]-->
    SFT           --[SFT beats base on heldout; format errors drop]-->
    RL stage 0    --[log-prob gap at noise level; best alignment shift is 0]-->
    RL stage 1    --[reward plateaus and heldout resolve rate rises]-->
    RL stage 2    --[heldout rises again; mismatch KL stays flat]-->
    RL stage 3    --[beats SFT on heldout and external benchmarks]-->
    Deploy

A failed gate sends you back one stage, not forward with a weaker model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

STAGES = ("environments", "harness", "sft", "rl0", "rl1", "rl2", "rl3", "deploy")


@dataclass
class GateResult:
    stage: str
    passed: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def next_stage(self) -> str:
        i = STAGES.index(self.stage)
        return STAGES[i + 1] if self.passed else STAGES[max(i - 1, 0)]


def _ci_above(cand: dict, base: dict) -> bool:
    """Candidate's resolve-rate interval sits entirely above the baseline's."""
    return cand["resolve_ci95"][0] > base["resolve_ci95"][1]


def _mean_fmt(m: dict) -> float:
    rates = list(m["format_error_rate"].values())
    return sum(rates) / len(rates) if rates else 0.0


def gate_environments(oracle_resolved: Sequence[bool], contamination: Sequence[str] = ()) -> GateResult:
    reasons = []
    if not oracle_resolved:
        reasons.append("no tasks")
    elif not all(oracle_resolved):
        reasons.append(f"oracle agent resolved {sum(oracle_resolved)}/{len(oracle_resolved)} tasks (need 100%)")
    reasons += list(contamination)
    return GateResult("environments", not reasons, reasons)


def gate_harness(frozen: bool) -> GateResult:
    return GateResult("harness", frozen, [] if frozen else ["keep iterating: the last changes still moved the metric"])


def gate_sft(sft: dict, base: dict, format_drop: float = 0.5) -> GateResult:
    reasons = []
    if not _ci_above(sft, base):
        reasons.append(f"resolve {sft['resolve_rate']:.3f} does not clear base {base['resolve_rate']:.3f} (CI overlap)")
    b, s = _mean_fmt(base), _mean_fmt(sft)
    if b > 0 and s > b * (1 - format_drop):
        reasons.append(f"format error rate {s:.3f} did not drop sharply from {b:.3f}")
    return GateResult("sft", not reasons, reasons)


def gate_rl0(verdict) -> GateResult:  # swarmrl.rl.loop.Stage0Verdict
    return GateResult("rl0", verdict.passed, list(verdict.reasons))


def _plateau(xs: Sequence[float], window: int, tol: float) -> bool:
    if len(xs) < 2 * window:
        return False
    a = sum(xs[-2 * window : -window]) / window
    b = sum(xs[-window:]) / window
    return abs(b - a) <= tol


def gate_rl1(
    train_rewards: Sequence[float], heldout: dict, prev_heldout: dict, window: int = 10, tol: float = 0.01
) -> GateResult:
    reasons = []
    if not _plateau(train_rewards, window, tol):
        reasons.append("train reward has not plateaued")
    if not _ci_above(heldout, prev_heldout):
        reasons.append("heldout resolve rate did not rise above the SFT checkpoint")
    return GateResult("rl1", not reasons, reasons)


def gate_rl2(heldout: dict, prev_heldout: dict, mismatch_kl: Sequence[float], max_growth: float = 2.0) -> GateResult:
    reasons = []
    if not _ci_above(heldout, prev_heldout):
        reasons.append("heldout resolve rate did not rise again")
    if len(mismatch_kl) >= 4:
        q = len(mismatch_kl) // 4
        early = sum(mismatch_kl[:q]) / q
        late = sum(mismatch_kl[-q:]) / q
        if early > 0 and late > early * max_growth:
            reasons.append(f"mismatch KL grew {late / early:.1f}x ({early:.2e} -> {late:.2e})")
    return GateResult("rl2", not reasons, reasons)


def gate_rl3(heldout: dict, sft_heldout: dict, external: dict[str, float], sft_external: dict[str, float],
             external_tol: float = 0.01) -> GateResult:  # fmt: skip
    reasons = []
    if not _ci_above(heldout, sft_heldout):
        reasons.append("does not beat SFT on heldout")
    for k, v in sft_external.items():
        if external.get(k, float("-inf")) < v - external_tol:
            reasons.append(f"external benchmark {k} regressed: {external.get(k)} < {v}")
    return GateResult("rl3", not reasons, reasons)


@dataclass(frozen=True)
class DeployCandidate:
    name: str  # "swarm" | "single_agent"
    resolve_rate: float
    cost_per_task: float  # tokens or dollars, same unit for both


def choose_deployment(candidates: Sequence[DeployCandidate], budget: float) -> DeployCandidate:
    """Ship whichever scores higher at the allowed budget; ties go to the cheaper one."""
    ok = [c for c in candidates if c.cost_per_task <= budget]
    if not ok:
        raise ValueError("no candidate fits the budget")
    return max(ok, key=lambda c: (c.resolve_rate, -c.cost_per_task))
