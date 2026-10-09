"""The iteration loop's decision rule (before any training).

Run path A with k samples per task and fixed seeds; change one thing; re-run with the same
seeds; keep the change only if per-task deltas agree and the intervals separate. Freeze the
swarm fingerprint once two changes in a row move nothing.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from swarmrl.evals.metrics import per_task, wilson
from swarmrl.traces.schema import Episode


@dataclass
class Comparison:
    rate_a: float
    rate_b: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    mean_task_delta: float
    delta_ci95: tuple[float, float]  # paired bootstrap over tasks
    tasks_improved: int
    tasks_regressed: int
    intervals_separate: bool
    deltas_agree: bool

    @property
    def keep(self) -> bool:
        return self.intervals_separate and self.deltas_agree and self.mean_task_delta > 0


def compare_runs(
    a: Sequence[Episode], b: Sequence[Episode], agree_frac: float = 0.7, n_boot: int = 2000, seed: int = 0
) -> Comparison:
    ta, tb = per_task(a), per_task(b)
    common = sorted(set(ta) & set(tb))
    if not common:
        raise ValueError("runs share no tasks")
    deltas = [tb[t][0] / tb[t][1] - ta[t][0] / ta[t][1] for t in common]
    ka, na = sum(ta[t][0] for t in common), sum(ta[t][1] for t in common)
    kb, nb = sum(tb[t][0] for t in common), sum(tb[t][1] for t in common)
    ci_a, ci_b = wilson(ka, na), wilson(kb, nb)
    mean_d = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    up, down = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
    moved = up + down
    agree = moved > 0 and max(up, down) / moved >= agree_frac and (up > down) == (mean_d > 0)
    separate = ci_b[0] > ci_a[1] or ci_a[0] > ci_b[1]
    return Comparison(ka / na, kb / nb, ci_a, ci_b, mean_d, (lo, hi), up, down, separate, agree)


class FreezeTracker:
    """Freeze the swarm fingerprint once `patience` changes in a row move nothing."""

    def __init__(self, patience: int = 2):
        self.patience = patience
        self.flat_streak = 0

    def record(self, cmp: Comparison) -> bool:
        self.flat_streak = 0 if cmp.keep else self.flat_streak + 1
        return self.frozen

    @property
    def frozen(self) -> bool:
        return self.flat_streak >= self.patience
