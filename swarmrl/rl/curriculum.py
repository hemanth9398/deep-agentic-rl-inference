"""Curriculum: spend rollouts where groups carry signal.

A group is informative only if its rewards differ: 1 - p^G - (1-p)^G (0.15 at p=0.02 and
0.99 at p=0.5 for G=8). Keep tasks with 0 < p < 1, weight sampling by p(1-p), revisit
mastered and hopeless tasks ~5% of the time, re-measure p with each new checkpoint.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from swarmrl.traces.schema import Episode


class Curriculum:
    def __init__(self, solve_rates: dict[str, float], explore: float = 0.05, seed: int = 0):
        self.p = dict(solve_rates)
        self.explore = explore
        self.rng = random.Random(seed)

    def update(self, solve_rates: dict[str, float], weight_new: float = 1.0) -> None:
        for t, p in solve_rates.items():
            old = self.p.get(t)
            self.p[t] = p if old is None else (1 - weight_new) * old + weight_new * p

    def informative(self) -> list[str]:
        return [t for t, p in self.p.items() if 0.0 < p < 1.0]

    def sample(self, b: int, allowed: Iterable[str] | None = None) -> list[str]:
        pool = set(allowed) if allowed is not None else set(self.p)
        inf = sorted(t for t in self.informative() if t in pool)
        rest = sorted(t for t in pool if t not in set(inf))
        chosen: list[str] = []
        while len(chosen) < b and (inf or rest):
            use_rest = rest and (not inf or self.rng.random() < self.explore)
            if use_rest:
                t = rest.pop(self.rng.randrange(len(rest)))
            else:
                w = [self.p[x] * (1 - self.p[x]) for x in inf]
                t = self.rng.choices(inf, weights=w, k=1)[0]
                inf.remove(t)
            chosen.append(t)
        return chosen

    def expected_informative_fraction(self, tasks: Sequence[str], g: int) -> float:
        from swarmrl.rl.advantages import informative_probability

        return sum(informative_probability(self.p.get(t, 0.0), g) for t in tasks) / max(len(tasks), 1)


def measure_solve_rates(episodes: Iterable[Episode]) -> dict[str, float]:
    k: dict[str, list[int]] = {}
    for e in episodes:
        if e.scored:
            v = k.setdefault(e.task_id, [0, 0])
            v[0] += int(e.resolved and e.submitted)
            v[1] += 1
    return {t: a / n for t, (a, n) in k.items()}


def min_turns_resolved(episodes: Iterable[Episode]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in episodes:
        if e.scored and e.resolved and e.submitted:
            out[e.task_id] = min(out.get(e.task_id, 10**9), e.num_turns)
    return out


def stage_pool(
    solve_rates: dict[str, float], lo: float, hi: float, max_turns: int | None = None, min_turns: dict[str, int] | None = None
) -> list[str]:
    """Stage data filter, e.g. stage 1: solve rate 0.125-0.875 and solvable within 10 turns."""
    out = []
    for t, p in solve_rates.items():
        if not lo <= p <= hi:
            continue
        if max_turns is not None and min_turns is not None and min_turns.get(t, 10**9) > max_turns:
            continue
        out.append(t)
    return sorted(out)
