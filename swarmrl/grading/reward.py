"""Reward design.

| Term          | Weight          | Rule                                                              |
| Team reward   | main            | 1 if every hidden test passes in a clean container, else 0        |
| Hack penalty  | hard zero       | any protected-file edit sets the reward to 0                      |
| Rubric terms  | small, optional | format validity, no repeated calls; never outweigh team reward    |
| Role shaping  | avoid           | only outcome-tied (reviewer verdict matches hidden tests)         |
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from swarmrl.grading.grader import GradeResult

# Rubric terms are bounded so that the best failing episode always scores below the worst
# resolved one: sum(|weights|) <= RUBRIC_BUDGET < 0.5.
RUBRIC_BUDGET = 0.2


@dataclass(frozen=True)
class RubricTerm:
    name: str
    weight: float
    fn: Callable[["object"], float]  # Episode -> score in [0, 1]


def validate_rubric(terms: Sequence[RubricTerm]) -> None:
    total = sum(abs(t.weight) for t in terms)
    if total > RUBRIC_BUDGET:
        raise ValueError(f"rubric weights sum to {total:.3f} > {RUBRIC_BUDGET}: they could outweigh the team reward")


def team_reward(grade: GradeResult) -> float:
    if grade.protected_edits:
        return 0.0  # hard zero on any hack attempt
    return 1.0 if grade.resolved else 0.0


def total_reward(grade: GradeResult, episode=None, rubric: Sequence[RubricTerm] = ()) -> tuple[float, dict[str, float]]:
    validate_rubric(rubric)
    base = team_reward(grade)
    details = {"team": base}
    if grade.protected_edits:
        return 0.0, details | {"hack_penalty": 1.0}
    extra = 0.0
    for t in rubric:
        s = min(max(float(t.fn(episode)), 0.0), 1.0)
        details[t.name] = s
        extra += t.weight * s
    return base + extra, details


# --- stock rubric terms -------------------------------------------------------------


def format_validity(episode) -> float:
    n = len(episode.transitions)
    if n == 0:
        return 0.0
    from swarmrl.traces.schema import ErrorKind

    bad = sum(1 for t in episode.transitions if t.error_kind and ErrorKind(t.error_kind).is_format_error)
    return 1.0 - bad / n


def no_repeated_calls(episode) -> float:
    return 1.0 if episode.repeat_calls == 0 else 0.0


def verifiers_rubric_term(name: str, weight: float, rubric, **kwargs) -> RubricTerm:
    """Wrap a Verifiers Rubric as a term. The Verifiers API moves between versions
    (legacy vs v1); this adapter only assumes a callable scoring the final diff."""

    def fn(episode) -> float:
        return float(rubric(episode.final_diff, **kwargs))

    return RubricTerm(name, weight, fn)
