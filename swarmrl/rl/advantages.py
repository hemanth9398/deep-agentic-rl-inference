"""GRPO advantages for a shared-policy swarm.

Within each task's group (G episodes), advantage = team reward minus the group mean,
normalized per role: each role's statistics come from the episodes in which that role
acted. Every token a role produced in an episode gets that episode's advantage for that role.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from swarmrl.traces.schema import Episode


@dataclass
class AdvantageStats:
    groups: int
    informative_groups: int  # rewards differ within the group
    all_zero: int
    all_one: int


def group_key(e: Episode) -> str:
    return e.group_id or e.task_id


def grpo_advantages(
    episodes: Sequence[Episode], per_role_norm: bool = True, norm_by_std: bool = True, eps: float = 1e-6
) -> tuple[dict[tuple[str, str], float], AdvantageStats]:
    groups: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        if e.scored:
            groups[group_key(e)].append(e)
    adv: dict[tuple[str, str], float] = {}
    informative = all_zero = all_one = 0
    for eps_ in groups.values():
        rewards = [float(e.team_reward) for e in eps_]
        if max(rewards) - min(rewards) < 1e-12:
            all_zero += max(rewards) <= 0
            all_one += max(rewards) >= 1
        else:
            informative += 1
        role_sets = [set(t.role for t in e.transitions) for e in eps_]
        roles = set().union(*role_sets) if role_sets else set()
        for role in roles:
            pool = [r for r, rs in zip(rewards, role_sets, strict=True) if role in rs] if per_role_norm else rewards
            mu = sum(pool) / len(pool)
            sd = math.sqrt(sum((r - mu) ** 2 for r in pool) / len(pool))
            for e, r, rs in zip(eps_, rewards, role_sets, strict=True):
                if role not in rs:
                    continue
                a = r - mu
                if norm_by_std:
                    a = a / (sd + eps) if sd > 0 else 0.0
                adv[(e.episode_id, role)] = a
    return adv, AdvantageStats(len(groups), informative, all_zero, all_one)


def informative_probability(p: float, g: int) -> float:
    """Probability that a group of G episodes has differing rewards: 1 - p^G - (1-p)^G."""
    return 1.0 - p**g - (1.0 - p) ** g
