"""Splits and contamination checks.

| Split    | Source                                                         | Used for                        |
| train    | procedural mutants, SWE-Gym / SWE-smith / R2E-Gym-style tasks  | rejection sampling, RL          |
| dev      | handwritten tasks                                              | harness and swarm iteration     |
| heldout  | tasks from repos never used for generation                     | stage gates, final numbers      |
| external | Environments Hub benchmarks via Verifiers                      | regression on general ability   |
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence


def contamination(tasks: Sequence) -> list[str]:
    """Heldout repos must never appear in train or dev. Returns human-readable violations."""
    repos: dict[str, set[str]] = defaultdict(set)
    for t in tasks:
        repos[t.split].add(t.repo)
    bad = sorted(repos["heldout"] & (repos["train"] | repos["dev"]))
    return [f"repo {r!r} is in heldout and in train/dev" for r in bad]


def assign_repo_splits(repos: Sequence[str], heldout_frac: float = 0.2, seed: int = 0) -> dict[str, str]:
    """Split by repository, not by task, so heldout repos are never used for generation."""
    rs = sorted(set(repos))
    random.Random(seed).shuffle(rs)
    k = max(1, round(len(rs) * heldout_frac)) if len(rs) > 1 else 0
    return {r: ("heldout" if i < k else "train") for i, r in enumerate(rs)}
