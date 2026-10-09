"""Staged RL schedule.

| Stage               | Mode                                 | G     | Turns | Resp. tokens | Off-policy correction |
| 0 correctness check | sync, lr 0, 2 steps                  | 4     | short | 8k           | none: gap = noise     |
| 1 short horizon     | sync, colocated, 1 update per step   | 8     | 10    | 12k          | none (on-policy)      |
| 2 longer horizon    | sync, colocated                      | 8     | 20    | 24k          | token-level IS        |
| 3 async             | disaggregated, staleness < 1         | 8-16  | 40    | 48k          | IS + rejection        |
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from swarmrl.rl.loss import OffPolicyConfig
from swarmrl.swarm.config import SwarmConfig


@dataclass(frozen=True)
class StageConfig:
    name: str
    group_size: int
    turn_budget: int
    response_tokens: int
    colocated: bool
    asynchronous: bool
    off_policy: OffPolicyConfig
    lr_scale: float = 1.0
    steps: int | None = None
    max_staleness: int = 0  # in optimizer steps; stage 3 keeps it under 1
    data_lo: float = 0.0
    data_hi: float = 1.0
    max_turns_to_solve: int | None = None
    clip_low: float = 0.2
    clip_high: float = 0.28

    def apply(self, swarm: SwarmConfig) -> SwarmConfig:
        """Stage budgets apply to the episode, not the protocol: role prompts and tools are untouched."""
        return replace(swarm, episode_turn_budget=self.turn_budget, episode_token_budget=self.response_tokens)


STAGES: tuple[StageConfig, ...] = (
    StageConfig("0_correctness", 4, 8, 8_000, True, False, OffPolicyConfig("none"), lr_scale=0.0, steps=2),
    StageConfig("1_short", 8, 10, 12_000, True, False, OffPolicyConfig("none"),
                data_lo=0.125, data_hi=0.875, max_turns_to_solve=10),
    StageConfig("2_long", 8, 20, 24_000, True, False, OffPolicyConfig("token_tis")),
    StageConfig("3_async", 16, 40, 48_000, False, True, OffPolicyConfig("token_tis_reject"), max_staleness=1),
)  # fmt: skip


def stage(name_or_index: str | int) -> StageConfig:
    if isinstance(name_or_index, int):
        return STAGES[name_or_index]
    for s in STAGES:
        if s.name == name_or_index or s.name.startswith(f"{name_or_index}_"):
            return s
    raise KeyError(name_or_index)
