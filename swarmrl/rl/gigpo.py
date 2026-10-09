"""GiGPO: Group-in-Group Policy Optimization (Feng et al., 2025), adapted to the swarm.

An experimental, opt-in alternative to the default per-role GRPO (`swarmrl rl --advantage gigpo`).
Same rollouts, same rewards, same loss; only the advantage on each token changes.

Two levels, both inside one task's group of G episodes:

  episode level  A_E(episode, role)  the existing per-role GRPO advantage (swarmrl.rl.advantages)
  step level     A_S(episode, turn)  turns taken from the SAME state in different episodes form an
                                     "anchor group"; each turn is compared with the others by its
                                     discounted return  R_t = gamma^(T-1-t) * team_reward

  token advantage = A_E + omega * A_S      (A_S = 0 for a turn whose state no other episode reached)

What counts as "the same state" (`state_key`):
  "env"      same role and the same history of tool calls, errors and observations so far in the
             episode, ignoring the model's free-text reasoning and run-to-run noise in tool output
             (timings, addresses, temp paths: see normalize_observation). Closest to the paper, which anchors on
             identical environment states. Two episodes that did the same things and saw the same
             outputs are at the same decision point even if they phrased their thinking differently.
  "context"  the role's exact input token ids. Strict: only identical prompts match, which in
             practice is the planner's first turn of every episode in the group.
  callable   (episode, transition) -> hashable, for experiments.

Not supported on the SkyRL path: SkyRL computes its own GRPO advantages from rewards, so GiGPO runs
only in the single-process loop (swarmrl.rl.loop / `swarmrl rl`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from swarmrl.rl.advantages import group_key, grpo_advantages
from swarmrl.rl.packing import PackedSample
from swarmrl.traces.schema import Episode, Transition

StateKeyFn = Callable[[Episode, Transition], Hashable]


@dataclass(frozen=True)
class GiGPOConfig:
    omega: float = 1.0  # weight of the step-level advantage (the paper uses 1)
    gamma: float = 0.95  # discount for step returns; < 1 prefers reaching the outcome in fewer turns
    step_norm_by_std: bool = True
    state_key: Literal["env", "context"] | StateKeyFn = "env"
    min_anchor_size: int = 2  # a state seen by fewer episodes than this gets no step-level advantage
    eps: float = 1e-6


@dataclass
class GiGPOStats:
    anchor_groups: int  # states reached by >= min_anchor_size episodes
    anchored_turns: int  # turns that received a step-level advantage
    total_turns: int
    mean_anchor_size: float
    max_anchor_size: int

    @property
    def anchored_fraction(self) -> float:
        return self.anchored_turns / self.total_turns if self.total_turns else 0.0


# --- state keys --------------------------------------------------------------------------


def _digest(obj) -> str:
    return hashlib.blake2b(json.dumps(obj, sort_keys=True, default=str).encode(), digest_size=16).hexdigest()


# Run-to-run noise in tool output that does not change the environment state. Without this, two
# episodes that did exactly the same thing never share a state (pytest prints wall-clock timings).
_NOISE = [
    (re.compile(r"\b\d+(?:\.\d+)?(?:s|ms|sec|seconds)\b"), "<dur>"),  # "in 0.02s", "12ms"
    (re.compile(r"\b0x[0-9a-fA-F]{6,}\b"), "<addr>"),  # object addresses
    (re.compile(r"/tmp/[^\s'\":]+"), "<tmp>"),  # sandbox temp paths
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"), "<ts>"),  # timestamps
]


def normalize_observation(text: str) -> str:
    for pat, rep in _NOISE:
        text = pat.sub(rep, text)
    return text


def context_key(ep: Episode, t: Transition) -> Hashable:
    return (t.role, hashlib.blake2b(np.asarray(t.input_ids, dtype=np.int64).tobytes(), digest_size=16).hexdigest())


def env_keys(ep: Episode) -> dict[int, Hashable]:
    """Key per turn: the role plus everything that happened before it, minus free-text reasoning."""
    out: dict[int, Hashable] = {}
    history: list = []
    for t in ep.transitions:
        out[t.turn] = (t.role, _digest(history))
        history.append((t.role, t.tool_call, t.error_kind, normalize_observation(t.observation)))
    return out


def _keys(ep: Episode, cfg: GiGPOConfig) -> dict[int, Hashable]:
    if cfg.state_key == "env":
        return env_keys(ep)
    fn = context_key if cfg.state_key == "context" else cfg.state_key
    return {t.turn: fn(ep, t) for t in ep.transitions}


# --- advantages ----------------------------------------------------------------------------


def discounted_returns(ep: Episode, gamma: float) -> dict[int, float]:
    """R_t = gamma^(T-1-t) * R for a terminal-only team reward, over the episode's turns (all roles)."""
    r = float(ep.team_reward or 0.0)
    n = len(ep.transitions)
    return {t.turn: (gamma ** (n - 1 - i)) * r for i, t in enumerate(ep.transitions)}


def step_advantages(
    episodes: Sequence[Episode], cfg: GiGPOConfig = GiGPOConfig()
) -> tuple[dict[tuple[str, int], float], GiGPOStats]:
    """A_S per (episode_id, turn), from anchor groups formed within each task's group."""
    groups: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        if e.scored:
            groups[group_key(e)].append(e)
    adv: dict[tuple[str, int], float] = {}
    sizes: list[int] = []
    total = 0
    for eps_ in groups.values():
        anchors: dict[Hashable, list[tuple[str, int, float]]] = defaultdict(list)
        for e in eps_:
            keys = _keys(e, cfg)
            rets = discounted_returns(e, cfg.gamma)
            for t in e.transitions:
                total += 1
                anchors[keys[t.turn]].append((e.episode_id, t.turn, rets[t.turn]))
        for members in anchors.values():
            if len({m[0] for m in members}) < cfg.min_anchor_size:
                continue
            sizes.append(len(members))
            vals = [m[2] for m in members]
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
            for eid, turn, v in members:
                a = v - mu
                if cfg.step_norm_by_std:
                    a = a / (sd + cfg.eps) if sd > 0 else 0.0
                adv[(eid, turn)] = a
    stats = GiGPOStats(
        anchor_groups=len(sizes),
        anchored_turns=len(adv),
        total_turns=total,
        mean_anchor_size=float(np.mean(sizes)) if sizes else 0.0,
        max_anchor_size=max(sizes, default=0),
    )
    return adv, stats


def gigpo_token_advantages(
    samples: Sequence[PackedSample], episodes: Sequence[Episode], cfg: GiGPOConfig = GiGPOConfig()
) -> tuple[list[np.ndarray], GiGPOStats]:
    """Per-token advantages for packed samples: A_E(episode, role) + omega * A_S(episode, turn)."""
    ep_adv, _ = grpo_advantages(episodes)
    st_adv, stats = step_advantages(episodes, cfg)
    out = []
    for s in samples:
        arr = np.zeros(len(s.input_ids))
        base = ep_adv.get((s.episode_id, s.role), 0.0)
        for turn, a, b in s.spans:
            arr[a:b] = base + cfg.omega * st_adv.get((s.episode_id, turn), 0.0)
        out.append(arr)
    return out, stats


def explain(episodes: Sequence[Episode], cfg: GiGPOConfig = GiGPOConfig()) -> list[dict]:
    """One row per turn with both estimators side by side, for inspecting a run's traces."""
    ep_adv, _ = grpo_advantages(episodes)
    st_adv, _ = step_advantages(episodes, cfg)
    rows = []
    for e in episodes:
        if not e.scored:
            continue
        keys = _keys(e, cfg)
        for t in e.transitions:
            a_e = ep_adv.get((e.episode_id, t.role), 0.0)
            a_s = st_adv.get((e.episode_id, t.turn))
            rows.append({
                "group": group_key(e), "episode": e.episode_id[:8], "reward": e.team_reward, "turn": t.turn,
                "role": t.role, "tool": (t.tool_call or {}).get("name"), "state": str(hash(keys[t.turn]))[-6:],
                "grpo": round(a_e, 4), "step": None if a_s is None else round(a_s, 4),
                "gigpo": round(a_e + cfg.omega * (a_s or 0.0), 4),
            })  # fmt: skip
    return rows
