"""Pack prefix-sharing transitions back into one masked sample.

A role's turns extend its own context: turn k+1's input = turn k's input + output (+ closing
ids) + the new observation. Those transitions collapse into one sequence with loss on the
policy tokens only. Different roles' contexts are not prefixes of each other, so a swarm
episode yields one sample per role (more if a context was ever rebuilt).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from swarmrl.traces.schema import Episode, Transition


@dataclass
class PackedSample:
    episode_id: str
    task_id: str
    group_id: str | None
    role: str
    prompt_len: int  # tokens before the first policy token
    input_ids: list[int]  # full sequence
    loss_mask: list[int]  # 1 on policy-sampled tokens, 0 on prompt/observations/closing ids
    rollout_logprobs: list[float]  # aligned with input_ids; 0.0 where loss_mask == 0
    reward: float = 0.0
    advantage: float = 0.0
    transitions: list[int] = field(default_factory=list)  # episode turn indices packed here
    # (turn, start, end): where each packed turn's policy tokens sit in input_ids, so per-turn
    # advantages (GiGPO's step level) can be placed on exactly that turn's tokens.
    spans: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def response_ids(self) -> list[int]:
        return self.input_ids[self.prompt_len :]

    @property
    def response_mask(self) -> list[int]:
        return self.loss_mask[self.prompt_len :]

    @property
    def response_logprobs(self) -> list[float]:
        return self.rollout_logprobs[self.prompt_len :]

    def num_policy_tokens(self) -> int:
        return sum(self.loss_mask)


def _new(ep: Episode, t: Transition) -> PackedSample:
    ids = t.input_ids + t.output_ids + t.closing_ids
    mask = [0] * len(t.input_ids) + [1] * len(t.output_ids) + [0] * len(t.closing_ids)
    lps = [0.0] * len(t.input_ids) + list(t.rollout_logprobs) + [0.0] * len(t.closing_ids)
    start = len(t.input_ids)
    return PackedSample(ep.episode_id, ep.task_id, ep.group_id, t.role, start, ids, mask, lps,
                        transitions=[t.turn], spans=[(t.turn, start, start + len(t.output_ids))])  # fmt: skip


def pack_episode(ep: Episode, max_len: int | None = None) -> list[PackedSample]:
    by_role: dict[str, list[Transition]] = defaultdict(list)
    for t in ep.transitions:
        by_role[t.role].append(t)
    out: list[PackedSample] = []
    for ts in by_role.values():
        cur: PackedSample | None = None
        for t in ts:
            n = len(cur.input_ids) if cur else 0
            if cur is not None and len(t.input_ids) >= n and t.input_ids[:n] == cur.input_ids:
                obs = t.input_ids[n:]
                start = n + len(obs)
                cur.spans.append((t.turn, start, start + len(t.output_ids)))
                cur.input_ids += obs + t.output_ids + t.closing_ids
                cur.loss_mask += [0] * len(obs) + [1] * len(t.output_ids) + [0] * len(t.closing_ids)
                cur.rollout_logprobs += [0.0] * len(obs) + list(t.rollout_logprobs) + [0.0] * len(t.closing_ids)
                cur.transitions.append(t.turn)
            else:
                if cur is not None:
                    out.append(cur)
                cur = _new(ep, t)
        if cur is not None:
            out.append(cur)
    if max_len is not None:
        out = [s for s in out if len(s.input_ids) <= max_len]  # drop, never truncate
    return out


def pack_episodes(episodes: Sequence[Episode], max_len: int | None = None) -> list[PackedSample]:
    out: list[PackedSample] = []
    for e in episodes:
        if not e.scored:
            continue
        for s in pack_episode(e, max_len):
            s.reward = float(e.team_reward or 0.0)
            out.append(s)
    return out
