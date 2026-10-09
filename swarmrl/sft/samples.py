"""Transition-level SFT samples.

One sample per kept transition: that role's exact context, loss only on its output tokens
and end-of-turn token. Other roles' text and all tool outputs carry no loss. Over-length
samples are dropped, never truncated, so endings (verify, submit) are never cut off.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from swarmrl.traces.schema import Episode, ErrorKind

IGNORE = -100


@dataclass
class SFTSample:
    input_ids: list[int]
    labels: list[int]
    episode_id: str
    role: str
    turn: int


def build_sft_samples(
    episodes: Iterable[Episode], max_len: int, skip_format_error_turns: bool = True
) -> tuple[list[SFTSample], int]:
    out, dropped = [], 0
    for e in episodes:
        for t in e.transitions:
            if skip_format_error_turns and t.error_kind and ErrorKind(t.error_kind).is_format_error:
                continue  # a recoverable mistake inside a winning episode is still a mistake
            target = t.output_ids + t.closing_ids  # end-of-turn carries loss even if the server stripped it
            ids = t.input_ids + target
            if len(ids) > max_len:
                dropped += 1
                continue
            out.append(SFTSample(ids, [IGNORE] * len(t.input_ids) + target, e.episode_id, t.role, t.turn))
    return out, dropped


def write_jsonl(samples: Sequence[SFTSample], path: str) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def read_jsonl(path: str) -> list[SFTSample]:
    with open(path) as f:
        return [SFTSample(**json.loads(line)) for line in f if line.strip()]


def preference_pairs(episodes: Sequence[Episode]) -> list[tuple[str, str, str]]:
    """(task_id, chosen_episode_id, rejected_episode_id) from passes vs failures on the same task.

    Kept for an optional DPO stage; failures are never used as SFT targets.
    """
    wins, losses = defaultdict(list), defaultdict(list)
    for e in episodes:
        if not e.scored:
            continue
        (wins if e.resolved and e.submitted else losses)[e.task_id].append(e.episode_id)
    return [(t, w, lose) for t in wins for w, lose in zip(wins[t], losses.get(t, []), strict=False)]
