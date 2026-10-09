"""Episode-level filtering, then dedupe and a per-task cap.

| Keep only if                          | Why                                              |
| Resolved and submitted                | the point of rejection sampling                  |
| No protected-file edits               | never train on a hack                            |
| Tester ran tests after the last edit  | outcome-only filtering rewards lucky guesses     |
| No role ping-pong or repeated calls   | teaches loops otherwise                          |
| Turns under the cap                   | long wins are often wandering                    |
| Fingerprint matches the frozen swarm  | trained protocol must equal deployed protocol    |
| Not a heldout task                    | contamination                                    |
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from swarmrl.evals.metrics import tester_ran_tests_after_last_edit
from swarmrl.traces.schema import Episode


@dataclass
class FilterConfig:
    fingerprint: str
    turn_cap: int = 30
    heldout_task_ids: frozenset[str] = frozenset()
    max_per_task: int = 4
    require_tester: bool = True


def rules(cfg: FilterConfig) -> list[tuple[str, Callable[[Episode], bool]]]:
    return [
        ("resolved_and_submitted", lambda e: e.resolved and e.submitted),
        ("no_protected_edits", lambda e: not e.protected_edits),
        ("tester_ran_tests_after_last_edit", lambda e: not cfg.require_tester or tester_ran_tests_after_last_edit(e)),
        ("no_ping_pong_or_repeats", lambda e: e.ping_pong_events == 0 and e.repeat_calls == 0),
        ("turns_under_cap", lambda e: e.num_turns <= cfg.turn_cap),
        ("fingerprint_matches", lambda e: e.swarm_fingerprint == cfg.fingerprint),
        ("not_heldout", lambda e: e.split != "heldout" and e.task_id not in cfg.heldout_task_ids),
    ]


@dataclass
class FilterReport:
    kept: list[Episode]
    dropped: Counter = field(default_factory=Counter)  # first failing rule -> count
    deduped: int = 0
    capped: int = 0


def action_signature(e: Episode) -> str:
    seq = [(t.role, t.tool_call["name"] if t.tool_call else None, t.tool_call["arguments"] if t.tool_call else None)
           for t in e.transitions]  # fmt: skip
    return hashlib.sha256(json.dumps(seq, sort_keys=True).encode()).hexdigest()


def filter_episodes(episodes: Sequence[Episode], cfg: FilterConfig) -> FilterReport:
    rs = rules(cfg)
    rep = FilterReport([])
    survivors = []
    for e in episodes:
        failed = next((name for name, ok in rs if not ok(e)), None)
        if failed:
            rep.dropped[failed] += 1
        else:
            survivors.append(e)
    # dedupe identical action sequences, then cap per task (shortest episodes first)
    seen: set[tuple[str, str]] = set()
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for e in survivors:
        key = (e.task_id, action_signature(e))
        if key in seen:
            rep.deduped += 1
            continue
        seen.add(key)
        by_task[e.task_id].append(e)
    for eps in by_task.values():
        eps.sort(key=lambda e: e.num_turns)
        rep.kept.extend(eps[: cfg.max_per_task])
        rep.capped += max(0, len(eps) - cfg.max_per_task)
    return rep
