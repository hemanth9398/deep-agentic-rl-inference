"""Rejection sampling (uses the runtime, not the trainer).

Run the frozen swarm on the train split at temperature 0.8-1.0 with SGLang serving the base
model. Adaptive: stop a task at `target_successes`, keep sampling hard tasks up to
`max_attempts`. Keep every episode: passes feed SFT, failures become DPO negatives, and
solve rates seed the RL curriculum.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from swarmrl.swarm.dispatcher import Dispatcher, EpisodeRequest
from swarmrl.traces.schema import Episode


@dataclass
class AdaptiveSampler:
    tasks: Sequence
    target_successes: int = 4
    max_attempts: int = 16
    per_round: int = 4  # attempts per unfinished task per round
    attempts: Counter = field(default_factory=Counter)
    successes: Counter = field(default_factory=Counter)

    def done(self, task_id: str) -> bool:
        return self.successes[task_id] >= self.target_successes or self.attempts[task_id] >= self.max_attempts

    def next_requests(self, round_idx: int) -> list[EpisodeRequest]:
        reqs = []
        for t in self.tasks:
            if self.done(t.task_id):
                continue
            n = min(self.per_round, self.max_attempts - self.attempts[t.task_id])
            base = self.attempts[t.task_id]
            reqs += [EpisodeRequest(t, seed=base + i, split=t.split, group_id=f"rs{round_idx}:{t.task_id}") for i in range(n)]
        return reqs

    def record(self, episodes: Sequence[Episode]) -> None:
        for e in episodes:
            if not e.scored:
                continue  # infra errors do not count as attempts
            self.attempts[e.task_id] += 1
            self.successes[e.task_id] += int(e.resolved and e.submitted)

    def solve_rates(self) -> dict[str, float]:
        return {t.task_id: self.successes[t.task_id] / self.attempts[t.task_id] for t in self.tasks if self.attempts[t.task_id]}

    def never_solved(self) -> list[str]:
        return [t.task_id for t in self.tasks if self.attempts[t.task_id] >= self.max_attempts and not self.successes[t.task_id]]


def run_rejection_sampling(dispatcher: Dispatcher, sampler: AdaptiveSampler, max_rounds: int = 32) -> list[Episode]:
    """Runs rounds until every task is done. Episodes stream to the store via dispatcher.on_episode."""
    out: list[Episode] = []
    for r in range(max_rounds):
        reqs = sampler.next_requests(r)
        if not reqs:
            break
        eps = dispatcher.run_sync(reqs)
        sampler.record(eps)
        out.extend(eps)
    return out


def cap_teacher(episodes: Sequence[Episode], max_fraction: float = 0.5) -> list[Episode]:
    """Teacher episodes are allowed only for tasks the student never solves (choose them upstream)
    and only if competition rules allow; here they are capped at `max_fraction` of the dataset."""
    student = [e for e in episodes if not e.teacher]
    teacher = [e for e in episodes if e.teacher]
    limit = int(len(student) * max_fraction / (1 - max_fraction)) if max_fraction < 1 else len(teacher)
    return student + teacher[:limit]
