"""Staged dispatcher: a bounded pool per stage so container start-up and test runs overlap
with generation instead of idling the GPUs.

    init  (CPU pool)  -> run (in-flight episode cap; GPU via SGLang + CPU tools) -> reward (CPU pool)

Size the CPU pools explicitly: a swarm episode multiplies tool calls and test runs, and CPU
saturates before GPU memory does.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from swarmrl.swarm.runtime import EpisodeRunner
from swarmrl.traces.schema import Episode

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchLimits:
    init_workers: int = 16
    max_in_flight: int = 256  # episodes in the run stage at once
    reward_workers: int = 16
    episode_timeout_s: float | None = None  # None: each task's [agent] timeout_sec


@dataclass(frozen=True)
class EpisodeRequest:
    task: object
    seed: int
    split: str | None = None
    group_id: str | None = None


class Dispatcher:
    def __init__(
        self,
        runner: EpisodeRunner,
        limits: DispatchLimits = DispatchLimits(),
        on_episode: Callable[[Episode], None] | None = None,
    ):
        self.runner = runner
        self.limits = limits
        self.on_episode = on_episode  # e.g. TraceStore.put: transitions stream to the store

    async def _one(self, req: EpisodeRequest, sems: tuple[asyncio.Semaphore, ...]) -> Episode:
        r = self.runner
        st = r.new_state(req.task, req.seed, req.split, req.group_id)
        ep = await r.execute(st, self.limits.episode_timeout_s or req.task.agent_timeout_s, sems)
        if self.on_episode:
            try:
                await asyncio.to_thread(self.on_episode, ep)
            except Exception:
                log.exception("on_episode failed for %s", ep.episode_id)
        return ep

    async def run(self, requests: Sequence[EpisodeRequest]) -> list[Episode]:
        sems = (
            asyncio.Semaphore(self.limits.init_workers),
            asyncio.Semaphore(self.limits.max_in_flight),
            asyncio.Semaphore(self.limits.reward_workers),
        )
        return list(await asyncio.gather(*(self._one(q, sems) for q in requests)))

    def run_sync(self, requests: Sequence[EpisodeRequest]) -> list[Episode]:
        return asyncio.run(self.run(requests))
