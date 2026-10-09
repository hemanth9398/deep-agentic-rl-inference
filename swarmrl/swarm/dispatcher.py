"""Staged dispatcher: a bounded pool per stage so container start-up and test runs overlap
with generation instead of idling the GPUs.

    init  (CPU pool)  -> run (in-flight episode cap; GPU via SGLang + CPU tools) -> reward (CPU pool)

Size the CPU pools explicitly: a swarm episode multiplies tool calls and test runs, and CPU
saturates before GPU memory does.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from swarmrl.sandbox.base import InfraError
from swarmrl.swarm.runtime import EpisodeRunner
from swarmrl.traces.schema import Episode


@dataclass(frozen=True)
class DispatchLimits:
    init_workers: int = 16
    max_in_flight: int = 256  # episodes in the run stage at once
    reward_workers: int = 16


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
        init_sem, run_sem, reward_sem = sems
        r = self.runner
        st = r.new_state(req.task, req.seed, req.split, req.group_id)
        try:
            async with run_sem:  # bounds total in-flight episodes (and live containers)
                async with init_sem:
                    await asyncio.to_thread(r.init_stage, st)
                await r.run_stage(st)
                async with reward_sem:
                    await asyncio.to_thread(r.reward_stage, st)
        except InfraError as e:
            r._infra_fail(st, e)
        st.episode.wall_time_s = time.monotonic() - st.t0
        if self.on_episode:
            await asyncio.to_thread(self.on_episode, st.episode)
        return st.episode

    async def run(self, requests: Sequence[EpisodeRequest]) -> list[Episode]:
        sems = (
            asyncio.Semaphore(self.limits.init_workers),
            asyncio.Semaphore(self.limits.max_in_flight),
            asyncio.Semaphore(self.limits.reward_workers),
        )
        return list(await asyncio.gather(*(self._one(q, sems) for q in requests)))

    def run_sync(self, requests: Sequence[EpisodeRequest]) -> list[Episode]:
        return asyncio.run(self.run(requests))
