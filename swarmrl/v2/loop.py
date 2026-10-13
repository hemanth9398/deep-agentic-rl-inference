"""One v2 RL step: G episodes per task, credit for each role, three separate updates, weights back to the servers.

Update order inside a step does not matter: the three losses share data, not parameters. Two-timescale note:
the router is a value estimate the other two depend on; giving it a higher learning rate (lr_scale) than the
actors is the usual actor-critic choice and keeps its decisions current.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from swarmrl.inference.weight_sync import WeightSync
from swarmrl.rl.loss import importance_weights
from swarmrl.rl.schedule import StageConfig
from swarmrl.v2.credit import CreditConfig, coder_targets, decision_stats, planner_targets, router_targets
from swarmrl.v2.records import TriEpisode
from swarmrl.v2.runtime import TriRunner
from swarmrl.v2.train import VerifierTrainer, to_samples

log = logging.getLogger(__name__)


@dataclass
class TriLoop:
    runner: TriRunner
    tasks: Sequence
    planner_trainer: object  # Trainer (PPO): logprobs(samples), update(...), save(version)
    coder_trainer: object
    router_trainer: VerifierTrainer
    stage: StageConfig
    group_size: int = 4
    credit: CreditConfig = field(default_factory=CreditConfig)
    router_lr_scale: float = 1.0
    syncs: dict[str, WeightSync] = field(default_factory=dict)  # role -> its server's weight sync
    on_metrics: Callable[[dict], None] | None = None
    version: int = 0
    history: list[dict] = field(default_factory=list)

    def rollout(self, step: int) -> list[TriEpisode]:
        async def go():
            jobs = [self.runner.run(t, seed=step * 1000 + g, group_id=f"{t.task_id}#{step}")
                    for t in self.tasks for g in range(self.group_size)]  # fmt: skip
            return await asyncio.gather(*jobs)

        return asyncio.run(go())

    def _ppo(self, trainer, targets) -> dict:
        targets = [t for t in targets if t.advantage != 0.0]
        if not targets:
            return {"stepped": 0.0, "samples": 0.0}
        samples, adv = to_samples(targets)
        old = trainer.logprobs(samples)
        masks = [np.asarray(s.loss_mask, dtype=np.float64) for s in samples]
        weights, eff = [], []
        for o, s, m in zip(old, samples, masks, strict=True):
            w, em = importance_weights(o, np.asarray(s.rollout_logprobs), m, self.stage.off_policy)
            weights.append(w)
            eff.append(em)
        out = trainer.update(samples, adv, old, weights, eff, self.stage)
        return {**out, "samples": float(len(samples))}

    def step(self, step: int) -> dict:
        episodes = self.rollout(step)
        for sync in self.syncs.values():
            sync.before_training()
        p_t, c_t, r_t = (
            planner_targets(episodes, self.credit),
            coder_targets(episodes, self.credit),
            router_targets(episodes, self.credit),
        )
        vt = self.runner.verdict_tokens
        upd = {
            "planner": self._ppo(self.planner_trainer, p_t),
            "coder": self._ppo(self.coder_trainer, c_t),
            "router": self.router_trainer.update_verifier(
                r_t, vt.yes, vt.no, self.credit.rank_weight, self.router_lr_scale * self.stage.lr_scale
            ),  # fmt: skip
        }
        self.version += 1
        for role, trainer in (("planner", self.planner_trainer), ("coder", self.coder_trainer),
                              ("router", self.router_trainer)):  # fmt: skip
            path = trainer.save(self.version)
            if role in self.syncs:
                self.syncs[role].sync(path, self.version)
        temperature = self.runner.gate.adapt()
        scored = [e for e in episodes if e.scored]
        n_plans = sum(len(rd.plans) for e in scored for rd in e.rounds)
        n_exec = sum(sum(rd.gate.executed) for e in scored for rd in e.rounds)
        n_shadow = sum(len(rd.gate.shadows) for e in scored for rd in e.rounds)
        metrics = {
            "step": step,
            "episodes": len(episodes),
            "infra_errors": len(episodes) - len(scored),
            "resolve_rate": float(np.mean([e.reward for e in scored])) if scored else 0.0,
            "rounds_per_episode": float(np.mean([len(e.rounds) for e in scored])) if scored else 0.0,
            "plans": n_plans,
            "executed_fraction": n_exec / max(n_plans, 1),
            "shadows": n_shadow,
            "gate/temperature": temperature,
            "targets/planner": len(p_t),
            "targets/coder": len(c_t),
            "targets/router": len(r_t),
            "tokens": {r: int(sum(e.tokens.get(r, 0) for e in scored)) for r in ("planner", "coder", "router")},
            **decision_stats(scored),
            **{f"train/{role}/{k}": v for role, d in upd.items() for k, v in d.items()},
        }
        self.history.append(metrics)
        if self.on_metrics:
            self.on_metrics(metrics)
        log.info("v2 step %d: resolve %.3f, false-reject %.3f", step, metrics["resolve_rate"],
                 metrics["router/false_reject_rate"])  # fmt: skip
        return metrics
