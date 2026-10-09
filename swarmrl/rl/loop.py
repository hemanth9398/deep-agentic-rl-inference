"""One RL step, end to end (reference orchestration; SkyRL-train drives the same steps in production).

1. Sample tasks: the curriculum picks B tasks; each runs G episodes (one GRPO group per task).
2. Sync weights: push the latest weights into SGLang and flush its KV cache.
3. Roll out: init/run/reward stages; transitions stream to the trace store.
4. Assemble: team reward on every transition; pack prefix-sharing transitions; keep rollout log-probs.
5. Score: trainer log-probs over the same ids; the gap to rollout log-probs is logged as mismatch KL.
6. Advantages: reward minus group mean, normalized per role, broadcast over that role's tokens.
7. Update: clipped PG on policy tokens (clip 0.2 / 0.28), no KL penalty, one optimizer step on LoRA.
8. Log and repeat; every N steps run eval path A on heldout.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from swarmrl.rl.advantages import grpo_advantages
from swarmrl.rl.curriculum import Curriculum, measure_solve_rates
from swarmrl.rl.loss import alignment_shift, entropy_from_logprobs, importance_weights, mismatch_kl
from swarmrl.rl.packing import PackedSample, pack_episodes
from swarmrl.rl.schedule import StageConfig
from swarmrl.swarm.dispatcher import Dispatcher, EpisodeRequest
from swarmrl.traces.schema import Episode


class Trainer(Protocol):
    def logprobs(self, samples: Sequence[PackedSample]) -> list[np.ndarray]:
        """Per sample, log p(input_ids[i]) for every position (aligned with rollout_logprobs)."""

    def update(
        self,
        samples: Sequence[PackedSample],
        advantages: list[np.ndarray],
        old_logprobs: list[np.ndarray],
        is_weights: list[np.ndarray],
        masks: list[np.ndarray],
        stage: StageConfig,
    ) -> dict: ...

    def save(self, version: int) -> str: ...


class WeightSync(Protocol):
    def before_training(self) -> None: ...

    def sync(self, checkpoint_path: str, version: int) -> None: ...


@dataclass
class RLLoop:
    dispatcher: Dispatcher
    trainer: Trainer
    weight_sync: WeightSync
    curriculum: Curriculum
    stage: StageConfig
    tasks_by_id: dict[str, object]
    batch_tasks: int = 16
    max_len: int | None = None
    eval_every: int | None = None
    eval_fn: Callable[[int], dict] | None = None
    log: Callable[[dict], None] = print
    checkpoint: str | None = None
    version: int = 0
    history: list[dict] = field(default_factory=list)
    allowed_tasks: Sequence[str] | None = None

    def rollout(self, step: int) -> list[Episode]:
        task_ids = self.curriculum.sample(self.batch_tasks, self.allowed_tasks)
        reqs = [
            EpisodeRequest(self.tasks_by_id[t], seed=step * 1000 + g, group_id=f"step{step}:{t}")
            for t in task_ids
            for g in range(self.stage.group_size)
        ]
        return self.dispatcher.run_sync(reqs)

    def step(self, step: int) -> dict:
        if self.checkpoint is not None:
            self.weight_sync.sync(self.checkpoint, self.version)  # also flushes the KV cache
        episodes = self.rollout(step)
        samples = pack_episodes(episodes, self.max_len)
        adv_map, adv_stats = grpo_advantages(episodes)
        samples = [s for s in samples if (s.episode_id, s.role) in adv_map]
        for s in samples:
            s.advantage = adv_map[(s.episode_id, s.role)]

        self.weight_sync.before_training()
        old = self.trainer.logprobs(samples)
        masks = [np.asarray(s.loss_mask, dtype=np.float64) for s in samples]
        roll = [np.asarray(s.rollout_logprobs) for s in samples]
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
        m_all, old_all, roll_all = cat(masks), cat(old), cat(roll)
        kl = float(mismatch_kl(old_all, roll_all, m_all)) if m_all.size else 0.0
        shift = alignment_shift(old_all, roll_all, m_all) if m_all.size else 0

        weights, eff_masks = [], []
        for o, r, m in zip(old, roll, masks):
            w, em = importance_weights(o, r, m, self.stage.off_policy)
            weights.append(w)
            eff_masks.append(em)
        advs = [np.full(len(s.input_ids), s.advantage) * m for s, m in zip(samples, masks)]
        upd = self.trainer.update(samples, advs, old, weights, eff_masks, self.stage)

        self.version += 1
        self.checkpoint = self.trainer.save(self.version)
        scored = [e for e in episodes if e.scored]
        rates = measure_solve_rates(scored)
        self.curriculum.update(rates, weight_new=0.5)
        metrics = {
            "step": step,
            "stage": self.stage.name,
            "episodes": len(episodes),
            "infra_errors": len(episodes) - len(scored),
            "reward_mean": float(np.mean([e.team_reward for e in scored])) if scored else 0.0,
            "resolve_rate": float(np.mean([e.resolved for e in scored])) if scored else 0.0,
            "groups": adv_stats.groups,
            "informative_groups": adv_stats.informative_groups,
            "all_zero_groups": adv_stats.all_zero,
            "all_one_groups": adv_stats.all_one,
            "samples": len(samples),
            "policy_tokens": int(m_all.sum()),
            "mismatch_kl": kl,
            "alignment_shift": shift,
            "entropy_proxy": entropy_from_logprobs(roll_all, m_all) if m_all.size else 0.0,
            "tokens_by_role": _tokens_by_role(scored),
            **{f"train/{k}": v for k, v in upd.items()},
        }
        if self.eval_every and self.eval_fn and (step + 1) % self.eval_every == 0:
            metrics["eval"] = self.eval_fn(step)
        self.history.append(metrics)
        self.log(metrics)
        return metrics

    def run(self, steps: int | None = None) -> list[dict]:
        n = steps if steps is not None else (self.stage.steps or 1)
        return [self.step(i) for i in range(n)]


def _tokens_by_role(episodes: Sequence[Episode]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in episodes:
        for r, c in e.tokens_by_role().items():
            out[r] = out.get(r, 0) + c
    return out


@dataclass(frozen=True)
class Stage0Verdict:
    passed: bool
    reasons: list[str]


def stage0_verdict(history: Sequence[dict], kl_tol: float = 1e-3) -> Stage0Verdict:
    """Stage 0 (lr 0): the trainer must reproduce the sampler's log-probs on the same ids."""
    reasons = []
    for m in history:
        if m["alignment_shift"] != 0:
            reasons.append(f"step {m['step']}: best alignment shift is {m['alignment_shift']}, not 0 (off-by-one)")
        if m["mismatch_kl"] > kl_tol:
            reasons.append(f"step {m['step']}: mismatch KL {m['mismatch_kl']:.2e} > {kl_tol:.0e}")
        if m["policy_tokens"] == 0:
            reasons.append(f"step {m['step']}: no policy tokens")
    return Stage0Verdict(not reasons, reasons)
