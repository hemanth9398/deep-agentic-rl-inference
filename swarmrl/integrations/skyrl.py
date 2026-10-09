"""SkyRL-train integration: the swarm as a step-wise SkyRL generator.

Checked against SkyRL main (NovaSky-AI/SkyRL, commit b896332):
  - `GeneratorInterface.generate(GeneratorInput) -> GeneratorOutput` with step-wise rows
    (`generator.step_wise_trajectories=true`, `is_last_step`), GRPO groups keyed by
    `trajectory_ids[i].instance_id`, advantage of a trajectory's last step broadcast to all its steps.
  - We emit one trajectory per (episode, role) with instance_id "<task_id>::<role>", so SkyRL's own
    GRPO computes reward minus the group mean per task AND per role: the doc's per-role normalization,
    without patching the trainer.

IMPORTANT (doc: "verify before building", item 1): at that commit SkyRL-train's inference backend
is vLLM only (`generator.inference_engine.backend: "vllm"`; weight sync over NCCL/IPC/delta to
vLLM servers). There is no SGLang engine path. Two ways forward:
  a) SkyRL drives vLLM: pass `SkyRLEnginePolicy` (wraps SkyRL's inference client) as the policy.
     Token-in/token-out and log-probs come from vLLM; same runtime, traces and rewards.
  b) Keep SGLang: run SGLang outside SkyRL, use `SGLangClient` + `DiskWeightSync` here, and use
     SkyRL-train only as the optimizer. This needs a SkyRL release that accepts external,
     non-vLLM servers for weight sync, or a custom sync hook. Verify on your pinned version.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from swarmrl.inference.client import Generation, PolicyClient
from swarmrl.rl.packing import pack_episode
from swarmrl.rl.schedule import StageConfig
from swarmrl.swarm.config import SamplingConfig
from swarmrl.swarm.runtime import EpisodeRunner
from swarmrl.traces.schema import Episode


class SkyRLEnginePolicy(PolicyClient):
    """PolicyClient over SkyRL's InferenceEngineInterface (token-in, token-out, with log-probs)."""

    def __init__(self, inference_engine_client, model: str | None = None):
        self.client = inference_engine_client
        self.model = model

    async def generate(self, input_ids, sampling: SamplingConfig, stop_ids, session_key, max_new_tokens) -> Generation:
        out = await self.client.generate(
            {
                "prompts": None,
                "prompt_token_ids": [list(input_ids)],
                "sampling_params": {
                    "temperature": sampling.temperature,
                    "top_p": sampling.top_p,
                    "max_tokens": max_new_tokens,
                    "stop_token_ids": list(stop_ids),
                    "logprobs": 0,  # vLLM: sampled-token log-prob only
                    "skip_special_tokens": False,
                    "include_stop_str_in_output": True,
                },
                "session_ids": [session_key],  # sticky routing for prefix-cache hits
            },
            model=self.model,
        )
        ids = list(out["response_ids"][0])
        lps = out.get("response_logprobs")
        if not lps or lps[0] is None:
            raise RuntimeError("inference engine returned no response_logprobs; enable logprobs in the engine config")
        return Generation(ids, list(lps[0]), out["stop_reasons"][0])


def episode_rows(ep: Episode, repetition: int, max_len: int | None = None) -> list[dict[str, Any]]:
    """Step-wise rows for one episode: one trajectory per role, prefix-shared turns packed."""
    rows: list[dict[str, Any]] = []
    if not ep.scored:
        # Infra error: one inert row in its own group (zero reward, zero loss mask -> zero gradient).
        return [{
            "prompt_token_ids": [0], "response_ids": [0], "loss_mask": [0], "rollout_logprobs": [0.0],
            "reward": 0.0, "instance_id": f"{ep.task_id}::infra::{ep.episode_id}", "repetition": repetition,
            "is_last_step": True, "stop_reason": "infra_error",
        }]  # fmt: skip
    samples = pack_episode(ep, max_len)
    by_role: dict[str, list] = {}
    for s in samples:
        by_role.setdefault(s.role, []).append(s)
    for role, ss in by_role.items():
        for i, s in enumerate(ss):
            rows.append({
                "prompt_token_ids": s.input_ids[: s.prompt_len],
                "response_ids": s.response_ids,
                "loss_mask": s.response_mask,
                "rollout_logprobs": s.response_logprobs,
                "reward": float(ep.team_reward or 0.0),
                "instance_id": f"{ep.task_id}::{role}",
                "repetition": repetition,
                "is_last_step": i == len(ss) - 1,
                "stop_reason": ep.termination.value,
            })  # fmt: skip
    return rows


def rows_to_generator_output(rows: Sequence[dict[str, Any]], metrics: dict | None = None) -> dict[str, Any]:
    from skyrl.train.generators.base import TrajectoryID

    return {
        "prompt_token_ids": [r["prompt_token_ids"] for r in rows],
        "response_ids": [r["response_ids"] for r in rows],
        "rewards": [r["reward"] for r in rows],
        "loss_masks": [r["loss_mask"] for r in rows],
        "stop_reasons": [r["stop_reason"] for r in rows],
        "rollout_metrics": metrics or {},
        "rollout_logprobs": [r["rollout_logprobs"] for r in rows],
        "trajectory_ids": [TrajectoryID(r["instance_id"], r["repetition"]) for r in rows],
        "is_last_step": [r["is_last_step"] for r in rows],
    }


@dataclass
class SwarmGenerator:
    """Implements skyrl.train.generators.base.GeneratorInterface (duck-typed to avoid a hard import).

    The training dataset's `env_extras` must carry {"task_id": ...}; `prompts` are ignored because
    the swarm builds every role's prompt itself.
    """

    runner: EpisodeRunner
    tasks_by_id: dict[str, object]
    max_concurrency: int = 64
    max_len: int | None = None
    on_episode: Callable[[Episode], None] | None = None  # TraceStore.put

    async def generate(self, input_batch: dict[str, Any]) -> dict[str, Any]:
        extras = input_batch.get("env_extras") or []
        traj = input_batch.get("trajectory_ids") or [None] * len(extras)
        sem = asyncio.Semaphore(self.max_concurrency)

        async def one(i: int) -> list[dict[str, Any]]:
            task = self.tasks_by_id[extras[i]["task_id"]]
            rep = traj[i].repetition_id if traj[i] is not None else i
            async with sem:
                ep = await self.runner.run(task, seed=rep, group_id=extras[i]["task_id"])
            if self.on_episode:
                await asyncio.to_thread(self.on_episode, ep)
            return episode_rows(ep, rep, self.max_len)

        per_ep = await asyncio.gather(*(one(i) for i in range(len(extras))))
        rows = [r for rs in per_ep for r in rs]
        return rows_to_generator_output(rows, {"swarm/rows": len(rows)})


def skyrl_overrides(stage: StageConfig, lora_rank: int = 32) -> dict[str, Any]:
    """Hydra-style overrides mapping a schedule stage onto SkyRL-train's config (field names at b896332)."""
    o: dict[str, Any] = {
        "generator.step_wise_trajectories": True,
        "generator.sampling_params.temperature": 1.0,
        "generator.sampling_params.top_p": 1.0,
        "generator.sampling_params.max_generate_length": stage.response_tokens,
        "trainer.algorithm.advantage_estimator": "grpo",
        "trainer.algorithm.grpo_norm_by_std": True,
        "trainer.algorithm.eps_clip_low": stage.clip_low,
        "trainer.algorithm.eps_clip_high": stage.clip_high,
        "trainer.algorithm.use_kl_loss": False,
        "trainer.algorithm.use_kl_in_reward": False,
        "trainer.algorithm.loss_reduction": "token_mean",
        "trainer.policy.model.lora.rank": lora_rank,
        "trainer.strategy": "fsdp",  # composable fully_shard (FSDP2)
        "trainer.placement.colocate_all": stage.colocated,
    }
    if stage.lr_scale == 0.0:
        o["trainer.policy.optimizer_config.lr"] = 0.0
    if stage.off_policy.mode in ("token_tis", "token_tis_reject"):
        o["trainer.algorithm.off_policy_correction.tis_ratio_type"] = "token"
        o["trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high"] = stage.off_policy.tis_cap
    if stage.off_policy.mode == "token_tis_reject":
        o["trainer.algorithm.off_policy_correction.token_mask_is_threshold_low"] = stage.off_policy.reject_low
        o["trainer.algorithm.off_policy_correction.token_mask_is_threshold_high"] = stage.off_policy.reject_high
    if stage.asynchronous:
        o["trainer.fully_async.max_staleness_steps"] = stage.max_staleness
    return o
