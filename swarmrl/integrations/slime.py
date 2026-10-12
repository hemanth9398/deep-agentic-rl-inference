"""slime integration: Megatron training + SGLang rollouts, with the swarm as slime's generate function.

Checked against THUDM/slime v0.4.0 (main 05e3b01) and Megatron-Core 0.19:
  - `--custom-generate-function-path`: `async def generate(args, sample, sampling_params) -> Sample | list[Sample]`.
    slime calls it once per (prompt x n_samples_per_prompt); here one call runs one whole swarm episode.
  - A returned list holds one training sample per (episode, role): a role's turns extend its own context,
    so they pack into one sequence with `loss_mask` 1 on model tokens and 0 on observations
    (`swarmrl.rl.packing`). Every sample keeps the prompt's `group_index` and gets its own `rollout_id`.
  - `--custom-reward-post-process-path`: `fn(args, samples) -> (raw_rewards, rewards)`. slime's default
    normalizes per prompt group; `reward_post_process` normalizes per (group, role) over the episodes in
    which the role acted, exactly as `swarmrl.rl.advantages.grpo_advantages`. With
    `--advantage-estimator grpo` slime broadcasts each value to its sample's tokens.
  - Infra-error episodes become one `remove_sample` row: zero loss mask, excluded from group statistics.
  - Weights: colocated runs move Megatron shards to SGLang by CUDA IPC (`update_weights_from_tensor`);
    disaggregated runs (`--rollout-external-engine-addrs`, e.g. an llm-d pool) NCCL-broadcast them
    (`update_weights_from_distributed`), over RDMA when the pods have SR-IOV VFs (see deploy/).

Runtime settings come from the JSON file named by $SWARMRL_SLIME_CONFIG (`SlimeSwarmSettings`), because
slime passes only its own `args` to the hook. Build the launch command with `swarmrl slime-args`.

GiGPO is not wired here (it needs per-token advantages through `--custom-advantage-function-path`); A/B it
with `swarmrl rl --advantage gigpo` on the single-process trainer.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import os
import shlex
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from swarmrl.rl.packing import pack_episode
from swarmrl.rl.schedule import StageConfig
from swarmrl.traces.schema import Episode, Termination

log = logging.getLogger(__name__)

ENV = "SWARMRL_SLIME_CONFIG"
MAX_ROLLOUTS_PER_EPISODE = 64  # rollout ids are sample.index * 64 + role slot
TRUNCATED = {Termination.TURN_BUDGET, Termination.TOKEN_BUDGET, Termination.CONTEXT_LIMIT}


@dataclass
class SlimeSwarmSettings:
    tasks_dir: str
    sandbox: str = "k8s"  # "k8s" | "docker" | "local" (dev only)
    namespace: str = "swarmrl"
    pod_cpu: str = "2"
    pod_memory: str = "4Gi"
    pod_cpu_request: str = "500m"
    pod_memory_request: str = "1Gi"
    pod_ready_timeout: int = 900
    traces: str | None = None  # trace store for evals, SFT data and debugging
    tokenizer: str | None = None  # default: slime's --hf-checkpoint
    policy_url: str | None = None  # default: slime's router; set to the llm-d gateway when disaggregated
    routing_header: str | None = "X-SMG-Routing-Key"  # sgl-router consistent hashing; "x-session-id" for llm-d
    stage: int | str = 1
    single_agent: bool = False
    max_len: int | None = None  # drop (never truncate) packed samples longer than this
    max_in_flight: int = 64  # concurrent episodes (= sandboxes) per rollout process
    per_role_norm: bool = True
    norm_by_std: bool = True

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> SlimeSwarmSettings:
        path = path or os.environ.get(ENV)
        if not path:
            raise RuntimeError(f"set ${ENV} to a SlimeSwarmSettings JSON file")
        return cls(**json.loads(Path(path).read_text()))


# --- episode -> slime samples ----------------------------------------------------------------


def _status(ep: Episode, status_enum) -> Any:
    return status_enum.TRUNCATED if ep.termination in TRUNCATED else status_enum.COMPLETED


def episode_to_samples(ep: Episode, sample, roles: Sequence[str], max_len: int | None = None) -> list:
    """One slime Sample per (episode, role); `sample` is the prompt sample slime handed to `generate`.

    Works on any dataclass with slime's Sample fields (slime's own, or a stand-in in tests)."""
    status_enum = type(sample).Status
    base = (sample.index or 0) * MAX_ROLLOUTS_PER_EPISODE
    meta = {**(sample.metadata or {}), "swarm_episode": ep.episode_id, "swarm_task": ep.task_id,
            "swarm_termination": ep.termination.value, "raw_reward": float(ep.team_reward or 0.0)}  # fmt: skip
    if not ep.scored:
        # Never scored against the model: an inert row (zero loss mask) that the post-process skips.
        return [dataclasses.replace(
            sample, tokens=[0, 0], response_length=1, loss_mask=[0], rollout_log_probs=[0.0], reward=0.0,
            remove_sample=True, rollout_id=base, status=status_enum.FAILED,
            metadata={**meta, "swarm_role": None,
                      "swarm_infra_error": str(ep.metadata.get("infra_error", "infra_error"))},
        )]  # fmt: skip
    out = []
    for packed in pack_episode(ep, max_len):
        slot = roles.index(packed.role) if packed.role in roles else len(roles)
        out.append(dataclasses.replace(
            sample,
            tokens=list(packed.input_ids),
            response_length=len(packed.input_ids) - packed.prompt_len,
            loss_mask=list(packed.response_mask),
            rollout_log_probs=list(packed.response_logprobs),
            reward=float(ep.team_reward or 0.0),
            rollout_id=base + slot,
            status=_status(ep, status_enum),
            metadata={**meta, "swarm_role": packed.role, "swarm_turns": list(packed.transitions)},
        ))  # fmt: skip
    if not out:  # every role's sequence exceeded max_len: keep the group shape, train on nothing
        return [dataclasses.replace(sample, tokens=[0, 0], response_length=1, loss_mask=[0], rollout_log_probs=[0.0],
                                    reward=0.0, remove_sample=True, rollout_id=base, status=status_enum.TRUNCATED,
                                    metadata={**meta, "swarm_role": None})]  # fmt: skip
    return out


def _rid(s) -> int:
    return s.rollout_id if s.rollout_id is not None else s.index


def reward_post_process(
    args, samples: Sequence, per_role_norm: bool | None = None, norm_by_std: bool | None = None, eps: float = 1e-6
) -> tuple[list[float], list[float]]:
    """GRPO per (prompt group, role): reward minus the mean over the episodes where that role acted,
    divided by the population std; 0 for a group whose rewards are all equal. Matches `grpo_advantages`."""
    if per_role_norm is None or norm_by_std is None:
        st = _settings_or_none()
        per_role_norm = st.per_role_norm if (per_role_norm is None and st) else (per_role_norm is not False)
        norm_by_std = st.norm_by_std if (norm_by_std is None and st) else (norm_by_std is not False)
    raw = [float(s.reward or 0.0) if not isinstance(s.reward, dict) else float(s.reward.get("reward", 0.0))
           for s in samples]  # fmt: skip
    rollouts: dict[int, tuple[tuple, float]] = {}
    for s, r in zip(samples, raw, strict=True):
        if s.remove_sample:
            continue
        role = (s.metadata or {}).get("swarm_role")
        key = (s.group_index, role) if per_role_norm else (s.group_index,)
        prev = rollouts.setdefault(_rid(s), (key, r))
        if prev != (key, r):
            raise ValueError(f"rollout {_rid(s)} has samples in different groups or with different rewards")
    groups: dict[tuple, list[int]] = defaultdict(list)
    for rid, (key, _) in rollouts.items():
        groups[key].append(rid)
    adv: dict[int, float] = {}
    for rids in groups.values():
        pool = [rollouts[rid][1] for rid in rids]
        mu = sum(pool) / len(pool)
        sd = math.sqrt(sum((r - mu) ** 2 for r in pool) / len(pool))
        for rid in rids:
            a = rollouts[rid][1] - mu
            adv[rid] = (a / (sd + eps) if sd > 0 else 0.0) if norm_by_std else a
    return raw, [0.0 if s.remove_sample else adv[_rid(s)] for s in samples]


# --- the generate hook -----------------------------------------------------------------------


@dataclass
class _Context:
    runner: Any
    tasks: dict[str, Any]
    roles: tuple[str, ...]
    store: Any = None
    max_len: int | None = None
    max_in_flight: int = 64
    _sem: asyncio.Semaphore | None = field(default=None, repr=False)

    def semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:  # created inside slime's rollout event loop
            self._sem = asyncio.Semaphore(self.max_in_flight)
        return self._sem


_CTX: _Context | None = None


def _settings_or_none() -> SlimeSwarmSettings | None:
    return SlimeSwarmSettings.load() if os.environ.get(ENV) else None


def set_context(runner, tasks: Sequence, store=None, max_len: int | None = None, max_in_flight: int = 64) -> None:
    """Install a ready runner (tests, or embedding the swarm in another rollout function)."""
    global _CTX
    _CTX = _Context(runner, {t.task_id: t for t in tasks}, runner.cfg.role_names, store, max_len, max_in_flight)


def _build_context(args) -> _Context:
    from swarmrl.cli import _factory, _runner, _swarm, _valid_tasks, _validator
    from swarmrl.inference.client import SGLangClient
    from swarmrl.rl.schedule import stage as get_stage
    from swarmrl.tasks.harbor import load_tasks
    from swarmrl.traces.store import TraceStore

    st = SlimeSwarmSettings.load()
    a = SimpleNamespace(**dataclasses.asdict(st))
    factory = _factory(a)
    validator = _validator(st.tasks_dir, factory)
    url = st.policy_url or f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    a.tokenizer, a.server, a.checkpoint = st.tokenizer or args.hf_checkpoint, url, None
    stage_cfg = get_stage(int(st.stage) if str(st.stage).isdigit() else st.stage)
    cfg = stage_cfg.apply(_swarm(st.single_agent))
    policy = SGLangClient(url, routing_header=st.routing_header)
    runner = _runner(a, cfg, factory, validator, policy=policy, checkpoint=getattr(args, "hf_checkpoint", None))
    tasks = _valid_tasks(load_tasks(st.tasks_dir), validator)
    store = TraceStore(st.traces) if st.traces else None
    log.info("swarm ready: %d tasks, policy %s, sandbox %s", len(tasks), url, st.sandbox)
    return _Context(runner, {t.task_id: t for t in tasks}, cfg.role_names, store, st.max_len, st.max_in_flight)


def task_id_of(sample) -> str:
    meta = sample.metadata or {}
    return meta.get("task_id") or sample.label or str(sample.prompt)


async def generate(args, sample, sampling_params: dict, evaluation: bool = False) -> list:
    """slime custom generate function: one swarm episode for the task named by `sample`."""
    global _CTX
    if _CTX is None:
        _CTX = await asyncio.to_thread(_build_context, args)
    ctx = _CTX
    tid = task_id_of(sample)
    if tid not in ctx.tasks:
        raise KeyError(f"task {tid!r} is not a valid task in this run (check --prompt-data against tasks_dir)")
    runner = ctx.runner
    sp = sampling_params or {}
    if "temperature" in sp or "top_p" in sp:  # slime's --rollout-temperature / --rollout-top-p win
        s = runner.cfg.sampling
        runner.cfg = dataclasses.replace(runner.cfg, sampling=dataclasses.replace(
            s, temperature=sp.get("temperature", s.temperature), top_p=sp.get("top_p", s.top_p)))  # fmt: skip
    async with ctx.semaphore():
        ep = await runner.run(ctx.tasks[tid], seed=sample.index or 0, group_id=f"{tid}#{sample.group_index}",
                              split="eval" if evaluation else None)  # fmt: skip
    if ctx.store is not None:
        await asyncio.to_thread(ctx.store.put, ep)
    return episode_to_samples(ep, sample, ctx.roles, ctx.max_len)


# --- data and launch args ------------------------------------------------------------------


def prompt_rows(tasks: Sequence, solve_rates: dict[str, float] | None = None, lo: float = 0.0,
                hi: float = 1.0) -> list[dict]:  # fmt: skip
    """slime --prompt-data rows: the task id is the prompt (the swarm builds every role's real prompt)."""
    rows = []
    for t in tasks:
        p = (solve_rates or {}).get(t.task_id)
        if p is not None and not (lo <= p <= hi):
            continue
        rows.append({"prompt": t.task_id, "label": t.task_id, "metadata": {"task_id": t.task_id, "solve_rate": p}})
    return rows


def _sglang_flags(sglang_args: list[str]) -> list[str]:
    """Planner SGLang args -> slime flags: --tp-size becomes --rollout-num-gpus-per-engine, the rest gain --sglang-."""
    out, i = [], 0
    while i < len(sglang_args):
        flag = sglang_args[i]
        has_value = i + 1 < len(sglang_args) and not sglang_args[i + 1].startswith("--")
        val = [sglang_args[i + 1]] if has_value else []
        if flag == "--tp-size":
            out += ["--rollout-num-gpus-per-engine", *val]
        else:
            out += ["--sglang-" + flag[2:], *val]
        i += 2 if has_value else 1
    return out


def slime_args(plan, stage: StageConfig, *, hf_checkpoint: str, load: str, save: str, prompt_data: str,
               rollout_batch_size: int, num_rollout: int, max_response_len: int = 8192, lr: float = 1e-6,
               external_engines: Sequence[str] = (), save_interval: int = 20) -> list[str]:  # fmt: skip
    """train.py arguments for one stage: placement and parallelism from the planner, algorithm from the stage."""
    c = plan.cluster
    a = ["--actor-num-nodes", str(c.nodes), "--actor-num-gpus-per-node", str(c.gpus_per_node)]
    if external_engines:  # disaggregated: an llm-d / SGLang pool outside the job, NCCL weight broadcast
        a += ["--rollout-external-engine-addrs", *external_engines, "--update-weight-transport", "nccl"]
    elif plan.colocated:
        a += ["--colocate"]
    else:
        a += ["--rollout-num-gpus", str(plan.rollout.engines * plan.rollout.engine_tp)]
    a += plan.megatron_args()
    a += _sglang_flags(plan.sglang_args())
    a += ["--hf-checkpoint", hf_checkpoint, "--load", load, "--ref-load", load, "--save", save,
          "--save-interval", str(save_interval),
          "--prompt-data", prompt_data, "--input-key", "prompt", "--label-key", "label", "--metadata-key", "metadata",
          "--rollout-shuffle", "--num-rollout", str(num_rollout), "--rollout-batch-size", str(rollout_batch_size),
          "--n-samples-per-prompt", str(stage.group_size), "--num-steps-per-rollout", "1",
          "--rollout-max-response-len", str(max_response_len), "--rollout-temperature", "1.0", "--rollout-top-p", "1.0",
          "--advantage-estimator", "grpo", "--eps-clip", str(stage.clip_low), "--eps-clip-high", str(stage.clip_high),
          "--calculate-per-token-loss", "--clip-grad", "1.0", "--lr", str(lr * stage.lr_scale),
          "--custom-generate-function-path", "swarmrl.integrations.slime.generate",
          "--custom-reward-post-process-path", "swarmrl.integrations.slime.reward_post_process"]  # fmt: skip
    mode = stage.off_policy.mode
    if mode in ("token_tis", "token_tis_reject"):
        a += ["--use-tis", "--tis-clip", str(stage.off_policy.tis_cap)]
    if mode == "token_tis_reject":  # masked IS: tokens whose ratio leaves [low, high] get weight 0
        a += ["--custom-tis-function-path", "slime.backends.megatron_utils.loss.icepop_function",
              "--tis-clip-low", str(stage.off_policy.reject_low)]  # fmt: skip
        a[a.index("--tis-clip") + 1] = str(stage.off_policy.reject_high)
    return a


def launch_command(
    args: list[str],
    env: dict[str, str],
    slime_dir: str = "/root/slime",
    megatron_dir: str = "/root/Megatron-LM",
    ray_address: str = "http://127.0.0.1:8265",
) -> str:
    runtime_env = {"env_vars": {"PYTHONPATH": megatron_dir, **env}}
    return (f"ray job submit --address={ray_address} --runtime-env-json={shlex.quote(json.dumps(runtime_env))} "
            f"-- python3 {slime_dir}/train.py " + " ".join(shlex.quote(x) for x in args))  # fmt: skip


def check_hooks_importable(paths: Sequence[str] = ("generate", "reward_post_process")) -> list[Callable]:
    """What slime's load_function will resolve; fails fast in a broken image."""
    import importlib

    mod = importlib.import_module(__name__)
    return [getattr(mod, p) for p in paths]
