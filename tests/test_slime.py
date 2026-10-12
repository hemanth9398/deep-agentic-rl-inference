"""slime integration: swarm episodes -> slime Samples, per-role GRPO post-process, launch args, routing header."""

import asyncio
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest
from fakes.fake_sglang import StubSGLang

from swarmrl.inference.client import SGLangClient
from swarmrl.integrations import slime
from swarmrl.parallel.planner import PRESETS, ClusterSpec, TrainSpec, plan
from swarmrl.rl.advantages import grpo_advantages
from swarmrl.rl.schedule import stage
from swarmrl.swarm.config import SamplingConfig, default_swarm_config
from swarmrl.traces.schema import Episode, Termination


@dataclass
class Sample:
    """The fields of slime.utils.types.Sample (v0.4.0) that the integration reads or writes."""

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    group_index: int | None = None
    index: int | None = None
    rollout_id: int | None = None
    prompt: str = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | None = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    remove_sample: bool = False
    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)


class Store:
    def __init__(self):
        self.episodes: list[Episode] = []

    def put(self, ep):
        self.episodes.append(ep)


def _run_group(make_runner, task, script, g=4, group_index=7):
    runner = make_runner(script)
    store = Store()
    slime.set_context(runner, [task], store=store, max_in_flight=2)
    prompts = [Sample(group_index=group_index, index=group_index * g + i, prompt=task.task_id, label=task.task_id,
                      metadata={"task_id": task.task_id}) for i in range(g)]  # fmt: skip

    async def go():
        return await asyncio.gather(*(slime.generate(SimpleNamespace(), p, {"temperature": 1.0}) for p in prompts))

    try:
        return asyncio.run(go()), store.episodes
    finally:
        slime._CTX = None


def test_generate_returns_one_masked_sample_per_role(make_runner, task, mixed_script):
    groups, episodes = _run_group(make_runner, task, mixed_script)
    assert len(groups) == 4 and {e.team_reward for e in episodes} == {0.0, 1.0}
    roles = default_swarm_config().role_names
    for out, ep in zip(groups, sorted(episodes, key=lambda e: e.seed), strict=True):
        assert {s.metadata["swarm_role"] for s in out} == {t.role for t in ep.transitions}
        for s in out:
            n = s.response_length
            assert len(s.loss_mask) == n == len(s.rollout_log_probs) and n < len(s.tokens)
            assert all(lp == 0.0 for lp, m in zip(s.rollout_log_probs, s.loss_mask, strict=True) if m == 0)
            assert s.loss_mask[0] == 1  # the response starts at the role's first sampled token
            assert s.group_index == 7 and s.reward == ep.team_reward and s.status == Sample.Status.COMPLETED
            assert s.rollout_id == s.index * slime.MAX_ROLLOUTS_PER_EPISODE + roles.index(s.metadata["swarm_role"])
        # every policy token of the episode is trained exactly once
        assert sum(sum(s.loss_mask) for s in out) == sum(len(t.output_ids) for t in ep.transitions)
    rids = [s.rollout_id for out in groups for s in out]
    assert len(rids) == len(set(rids))


def test_reward_post_process_matches_per_role_grpo(make_runner, task, mixed_script):
    groups, episodes = _run_group(make_runner, task, mixed_script, g=6)
    flat = [s for out in groups for s in out]
    raw, adv = slime.reward_post_process(None, flat, per_role_norm=True, norm_by_std=True)
    for e in episodes:
        e.group_id = "g"
    want, _ = grpo_advantages(episodes)
    by_seed = {e.seed: e.episode_id for e in episodes}
    for s, r, a in zip(flat, raw, adv, strict=True):
        assert r == s.reward
        assert a == pytest.approx(want[(by_seed[s.index], s.metadata["swarm_role"])])
    assert {round(a, 4) for a in adv} == {1.0, -1.0}  # 3 solved / 3 failed: +-1 after std normalization


def test_infra_errors_are_inert_and_excluded():
    prompt = Sample(group_index=0, index=3, metadata={"task_id": "t"})
    ep = Episode("e", "t", "train", "fp", "ckpt", 0, termination=Termination.INFRA_ERROR,
                 metadata={"infra_error": "pod never became ready"})  # fmt: skip
    (row,) = slime.episode_to_samples(ep, prompt, ("planner",))
    assert row.remove_sample and row.loss_mask == [0] and row.status == Sample.Status.FAILED
    assert row.metadata["swarm_infra_error"] == "pod never became ready"
    ok = [dataclasses.replace(prompt, index=i, rollout_id=i, reward=float(i % 2), metadata={"swarm_role": "coder"})
          for i in range(4)]  # fmt: skip
    _, adv = slime.reward_post_process(None, [row, *ok], per_role_norm=True, norm_by_std=True)
    assert adv[0] == 0.0 and adv[1:] == pytest.approx(
        [-1.0, 1.0, -1.0, 1.0], abs=1e-4
    )  # the infra row moves no statistic


def test_post_process_rejects_inconsistent_rollouts():
    a = Sample(group_index=0, index=0, rollout_id=5, reward=1.0, metadata={"swarm_role": "coder"})
    b = dataclasses.replace(a, reward=0.0)
    with pytest.raises(ValueError, match="different rewards"):
        slime.reward_post_process(None, [a, b], per_role_norm=True, norm_by_std=True)


def test_budget_terminations_are_truncated():
    ep = Episode("e", "t", "train", "fp", "ckpt", 0, termination=Termination.TURN_BUDGET, team_reward=0.0)
    assert slime._status(ep, Sample.Status) == Sample.Status.TRUNCATED


@pytest.mark.parametrize("idx,expect,absent", [
    (0, ["--lr 0.0", "--colocate"], ["--use-tis"]),
    (1, ["--eps-clip-high 0.28", "--calculate-per-token-loss"], ["--use-tis"]),
    (2, ["--use-tis --tis-clip 2.0"], ["--custom-tis-function-path"]),
    (3, ["--custom-tis-function-path slime.backends.megatron_utils.loss.icepop_function --tis-clip-low 0.5",
         "--tis-clip 2.0"], []),
])  # fmt: skip
def test_slime_args_per_stage(idx, expect, absent):
    p = plan(PRESETS["qwen3-30b-a3b"], ClusterSpec(nodes=1), TrainSpec())
    a = slime.slime_args(p, stage(idx), hf_checkpoint="/m/hf", load="/m/dist", save="/ckpt", prompt_data="/d.jsonl",
                         rollout_batch_size=32, num_rollout=100)  # fmt: skip
    joined = " ".join(a)
    assert all(x in joined for x in expect)
    assert not any(x in a for x in absent)
    assert a[a.index("--n-samples-per-prompt") + 1] == str(stage(idx).group_size)
    assert a[a.index("--rollout-num-gpus-per-engine") + 1] == str(p.rollout.engine_tp)
    assert "--tp-size" not in a and "--sglang-enable-dp-attention" in a
    assert a[a.index("--expert-model-parallel-size") + 1] == str(p.best.layout.ep)
    assert "swarmrl.integrations.slime.generate" in a and "swarmrl.integrations.slime.reward_post_process" in a
    cmd = slime.launch_command(a, p.env())
    assert cmd.startswith("ray job submit") and "CUDA_DEVICE_MAX_CONNECTIONS" in cmd


def test_slime_args_disaggregated_pool():
    p = plan(PRESETS["qwen3-8b"], ClusterSpec(nodes=1), TrainSpec(), rollout_gpus=8)
    a = slime.slime_args(p, stage(1), hf_checkpoint="h", load="l", save="s", prompt_data="d", rollout_batch_size=8,
                         num_rollout=1, external_engines=["10.0.0.5:8000", "10.0.0.6:8000"])  # fmt: skip
    assert "--colocate" not in a and a[a.index("--update-weight-transport") + 1] == "nccl"
    i = a.index("--rollout-external-engine-addrs")
    assert a[i + 1 : i + 3] == ["10.0.0.5:8000", "10.0.0.6:8000"]


def test_prompt_rows_filter_by_solve_rate(task):
    rows = slime.prompt_rows([task], {task.task_id: 0.5}, lo=0.1, hi=0.9)
    assert rows == [{"prompt": task.task_id, "label": task.task_id,
                     "metadata": {"task_id": task.task_id, "solve_rate": 0.5}}]  # fmt: skip
    assert slime.prompt_rows([task], {task.task_id: 1.0}, lo=0.1, hi=0.9) == []
    assert slime.task_id_of(Sample(prompt="x", metadata={"task_id": "y"})) == "y"


def test_routing_header_carries_the_session_key():
    stub = StubSGLang(lambda role, turn, text: "ok")
    try:
        client = SGLangClient(stub.url, routing_header="X-SMG-Routing-Key", retries=1)
        ids = stub.tok.encode("You are the Coder")
        asyncio.run(client.generate(ids, SamplingConfig(), [], "ep1:coder", 16))
        sent = {k.lower(): v for k, v in stub.headers[-1].items()}  # HTTP header names are case-insensitive
        assert sent.get("x-smg-routing-key") == "ep1:coder"
    finally:
        stub.close()


def test_hooks_resolve_like_slime_load_function():
    gen, post = slime.check_hooks_importable()
    assert asyncio.iscoroutinefunction(gen) and callable(post)


def test_cli_slime_data_and_args(task, tmp_path, capsys):
    import json

    from swarmrl.cli import main

    toml = task.task_dir / "task.toml"
    toml.write_text(toml.read_text().replace('split = "dev"', 'split = "train"'))
    out = tmp_path / "tasks.jsonl"
    assert main(["slime-data", str(task.task_dir.parent), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["prompt"] == task.task_id
    capsys.readouterr()
    assert main(["slime-args", "--model", "qwen3-30b-a3b", "--stage", "2", "--hf-checkpoint", "/m", "--load", "/d",
                 "--save", "/s", "--prompt-data", str(out), "--json"]) == 0  # fmt: skip
    got = json.loads(capsys.readouterr().out)
    assert "--use-tis" in got["args"] and got["env"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert main(["slime-args", "--model", "qwen3-8b", "--hf-checkpoint", "/m", "--load", "/d", "--save", "/s",
                 "--prompt-data", str(out)]) == 0  # fmt: skip
    assert capsys.readouterr().out.startswith("ray job submit")
