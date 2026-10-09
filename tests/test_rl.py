import asyncio

import numpy as np
import pytest

from conftest import call

from swarmrl.integrations.skyrl import episode_rows, skyrl_overrides
from swarmrl.rl.advantages import grpo_advantages, informative_probability
from swarmrl.rl.curriculum import Curriculum, stage_pool
from swarmrl.rl.loop import RLLoop, stage0_verdict
from swarmrl.rl.loss import (OffPolicyConfig, alignment_shift, clipped_pg_loss, importance_weights,
                             mismatch_kl)
from swarmrl.rl.packing import pack_episode
from swarmrl.rl.schedule import STAGES, stage
from swarmrl.inference.weight_sync import NullWeightSync
from swarmrl.swarm.config import default_swarm_config
from swarmrl.swarm.dispatcher import Dispatcher
from swarmrl.tasks.oracle_agent import oracle_file_states, oracle_script
from swarmrl.traces.schema import Episode, GradeSummary, Termination


@pytest.fixture
def oracle_ep(make_runner, task, factory):
    cfg = default_swarm_config()
    return asyncio.run(make_runner(oracle_script(oracle_file_states(task, factory), cfg), cfg).run(task))


def test_packing_rebuilds_each_role_context(oracle_ep):
    samples = pack_episode(oracle_ep)
    assert sorted(s.role for s in samples) == ["coder", "planner", "reviewer", "tester"]
    for s in samples:
        ts = [t for t in oracle_ep.transitions if t.role == s.role]
        last = ts[-1]
        assert s.input_ids == last.input_ids + last.output_ids + last.closing_ids
        assert s.num_policy_tokens() == sum(len(t.output_ids) for t in ts)
        pol = [lp for lp, m in zip(s.rollout_logprobs, s.loss_mask) if m]
        assert pol == [lp for t in ts for lp in t.rollout_logprobs]
        assert s.prompt_len == len(ts[0].input_ids)


def _ep(eid, group, reward, roles):
    from swarmrl.traces.schema import Transition

    e = Episode(eid, "t", "train", "fp", "ck", 0, Termination.SUBMIT, reward, GradeSummary(resolved=reward > 0))
    e.group_id = group
    e.transitions = [Transition(eid, r, i, 0, [1], [2], [-0.5]) for i, r in enumerate(roles)]
    return e


def test_grpo_per_role_normalization():
    eps = [
        _ep("a", "g", 1.0, ["planner", "coder", "tester"]),
        _ep("b", "g", 0.0, ["planner", "coder"]),
        _ep("c", "g", 0.0, ["planner", "coder", "tester"]),
        _ep("d", "g", 1.0, ["planner", "coder", "tester"]),
    ]
    adv, st = grpo_advantages(eps)
    assert st.informative_groups == 1
    assert adv[("a", "planner")] == pytest.approx(1.0, abs=1e-4) and adv[("b", "planner")] == pytest.approx(-1.0, abs=1e-4)
    # tester acted in a, c, d only: mean 2/3
    assert adv[("a", "tester")] > 0 > adv[("c", "tester")]
    assert ("b", "tester") not in adv
    flat, st2 = grpo_advantages([_ep("x", "h", 1.0, ["coder"]), _ep("y", "h", 1.0, ["coder"])])
    assert set(flat.values()) == {0.0} and st2.all_one == 1


def test_informative_probability_matches_doc():
    assert informative_probability(0.02, 8) == pytest.approx(0.15, abs=0.005)
    assert informative_probability(0.5, 8) == pytest.approx(0.99, abs=0.005)


def test_loss_and_corrections():
    mask = np.array([1, 1, 1, 0.0])
    lp = np.log(np.array([0.5, 0.5, 0.5, 0.5]))
    out = clipped_pg_loss(lp, lp, np.ones(4), mask)
    assert out.loss == pytest.approx(-1.0) and out.clipfrac == 0.0
    big = clipped_pg_loss(lp + np.log(2.0), lp, np.ones(4), mask, clip_high=0.28)
    assert big.loss == pytest.approx(-1.28) and big.clipfrac == 1.0
    small = clipped_pg_loss(lp + np.log(0.5), lp, -np.ones(4), mask, clip_low=0.2)
    assert small.loss == pytest.approx(0.8)
    assert float(mismatch_kl(lp, lp, mask)) == pytest.approx(0.0)
    w, m = importance_weights(lp + np.log(3.0), lp, mask, OffPolicyConfig("token_tis", tis_cap=2.0))
    assert np.allclose(w, 2.0) and (m == mask).all()
    _, m2 = importance_weights(lp + np.log(3.0), lp, mask, OffPolicyConfig("token_tis_reject"))
    assert m2.sum() == 0


def test_alignment_shift_detects_off_by_one():
    rng = np.random.default_rng(0)
    roll = rng.normal(-1, 0.5, 64)
    mask = np.ones(64)
    assert alignment_shift(roll.copy(), roll, mask) == 0
    assert alignment_shift(np.roll(roll, -1), roll, mask) == 1


def test_curriculum_sampling():
    cur = Curriculum({"easy": 1.0, "hard": 0.0, "mid": 0.5, "low": 0.1}, explore=0.0, seed=0)
    picks = cur.sample(2)
    assert set(picks) == {"mid", "low"}
    assert len(cur.sample(10)) == 4
    assert stage_pool({"a": 0.1, "b": 0.5, "c": 0.9}, 0.125, 0.875) == ["b"]
    assert stage_pool({"b": 0.5}, 0.125, 0.875, max_turns=10, min_turns={"b": 12}) == []


class FakeTrainer:
    """Scores with the rollout log-probs (+ optional drift): what a correct data path produces."""

    def __init__(self, drift=0.0, shift=0):
        self.drift, self.shift, self.updates = drift, shift, []

    def logprobs(self, samples):
        return [np.roll(np.asarray(s.rollout_logprobs) + self.drift, self.shift) for s in samples]

    def update(self, samples, advantages, old, weights, masks, stage):
        self.updates.append((len(samples), stage.name))
        return {"loss": 0.0}

    def save(self, version):
        return f"/ckpt/v{version}"


def test_rl_loop_stage0(make_runner, task, factory):
    cfg = default_swarm_config()
    s0 = stage(0)
    files = oracle_file_states(task, factory)
    good = oracle_script(files, cfg)

    runner = make_runner(good, s0.apply(cfg))
    sync = NullWeightSync()
    loop = RLLoop(Dispatcher(runner), FakeTrainer(), sync, Curriculum({task.task_id: 0.5}), s0,
                  {task.task_id: task}, batch_tasks=1, log=lambda m: None)
    hist = loop.run()
    assert len(hist) == 2 and loop.version == 2
    assert sync.calls == [("/ckpt/v1", 1)]  # step 2 syncs the step-1 weights first
    assert hist[0]["episodes"] == s0.group_size and hist[0]["policy_tokens"] > 0
    assert stage0_verdict(hist).passed
    bad = RLLoop(Dispatcher(runner), FakeTrainer(shift=1), NullWeightSync(), Curriculum({task.task_id: 0.5}), s0,
                 {task.task_id: task}, batch_tasks=1, log=lambda m: None).run(1)
    v = stage0_verdict(bad)
    assert not v.passed and "alignment shift" in v.reasons[0]


def test_skyrl_rows_and_overrides(oracle_ep):
    rows = episode_rows(oracle_ep, repetition=3)
    assert {r["instance_id"] for r in rows} == {f"{oracle_ep.task_id}::{r}" for r in ("planner", "coder", "tester", "reviewer")}
    assert all(r["is_last_step"] for r in rows)  # one packed sample per role here
    for r in rows:
        assert len(r["response_ids"]) == len(r["loss_mask"]) == len(r["rollout_logprobs"])
        assert r["reward"] == 1.0
    infra = Episode("i", "t", "train", "fp", "ck", 0)
    assert episode_rows(infra, 0)[0]["loss_mask"] == [0]
    o3 = skyrl_overrides(STAGES[3])
    assert o3["generator.step_wise_trajectories"] is True
    assert o3["trainer.algorithm.off_policy_correction.tis_ratio_type"] == "token"
    assert o3["trainer.algorithm.eps_clip_high"] == 0.28 and o3["trainer.algorithm.use_kl_loss"] is False
    assert skyrl_overrides(STAGES[0])["trainer.policy.optimizer_config.lr"] == 0.0
