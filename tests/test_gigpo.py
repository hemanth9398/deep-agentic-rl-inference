"""GiGPO: two-level advantages (episode-level per-role GRPO + step-level anchor groups)."""

import numpy as np
import pytest

from swarmrl.rl.gigpo import GiGPOConfig, discounted_returns, explain, gigpo_token_advantages, step_advantages
from swarmrl.rl.packing import pack_episodes
from swarmrl.traces.schema import Episode, GradeSummary, Termination, Transition


def ep(eid, reward, turns, group="g"):
    """turns: list of (role, tool_name, observation, input_ids)."""
    e = Episode(eid, "t", "train", "fp", "ck", 0, Termination.SUBMIT, reward, GradeSummary(resolved=reward > 0))
    e.group_id = group
    e.transitions = [
        Transition(eid, role, i, 0, list(ids), [9, 9], [-0.5, -0.5], {"name": tool, "arguments": {}}, obs)
        for i, (role, tool, obs, ids) in enumerate(turns)
    ]
    return e


START = ("planner", "view", "file shown", [1, 2, 3])  # every episode opens with the same move


def test_discounted_returns():
    e = ep("a", 1.0, [START, ("coder", "submit", "", [4]), ("coder", "submit", "", [5])])
    r = discounted_returns(e, 0.5)
    assert r == {0: 0.25, 1: 0.5, 2: 1.0}


def test_step_level_follows_returns_within_an_anchor():
    # Same first state everywhere; after it, E1 and E2 make the same move, E3 diverges.
    e1 = ep("E1", 1.0, [START, ("coder", "edit", "ok", [7]), ("coder", "submit", "", [8])])
    e2 = ep("E2", 0.0, [START, ("coder", "edit", "ok", [7]), ("coder", "submit", "", [8])])
    e3 = ep("E3", 0.0, [START, ("coder", "bash", "boom", [7]), ("coder", "submit", "", [8])])
    adv, stats = step_advantages([e1, e2, e3], GiGPOConfig(gamma=0.95))
    # turn 0: all 3 episodes share the state; returns 0.95^2, 0, 0 -> +1.414, -0.707, -0.707
    assert adv[("E1", 0)] == pytest.approx(np.sqrt(2), abs=1e-4)
    assert adv[("E2", 0)] == pytest.approx(-np.sqrt(2) / 2, abs=1e-4)
    # turn 1: same env state for all (history = START only) -> anchored
    assert ("E3", 1) in adv
    # turn 2: E1/E2 share history; E3's history differs (bash/boom) -> E3 alone, no step advantage
    assert ("E1", 2) in adv and ("E2", 2) in adv and ("E3", 2) not in adv
    assert stats.anchor_groups == 3 and stats.total_turns == 9 and stats.anchored_turns == 8


def test_gigpo_has_signal_where_grpo_has_none():
    """All episodes succeed: GRPO gives every token 0. GiGPO prefers the shorter success."""
    fast = ep("fast", 1.0, [START, ("coder", "submit", "", [4])])
    slow = ep(
        "slow", 1.0, [START, ("coder", "view", "x", [4]), ("coder", "view", "y", [5]), ("coder", "submit", "", [6])]
    )
    samples = pack_episodes([fast, slow])
    advs, _ = gigpo_token_advantages(samples, [fast, slow], GiGPOConfig())
    by = {(s.episode_id, s.role): a for s, a in zip(samples, advs, strict=True)}
    assert by[("fast", "planner")].max() > 0 > by[("slow", "planner")].min()  # same first move, faster outcome
    from swarmrl.rl.advantages import grpo_advantages

    grpo, _ = grpo_advantages([fast, slow])
    assert set(grpo.values()) == {0.0}


def test_token_advantages_land_on_each_turns_own_tokens():
    e1 = ep("E1", 1.0, [START, ("coder", "edit", "ok", [7])])
    e2 = ep("E2", 0.0, [START, ("coder", "bash", "x", [7])])
    samples = pack_episodes([e1, e2])
    advs, _ = gigpo_token_advantages(samples, [e1, e2], GiGPOConfig(omega=0.5))
    for s, a in zip(samples, advs, strict=True):
        mask = np.asarray(s.loss_mask, dtype=bool)
        assert np.all(a[~mask] == 0.0)  # prompts and observations never carry advantage
        for _, lo, hi in s.spans:
            assert len(set(a[lo:hi])) == 1  # one value per turn


def test_context_key_is_stricter_than_env_key():
    # Same tool history, different reasoning tokens in the coder's input -> env anchors, context doesn't.
    e1 = ep("E1", 1.0, [START, ("coder", "edit", "ok", [70])])
    e2 = ep("E2", 0.0, [START, ("coder", "edit", "ok", [71])])
    env, _ = step_advantages([e1, e2], GiGPOConfig(state_key="env"))
    ctx, _ = step_advantages([e1, e2], GiGPOConfig(state_key="context"))
    assert ("E1", 1) in env and ("E1", 1) not in ctx
    assert ("E1", 0) in ctx  # identical first prompts still anchor


def test_groups_never_mix():
    a = ep("A", 1.0, [START], group="g1")
    b = ep("B", 0.0, [START], group="g2")
    adv, stats = step_advantages([a, b])
    assert adv == {} and stats.anchor_groups == 0


def test_explain_rows():
    e1 = ep("E1", 1.0, [START])
    e2 = ep("E2", 0.0, [START])
    rows = explain([e1, e2])
    assert {r["role"] for r in rows} == {"planner"} and all(r["step"] is not None for r in rows)


def test_rl_loop_with_gigpo(make_runner, task, factory):
    from dataclasses import replace

    from test_rl import FakeTrainer

    from swarmrl.inference.weight_sync import NullWeightSync
    from swarmrl.rl.curriculum import Curriculum
    from swarmrl.rl.loop import RLLoop
    from swarmrl.rl.schedule import stage
    from swarmrl.swarm.config import default_swarm_config
    from swarmrl.swarm.dispatcher import Dispatcher
    from swarmrl.tasks.oracle_agent import oracle_file_states, oracle_script

    cfg = default_swarm_config()
    st = replace(stage(1), group_size=3)
    runner = make_runner(oracle_script(oracle_file_states(task, factory), cfg), st.apply(cfg))
    rows = []
    loop = RLLoop(Dispatcher(runner), FakeTrainer(), NullWeightSync(), Curriculum({task.task_id: 0.5}), st,
                  {task.task_id: task}, batch_tasks=1, log=rows.append, advantage_estimator="gigpo")  # fmt: skip
    loop.run(1)
    m = rows[0]
    assert m["advantage_estimator"] == "gigpo"
    # The oracle agent plays identically every time, so every turn is a shared state across the 3 episodes.
    assert m["gigpo/anchored_fraction"] == 1.0 and m["gigpo/max_anchor_size"] == 3


def test_timing_noise_does_not_split_states():
    from swarmrl.rl.gigpo import normalize_observation

    a = "1 passed in 0.02s at 0x7f3a9c2b1d40, /tmp/swarmrl-ab12/workspace"
    b = "1 passed in 0.31s at 0x7f11aa00ff20, /tmp/swarmrl-zz99/workspace"
    assert normalize_observation(a) == normalize_observation(b)
    assert normalize_observation("1 passed") != normalize_observation("1 failed")  # real differences survive
