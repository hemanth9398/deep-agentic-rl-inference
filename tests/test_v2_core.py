"""v2 decision rule and credit math, plus the learning-dynamics simulation."""

import random

import pytest

from swarmrl.v2.credit import (
    CreditConfig,
    coder_targets,
    decision_stats,
    planner_targets,
    ranking_pairs,
    router_targets,
    team_advantages,
)
from swarmrl.v2.records import Call, PatchRecord, PlanRecord, Round, TriEpisode, Verdict
from swarmrl.v2.router import Gate, GateConfig, GateDecision, binary_entropy
from swarmrl.v2.sim import SimConfig, run, summary


def call(role, tag):
    return Call(role, [1], [2], [-0.1], text=tag)


def plan_with(value_patches, p, tag, valid=True):
    plan = PlanRecord(tag, call("planner", tag), valid)
    plan.verdict = Verdict(p, "", True, call("router", tag))
    for j, ok in enumerate(value_patches):
        patch = PatchRecord({}, call("coder", f"{tag}/{j}"), True, hidden_resolved=ok)
        patch.verdict = Verdict(0.5, "", True, call("router", f"{tag}/{j}/v"))
        plan.patches.append(patch)
    return plan


def test_cost_optimal_threshold_and_execution_probabilities():
    cfg = GateConfig(cost_false_accept=1, cost_false_reject=3, epsilon=0.1, temperature=1.0)
    assert cfg.threshold == pytest.approx(0.25)
    q = Gate(cfg).execution_probs([0.25, 0.01, 0.99])
    assert q[0] == pytest.approx(0.1 + 0.9 * 0.5)  # exactly at the threshold: a coin flip above the floor
    assert 0.1 <= q[1] < 0.15 and q[2] > 0.99  # rejected ideas keep >= eps; confident accepts ~always run
    assert binary_entropy(0.5) == pytest.approx(1.0) and binary_entropy(1.0) == 0.0


def test_gate_main_plan_shadows_and_forcing():
    gate = Gate(GateConfig(epsilon=0.0, temperature=0.01, max_main=1), random.Random(0))
    d = gate.decide([0.9, 0.6, 0.05])
    assert d.executed == [True, True, False] and d.main == [0] and d.shadows == [1]
    forced = gate.decide([0.01, 0.02], force=True)
    assert forced.forced and forced.main == [1] and forced.q[1] == 1.0
    assert GateDecision([0.5], [0.05], [True], [True], [0]).ips_weight(0, cap=10) == 10  # capped at 1/q <= 10


def test_temperature_adapts_toward_target_entropy():
    gate = Gate(GateConfig(epsilon=0.0, target_entropy=0.8, temperature=1.0))
    for _ in range(5):
        gate.decide([0.999, 0.001])  # very confident -> low gate entropy
    assert gate.adapt() > 1.0  # raises T to explore more


def _episode(eid, rounds, final, reward, group="t#0"):
    ep = TriEpisode(eid, "t", group, 0, rounds, final, reward)
    return ep


def test_planner_credit_uses_real_outcomes_with_ips():
    # router accepted plan 0 (bad) and rejected plan 1 (good) which still ran as a shadow with q = 0.25
    plans = [plan_with([False, False], 0.9, "a"), plan_with([True, True], 0.1, "b"), plan_with([], 0.1, "c")]
    dec = GateDecision([0.9, 0.1, 0.1], [1.0, 0.25, 0.25], [True, False, False], [True, True, False], [0])
    ep = _episode("e", [Round(plans, dec)], (0, 0, 0), 0.0)
    t = {x.call.text: x.advantage for x in planner_targets([ep], CreditConfig(beta_team=0.0))}
    assert set(t) == {"a", "b"}  # the unexecuted plan has no observed outcome: no credit either way
    # IPS weights 1 and 4: baseline b = (1*0 + 4*1)/5 = 0.8, std = 0.4; normalized weights 0.4 and 1.6
    assert t["a"] == pytest.approx(0.4 * (0 - 0.8) / 0.4, rel=1e-4)
    assert t["b"] == pytest.approx(1.6 * (1 - 0.8) / 0.4, rel=1e-4)
    assert t["b"] > 0 > t["a"]  # the rejected-but-good idea is reinforced, the accepted-but-bad one is not


def test_invalid_plans_count_as_observed_failures():
    plans = [plan_with([True], 0.9, "ok"), plan_with([], 0.5, "bad", valid=False)]
    dec = GateDecision([0.9, 0.5], [1.0, 0.0], [True, True], [True, False], [0])
    t = {x.call.text: x.advantage for x in planner_targets([_episode("e", [Round(plans, dec)], (0, 0, 0), 1.0)],
                                                           CreditConfig(beta_team=0.0))}  # fmt: skip
    assert t["bad"] < 0 < t["ok"]


def test_coder_credit_is_plan_conditioned_and_team_bonus_on_main_path():
    main = plan_with([True, False, False, False], 0.9, "m")
    shadow = plan_with([True, True, False, False], 0.1, "s")
    dec = GateDecision([0.9, 0.1], [1.0, 0.5], [True, False], [True, True], [0])
    e1 = _episode("e1", [Round([main, shadow], dec)], (0, 0, 0), 1.0)
    e2 = _episode("e2", [], None, 0.0)  # same group, failed: A_team(e1) = +1
    assert team_advantages([e1, e2]) == {"e1": pytest.approx(1.0, rel=1e-4), "e2": pytest.approx(-1.0, rel=1e-4)}
    adv = {t.call.text: t.advantage for t in coder_targets([e1, e2], CreditConfig(beta_team=0.5))}
    sd = (0.25 * 0.75) ** 0.5
    assert adv["m/0"] == pytest.approx(0.75 / sd + 0.5, rel=1e-4)  # within-plan GRPO + team bonus
    assert adv["s/0"] == pytest.approx(1.0, rel=1e-4)  # shadow: local only (it did not shape the answer)


def test_router_labels_weights_ranking_and_decision_stats():
    plans = [plan_with([False, False], 0.9, "a"), plan_with([True, False], 0.1, "b")]
    dec = GateDecision([0.9, 0.1], [1.0, 0.2], [True, False], [True, True], [0])
    ep = _episode("e", [Round(plans, dec)], (0, 0, 0), 0.0)
    ts = router_targets([ep])
    plan_t = {t.call.text: (t.label, t.weight) for t in ts if t.kind == "plan"}
    assert plan_t == {"a": (0.0, 1.0), "b": (0.5, 5.0)}  # value labels, IPS-weighted like the planner
    pairs = ranking_pairs(ts)
    assert [(ts[a].call.text, ts[b].call.text) for a, b in pairs] == [("b/0/v", "b/1/v")]
    s = decision_stats([ep])
    assert s["router/false_reject_rate"] == 1.0 and s["router/false_accept_rate"] == 1.0
    assert s["router/brier"] == pytest.approx(((0.9 - 0) ** 2 + (0.1 - 0.5) ** 2) / 2)


def test_simulation_recovers_the_idea_a_biased_router_rejects():
    """The design claim, end to end through the real gate and credit code."""
    v2 = summary(run(SimConfig(explore=True, steps=120)))
    hard = summary(run(SimConfig(explore=False, steps=120)))
    assert v2["planner_p_best"] > 0.9 and hard["planner_p_best"] < 0.3
    assert abs(v2["router_p_best"] - 0.8) < 0.15 and hard["router_p_best"] < 0.15  # only v2 gets calibrated
    assert v2["resolve_rate"] > hard["resolve_rate"]


def test_cli_v2_sim(capsys):
    import json

    from swarmrl.cli import main

    assert main(["v2-sim", "--steps", "60", "--seeds", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["v2 (shadows + IPS)"]["planner_p_best"] > out["hard router gate"]["planner_p_best"]
