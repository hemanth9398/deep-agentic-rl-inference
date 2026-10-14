"""A small simulation of the triad's learning dynamics, run through the real gate and credit code.

Plans are one of a few "ideas" with a true success probability (what the coder achieves with them). The planner
is a softmax over ideas, the router a per-idea logit; both start wrong: the router is confident the best idea is
bad. The question the design must answer: does the system recover the idea the router wrongly rejects?

  explore=True   gate with shadow exploration + IPS (the v2 design)
  explore=False  a hard router gate: only accepted plans ever run (what "trust the router" gives you)

Only the learners are tabular; decisions, credit and labels come from router.py and credit.py unchanged.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swarmrl.v2.credit import CreditConfig, decision_stats, planner_targets, router_targets
from swarmrl.v2.records import Call, PatchRecord, PlanRecord, Round, TriEpisode, Verdict
from swarmrl.v2.router import Gate, GateConfig, sigmoid


@dataclass
class SimConfig:
    true_value: tuple[float, ...] = (0.8, 0.35, 0.2, 0.05)  # idea 0 is the best
    router_init: tuple[float, ...] = (0.03, 0.5, 0.5, 0.5)  # router's initial belief: idea 0 looks bad
    steps: int = 150
    episodes_per_step: int = 8
    plans_per_round: int = 4
    patches_per_plan: int = 4
    planner_lr: float = 0.5
    router_lr: float = 1.0
    explore: bool = True
    seed: int = 0


@dataclass
class SimResult:
    best_prob: list[float] = field(default_factory=list)  # planner's probability of proposing the best idea
    router_p_best: list[float] = field(default_factory=list)
    resolve_rate: list[float] = field(default_factory=list)
    false_reject_rate: list[float] = field(default_factory=list)


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def _call(role: str, idea: int) -> Call:
    return Call(role, [idea], [idea], [0.0], text=str(idea))


def run(cfg: SimConfig = SimConfig()) -> SimResult:
    rng = random.Random(cfg.seed)
    n = len(cfg.true_value)
    theta = [0.0] * n
    phi = [math.log(p / (1 - p)) for p in cfg.router_init]
    gcfg = GateConfig() if cfg.explore else GateConfig(epsilon=0.0, temperature=0.01, min_temperature=0.01,
                                                         max_temperature=0.01)  # fmt: skip
    gate = Gate(gcfg, random.Random(cfg.seed + 1))
    out = SimResult()
    for step in range(cfg.steps):
        pi = _softmax(theta)
        episodes = []
        for g in range(cfg.episodes_per_step):
            ideas = rng.choices(range(n), weights=pi, k=cfg.plans_per_round)
            ps = [sigmoid(phi[i]) for i in ideas]
            dec = gate.decide(ps, force=True)
            plans = []
            for k, idea in enumerate(ideas):
                plan = PlanRecord(str(idea), _call("planner", idea), True)
                plan.verdict = Verdict(ps[k], "", True, _call("router", idea))
                if dec.executed[k]:
                    for _ in range(cfg.patches_per_plan):
                        ok = rng.random() < cfg.true_value[idea]
                        plan.patches.append(PatchRecord({}, _call("coder", idea), True, hidden_resolved=ok))
                plans.append(plan)
            ep = TriEpisode(f"{step}-{g}", "task", f"task#{step}", g, [Round(plans, dec)])
            k_main = dec.main[0]
            ep.reward = max(p.y for p in plans[k_main].patches)  # router ranks patches perfectly here
            episodes.append(ep)
        # planner: policy gradient on the softmax with the v2 planner advantages
        grad = [0.0] * n
        targets = planner_targets(episodes, CreditConfig(beta_team=0.5))
        for t in targets:
            i = int(t.call.text)
            for j in range(n):
                grad[j] += t.advantage * ((1.0 if j == i else 0.0) - pi[j])
        theta = [th + cfg.planner_lr * gr / max(len(targets), 1) for th, gr in zip(theta, grad, strict=True)]
        # router: weighted BCE gradient on its per-idea logit, with the v2 labels and IPS weights
        rgrad, wsum = [0.0] * n, 0.0
        for t in router_targets(episodes):
            i = int(t.call.text)
            rgrad[i] += t.weight * (t.label - sigmoid(phi[i]))
            wsum += t.weight
        phi = [ph + cfg.router_lr * gr / max(wsum, 1e-9) * n for ph, gr in zip(phi, rgrad, strict=True)]
        if cfg.explore:
            gate.adapt()
        stats = decision_stats(episodes)
        out.best_prob.append(_softmax(theta)[0])
        out.router_p_best.append(sigmoid(phi[0]))
        out.resolve_rate.append(sum(e.reward for e in episodes) / len(episodes))
        out.false_reject_rate.append(stats["router/false_reject_rate"])
    return out


def summary(res: SimResult, tail: int = 20) -> dict[str, float]:
    def avg(xs):
        return sum(xs[-tail:]) / min(tail, len(xs))

    return {"planner_p_best": avg(res.best_prob), "router_p_best": avg(res.router_p_best),
            "resolve_rate": avg(res.resolve_rate), "false_reject_rate": avg(res.false_reject_rate)}  # fmt: skip
