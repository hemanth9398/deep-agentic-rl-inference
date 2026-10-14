"""Credit assignment for the planner / coder / router triad: one loss per role, all tied to the final outcome.

Notation (one task, G episodes; in each round the planner proposes K plans; each executed plan gets M patches):
  y_kj   hidden-test outcome of patch j of plan k (every executed patch is graded, shadows included)
  V_k    = mean_j y_kj, the Monte-Carlo value of plan k (what the plan achieves with this coder)
  q_k    execution probability the gate drew plan k with (router.py); w_k = min(1/q_k, cap)
  R_e    final reward of episode e (the submitted patch's y); A_team(e) = GRPO of R over the G episodes

Planner (actor over plans; contextual bandit per round, off-policy because the router filters):
  b = sum_k w_k V_k / sum_k w_k,  s^2 = sum_k w_k (V_k - b)^2 / sum_k w_k       (executed siblings only)
  A_P(k) = (w_k / mean w) * (V_k - b) / (s + d)  +  beta * A_team(e) * [k is a main plan]
  A plan the router rejected but that ran as a shadow is scored by what it achieved, not by the router's
  opinion: a good idea the router wrongly rejected still pulls the planner toward it.

Coder (actor over patches, conditioned on the plan, so plan quality cancels out):
  A_C(k, j) = (y_kj - mean_j y_kj) / (std_j y_kj + d)  +  beta * A_team(e) * [k is a main plan]

Router (verifier / value: p_i = sigmoid(z_i), z_i = logit(YES) - logit(NO) at its first output token):
  L_R = sum_i w_i BCE(p_i, label_i) / sum_i w_i  +  lambda * mean_{(+,-) siblings} softplus(z_- - z_+)
  labels: V_k for plans (weight w_k: the gate's selection is corrected exactly like the planner's),
          y_kj for patches (weight 1; ranking pairs among a plan's patches with different y).
  Decisions are derived from the calibrated p and the cost matrix (router.py), so learning p is learning
  "how good would this be if accepted", including for ideas it rejected (their shadows provide the labels).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from swarmrl.v2.records import Call, TriEpisode


@dataclass(frozen=True)
class CreditConfig:
    beta_team: float = 0.5  # weight of the shared final-outcome advantage on main-path calls
    std_eps: float = 1e-6
    ips_cap: float = 10.0
    rank_weight: float = 0.5  # lambda for the router's pairwise ranking loss


@dataclass
class PolicyTarget:
    call: Call
    advantage: float
    kind: str  # "plan" | "patch"


@dataclass
class VerifierTarget:
    call: Call
    label: float
    weight: float
    group: tuple  # siblings that form ranking pairs
    kind: str  # "plan" | "patch"


def _norm(xs: Sequence[float], ws: Sequence[float] | None = None, d: float = 1e-6) -> list[float]:
    ws = list(ws) if ws is not None else [1.0] * len(xs)
    tot = sum(ws)
    if not xs or tot <= 0:
        return [0.0] * len(xs)
    b = sum(w * x for w, x in zip(ws, xs, strict=True)) / tot
    s = math.sqrt(sum(w * (x - b) ** 2 for w, x in zip(ws, xs, strict=True)) / tot)
    return [0.0 if s == 0 else (x - b) / (s + d) for x in xs]


def team_advantages(episodes: Sequence[TriEpisode], d: float = 1e-6) -> dict[str, float]:
    groups: dict[str, list[TriEpisode]] = defaultdict(list)
    for e in episodes:
        if e.scored:
            groups[e.group_id].append(e)
    out: dict[str, float] = {}
    for eps in groups.values():
        for e, a in zip(eps, _norm([e.reward for e in eps], d=d), strict=True):
            out[e.episode_id] = a
    return out


def planner_targets(episodes: Sequence[TriEpisode], cfg: CreditConfig = CreditConfig()) -> list[PolicyTarget]:
    team = team_advantages(episodes, cfg.std_eps)
    out: list[PolicyTarget] = []
    for e in episodes:
        if not e.scored:
            continue
        main = set(e.main_plans())
        for r, rd in enumerate(e.rounds):
            idx, vals, ws = [], [], []
            for k, plan in enumerate(rd.plans):
                if not plan.valid:  # unparseable plan: deterministically worthless, observed with certainty
                    idx.append(k), vals.append(0.0), ws.append(1.0)
                elif rd.gate.executed[k] and plan.value is not None:
                    idx.append(k), vals.append(plan.value), ws.append(rd.gate.ips_weight(k, cfg.ips_cap))
            if not idx:
                continue
            local = _norm(vals, ws, cfg.std_eps)
            mean_w = sum(ws) / len(ws)
            for k, a, w in zip(idx, local, ws, strict=True):
                adv = (w / mean_w) * a + (cfg.beta_team * team.get(e.episode_id, 0.0) if (r, k) in main else 0.0)
                out.append(PolicyTarget(rd.plans[k].call, adv, "plan"))
    return out


def coder_targets(episodes: Sequence[TriEpisode], cfg: CreditConfig = CreditConfig()) -> list[PolicyTarget]:
    team = team_advantages(episodes, cfg.std_eps)
    out: list[PolicyTarget] = []
    for e in episodes:
        if not e.scored:
            continue
        main = set(e.main_plans())
        for r, rd in enumerate(e.rounds):
            for k, plan in enumerate(rd.plans):
                graded = [p for p in plan.patches if p.hidden_resolved is not None]
                if not graded:
                    continue
                local = _norm([p.y for p in graded], d=cfg.std_eps)
                bonus = cfg.beta_team * team.get(e.episode_id, 0.0) if (r, k) in main else 0.0
                out += [PolicyTarget(p.call, a + bonus, "patch") for p, a in zip(graded, local, strict=True)]
    return out


def router_targets(episodes: Sequence[TriEpisode], cfg: CreditConfig = CreditConfig()) -> list[VerifierTarget]:
    out: list[VerifierTarget] = []
    for e in episodes:
        if not e.scored:
            continue
        for r, rd in enumerate(e.rounds):
            for k, plan in enumerate(rd.plans):
                v = plan.verdict
                if v and v.call and v.valid and rd.gate.executed[k] and plan.value is not None:
                    w = rd.gate.ips_weight(k, cfg.ips_cap)
                    out.append(VerifierTarget(v.call, plan.value, w, (e.episode_id, r, "plans"), "plan"))
                for p in plan.patches:
                    pv = p.verdict
                    if pv and pv.call and pv.valid and p.hidden_resolved is not None:
                        out.append(VerifierTarget(pv.call, p.y, 1.0, (e.episode_id, r, k), "patch"))
    return out


def ranking_pairs(targets: Sequence[VerifierTarget]) -> list[tuple[int, int]]:
    """(better, worse) index pairs among patch siblings with different outcomes."""
    by_group: dict[tuple, list[int]] = defaultdict(list)
    for i, t in enumerate(targets):
        if t.kind == "patch":
            by_group[t.group].append(i)
    pairs = []
    for idx in by_group.values():
        pairs += [(a, b) for a in idx for b in idx if targets[a].label > targets[b].label]
    return pairs


def decision_stats(episodes: Sequence[TriEpisode]) -> dict[str, float]:
    """How good the router's decisions were, measured on plans whose real value is known."""
    fa = fr = ta = tr = 0
    brier: list[float] = []
    for e in episodes:
        for rd in e.rounds:
            for k, plan in enumerate(rd.plans):
                if not (rd.gate.executed[k] and plan.value is not None and plan.verdict):
                    continue
                good, acc = plan.value > 0, rd.gate.accepted[k]
                ta += acc and good
                fa += acc and not good
                fr += (not acc) and good
                tr += (not acc) and not good
                brier.append((plan.verdict.p - plan.value) ** 2)
    n = fa + fr + ta + tr
    return {
        "router/false_accept_rate": fa / max(fa + tr, 1),  # among bad plans
        "router/false_reject_rate": fr / max(fr + ta, 1),  # among good plans: ideas the router would lose
        "router/brier": sum(brier) / len(brier) if brier else 0.0,
        "router/judged_plans": float(n),
    }
