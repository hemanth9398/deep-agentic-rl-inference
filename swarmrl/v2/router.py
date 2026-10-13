"""The router's decision rule: which plans go to the coder, which are explored anyway, with known probabilities.

The router model outputs p = P(the final state is solved | context, candidate), a calibrated estimate trained on
real outcomes (see credit.py). Turning p into a decision is then a cost question, not a learning question:

  accepting a plan that fails costs c_FA (coder + sandbox compute spent for nothing)
  rejecting a plan that would succeed costs c_FR (a good idea lost)
  => accept iff  p * c_FR >= (1 - p) * c_FA   <=>   p >= tau = c_FA / (c_FA + c_FR)

With c_FR > c_FA (default 3:1, tau = 0.25) the router leans toward not losing ideas.

Execution is randomized so that every plan keeps a known chance of being tried:

  q_k = eps + (1 - eps) * sigmoid((logit p_k - logit tau) / T)

Each plan runs with probability q_k, independently. Plans the router rejected but that run anyway are
"shadows": their real outcome is what tells the planner and the router what a rejected idea was worth.
Because q_k is known, the planner's update reweights each executed plan by 1/q_k (inverse propensity), which
keeps its gradient unbiased however biased the router is. T adapts so that the mean entropy of the gate stays
near a target: the gate never collapses to always/never, and the planner keeps receiving signal on alternatives.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def binary_entropy(p: float) -> float:
    """In bits; 1.0 at p = 0.5, 0 at p in {0, 1}."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@dataclass
class GateConfig:
    cost_false_accept: float = 1.0
    cost_false_reject: float = 3.0
    epsilon: float = 0.1  # floor on every plan's execution probability (exploration of rejected ideas)
    temperature: float = 1.0
    target_entropy: float = 0.5  # bits, mean over the gate's execution probabilities
    temperature_lr: float = 0.5
    min_temperature: float = 0.1
    max_temperature: float = 10.0
    max_main: int = 1  # plans whose patches compete for the final answer
    ips_cap: float = 10.0  # cap on 1/q

    @property
    def threshold(self) -> float:
        return self.cost_false_accept / (self.cost_false_accept + self.cost_false_reject)


@dataclass
class GateDecision:
    p: list[float]
    q: list[float]  # execution probabilities (propensities)
    accepted: list[bool]  # p >= tau: the router's own verdict
    executed: list[bool]
    main: list[int]  # executed plans whose patches compete for the final answer
    forced: bool = False  # nothing was drawn and the best plan was executed with probability 1

    @property
    def shadows(self) -> list[int]:
        return [k for k, e in enumerate(self.executed) if e and k not in self.main]

    def ips_weight(self, k: int, cap: float) -> float:
        return min(1.0 / max(self.q[k], 1e-9), cap)


@dataclass
class Gate:
    cfg: GateConfig = field(default_factory=GateConfig)
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    entropy_history: list[float] = field(default_factory=list)

    def execution_probs(self, ps: Sequence[float]) -> list[float]:
        c, tau = self.cfg, self.cfg.threshold
        return [c.epsilon + (1 - c.epsilon) * sigmoid((logit(p) - logit(tau)) / c.temperature) for p in ps]

    def decide(self, ps: Sequence[float], force: bool = False) -> GateDecision:
        qs = self.execution_probs(ps)
        executed = [self.rng.random() < q for q in qs]
        forced = False
        if force and not any(executed) and ps:
            best = max(range(len(ps)), key=lambda k: ps[k])
            executed[best], qs[best], forced = True, 1.0, True
        accepted = [p >= self.cfg.threshold for p in ps]
        pool = [k for k in range(len(ps)) if executed[k] and accepted[k]] or [k for k in range(len(ps)) if executed[k]]
        main = sorted(pool, key=lambda k: -ps[k])[: self.cfg.max_main]
        self.entropy_history.append(sum(binary_entropy(q) for q in qs) / max(len(qs), 1))
        return GateDecision(list(ps), qs, accepted, executed, main, forced)

    def adapt(self) -> float:
        """After each training step: move T so the gate's mean entropy tracks the target."""
        if not self.entropy_history:
            return self.cfg.temperature
        h = sum(self.entropy_history) / len(self.entropy_history)
        self.entropy_history.clear()
        c = self.cfg
        c.temperature = min(max(c.temperature * math.exp(c.temperature_lr * (c.target_entropy - h)),
                                c.min_temperature), c.max_temperature)  # fmt: skip
        return c.temperature
