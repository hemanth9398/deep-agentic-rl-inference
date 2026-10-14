"""Per-role training for v2: PPO-clip for the planner and the coder, a calibrated verifier loss for the router.

Three models, three LoRA adapters, three optimizers; no gradient crosses between them (they talk in text).

  planner, coder:  L = -(1/N) sum_tokens m * w_IS * min(rho * A, clip(rho, 1-e_lo, 1+e_hi) * A)
                   (the v1 trainer's update, with A from credit.planner_targets / credit.coder_targets)
  router:          L = sum_i w_i BCE(sigmoid(z_i), label_i) / sum_i w_i + lambda * mean_pairs softplus(z_- - z_+)
                   z_i = log p(YES) - log p(NO) at the router's first output token (two-way softmax).

The router loss is computed one sequence at a time: dL/dz_i is known in closed form once every z is known,
  dL/dz_i = w_i (sigmoid(z_i) - label_i) / W
            + lambda / P * (sum_{(i,n)} -sigmoid(z_n - z_i) + sum_{(p,i)} sigmoid(z_i - z_p))
so a no-grad pass computes all z, then each sequence is re-run with grad and backpropagated with z_i * g_i.
That is the exact gradient of the batch loss with one sequence's activations in memory.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from swarmrl.rl.packing import PackedSample
from swarmrl.rl.trainer import TorchPolicyTrainer
from swarmrl.v2.credit import PolicyTarget, VerifierTarget, ranking_pairs


def to_samples(targets: Sequence[PolicyTarget]) -> tuple[list[PackedSample], list[np.ndarray]]:
    """Each call becomes one sequence: its prompt (no loss) + its output (loss, the call's advantage)."""
    samples, adv = [], []
    for t in targets:
        c = t.call
        ids = c.input_ids + c.output_ids + c.closing_ids
        n_in, n_out = len(c.input_ids), len(c.output_ids)
        mask = [0] * n_in + [1] * n_out + [0] * len(c.closing_ids)
        lps = [0.0] * n_in + list(c.rollout_logprobs) + [0.0] * len(c.closing_ids)
        samples.append(PackedSample("", "", None, c.role, n_in, ids, mask, lps, advantage=t.advantage))
        adv.append(np.asarray(mask, dtype=np.float64) * t.advantage)
    return samples, adv


class VerifierTrainer(TorchPolicyTrainer):
    """The router's trainer: the same LoRA machinery, plus the verdict-logit loss."""

    def verdict_logit(self, input_ids: Sequence[int], yes_id: int, no_id: int, grad: bool = False):
        torch = self.torch
        ids = torch.tensor(list(input_ids), device=self.device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = self.model(input_ids=ids[None]).logits[0, -1].float()  # predicts the first output token
            two = torch.log_softmax(logits[[yes_id, no_id]], dim=-1)
            return two[0] - two[1]

    def probabilities(self, targets: Sequence[VerifierTarget], yes_id: int, no_id: int) -> list[float]:
        self.model.eval()
        return [1 / (1 + math.exp(-float(self.verdict_logit(t.call.input_ids, yes_id, no_id)))) for t in targets]

    def verifier_backward(self, targets: Sequence[VerifierTarget], yes_id: int, no_id: int,
                          rank_weight: float = 0.5) -> dict:  # fmt: skip
        """Accumulate the exact gradient of the batch router loss into .grad, one sequence at a time."""
        self.model.eval()
        z = [float(self.verdict_logit(t.call.input_ids, yes_id, no_id)) for t in targets]
        sig = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
        wsum = sum(t.weight for t in targets)
        g = [t.weight * (sig(zi) - t.label) / wsum for t, zi in zip(targets, z, strict=True)]
        pairs = ranking_pairs(targets)
        for better, worse in pairs:
            s = sig(z[worse] - z[better]) * rank_weight / len(pairs)
            g[better] -= s
            g[worse] += s

        def nll(label: float, zi: float) -> float:
            return -(label * math.log(max(sig(zi), 1e-12)) + (1 - label) * math.log(max(1 - sig(zi), 1e-12)))

        bce = sum(t.weight * nll(t.label, zi) for t, zi in zip(targets, z, strict=True)) / wsum
        rank = (sum(math.log1p(math.exp(z[w] - z[b])) for b, w in pairs) / len(pairs)) if pairs else 0.0
        self.model.train()
        for t, gi in zip(targets, g, strict=True):
            if gi == 0.0:
                continue
            (self.verdict_logit(t.call.input_ids, yes_id, no_id, grad=True) * gi).backward()
        return {"loss": bce + rank_weight * rank, "bce": bce, "rank": rank, "items": float(len(targets)),
                "pairs": float(len(pairs))}  # fmt: skip

    def update_verifier(self, targets: Sequence[VerifierTarget], yes_id: int, no_id: int, rank_weight: float = 0.5,
                        lr_scale: float = 1.0) -> dict:  # fmt: skip
        torch = self.torch
        if not targets:
            return {"loss": 0.0, "stepped": 0.0, "items": 0.0}
        self.optimizer.zero_grad(set_to_none=True)
        stats = self.verifier_backward(targets, yes_id, no_id, rank_weight)
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm))
        lr = self.cfg.learning_rate * lr_scale
        stepped = lr > 0 and math.isfinite(grad_norm)
        if stepped:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {**stats, "grad_norm": grad_norm, "stepped": float(stepped)}
