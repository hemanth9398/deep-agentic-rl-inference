"""Policy loss, off-policy correction and rollout/trainer mismatch diagnostics.

Reference implementations over numpy arrays (the same functions run on torch tensors: only
exp/clip/minimum/where/sum are used through a tiny shim). SkyRL-train's own loss is what
trains in production; tests pin that its config reproduces these numbers.

Shapes: [batch, seq]; `mask` is 1 on policy tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _xp(x):
    if hasattr(x, "detach"):  # torch
        import torch

        return torch, (lambda a, lo, hi: torch.clamp(a, lo, hi))
    return np, (lambda a, lo, hi: np.clip(a, lo, hi))


def masked_mean(x, mask):
    xp, _ = _xp(x)
    return (x * mask).sum() / xp.maximum(mask.sum(), xp.ones_like(mask.sum()))


def mismatch_kl(logp_trainer, logp_rollout, mask):
    """k3 estimate of KL(rollout || trainer) on rollout samples. At stage 0 this must be noise-level."""
    xp, _ = _xp(logp_trainer)
    log_ratio = logp_trainer - logp_rollout
    k3 = xp.exp(log_ratio) - 1 - log_ratio
    return masked_mean(k3, mask)


def alignment_shift(logp_trainer: np.ndarray, logp_rollout: np.ndarray, mask: np.ndarray, max_shift: int = 2) -> int:
    """Best shift between trainer and rollout log-probs. Anything but 0 = an off-by-one in the data path."""
    best, best_err = 0, float("inf")
    for s in sorted(range(-max_shift, max_shift + 1), key=abs):  # ties go to the smallest shift
        a = np.roll(logp_trainer, s, axis=-1)
        m = mask * np.roll(mask, s, axis=-1)
        if m.sum() == 0:
            continue
        err = float((np.abs(a - logp_rollout) * m).sum() / m.sum())
        if err < best_err - 1e-12:
            best, best_err = s, err
    return best


OffPolicyMode = Literal["none", "token_tis", "token_tis_reject"]


@dataclass(frozen=True)
class OffPolicyConfig:
    mode: OffPolicyMode = "none"
    tis_cap: float = 2.0  # truncated IS: w = min(ratio, cap)
    reject_low: float = 0.5  # token rejection: mask tokens with ratio outside [low, high]
    reject_high: float = 2.0


def importance_weights(logp_old, logp_rollout, mask, cfg: OffPolicyConfig):
    """ratio = pi_old_trainer / pi_rollout per token. Returns (weights, effective_mask)."""
    xp, clip = _xp(logp_old)
    if cfg.mode == "none":
        return xp.ones_like(logp_old), mask
    ratio = xp.exp(logp_old - logp_rollout)
    w = clip(ratio, 0.0, cfg.tis_cap)
    if cfg.mode == "token_tis_reject":
        keep = (ratio >= cfg.reject_low) & (ratio <= cfg.reject_high)
        mask = mask * keep
    return w, mask


@dataclass
class LossOut:
    loss: object
    clipfrac: float
    approx_kl: float
    policy_tokens: float


def clipped_pg_terms(logp_new, logp_old, advantages, clip_low=0.2, clip_high=0.28):
    """Per-token clipped surrogate objective (to maximize) and a clipped indicator."""
    xp, clip = _xp(logp_new)
    ratio = xp.exp(logp_new - logp_old)
    surr1 = ratio * advantages
    surr2 = clip(ratio, 1.0 - clip_low, 1.0 + clip_high) * advantages
    return xp.minimum(surr1, surr2), surr2 < surr1


def clipped_pg_loss(logp_new, logp_old, advantages, mask, clip_low=0.2, clip_high=0.28, is_weights=None) -> LossOut:
    """PPO-style clipped surrogate with asymmetric clip (clip-higher), token-mean reduction, no KL term.

    advantages: [batch, seq] (the per-(episode, role) scalar broadcast over that role's tokens).
    """
    obj, clipped = clipped_pg_terms(logp_new, logp_old, advantages, clip_low, clip_high)
    if is_weights is not None:
        obj = obj * is_weights
    loss = -masked_mean(obj, mask)
    n = float(mask.sum())
    clipfrac = float((clipped * mask).sum()) / max(n, 1.0)
    kl = float(masked_mean(logp_old - logp_new, mask))
    return LossOut(loss, clipfrac, kl, n)


def entropy_from_logprobs(logp_sampled, mask) -> float:
    """Cheap entropy proxy: mean negative log-prob of sampled tokens (tracks collapse)."""
    return float(masked_mean(-logp_sampled, mask))
