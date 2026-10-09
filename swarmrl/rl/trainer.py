"""Trainers for the RL step.

`Trainer` is the contract RLLoop drives. `TorchPolicyTrainer` is a complete single-process
implementation (torch + PEFT LoRA) for the loop in swarmrl.rl.loop: scoring, the clipped
policy-gradient update and checkpointing. For multi-GPU FSDP training, SkyRL-train drives the
same step through swarmrl.integrations.skyrl instead.
"""

from __future__ import annotations

import abc
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from swarmrl.rl.loss import clipped_pg_terms
from swarmrl.rl.packing import PackedSample
from swarmrl.rl.schedule import StageConfig

log = logging.getLogger(__name__)


class Trainer(abc.ABC):
    @abc.abstractmethod
    def logprobs(self, samples: Sequence[PackedSample]) -> list[np.ndarray]:
        """Per sample, log p(input_ids[t] | input_ids[:t]) at every position t (0.0 at t = 0),
        aligned index for index with PackedSample.rollout_logprobs."""

    @abc.abstractmethod
    def update(
        self,
        samples: Sequence[PackedSample],
        advantages: list[np.ndarray],
        old_logprobs: list[np.ndarray],
        is_weights: list[np.ndarray],
        masks: list[np.ndarray],
        stage: StageConfig,
    ) -> dict:
        """One optimizer step on the clipped surrogate over all policy tokens (token-mean). Returns metrics."""

    @abc.abstractmethod
    def save(self, version: int) -> str:
        """Write the current policy for the inference servers; returns the path to load."""


@dataclass
class TorchTrainerConfig:
    model_name_or_path: str
    output_dir: str
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    target_modules: str | list[str] = "all-linear"
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logprob_chunk: int = 1024  # positions per log-softmax chunk: bounds memory to chunk x vocab
    gradient_checkpointing: bool = True
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    save_mode: Literal["adapter", "merged"] = "adapter"  # must match DiskWeightSync.mode
    keep_checkpoints: int = 3


class TorchPolicyTrainer(Trainer):
    def __init__(self, cfg: TorchTrainerConfig, model=None, tokenizer=None, device: str | None = None):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model

        self.torch = torch
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, dtype=dtype)
            if tokenizer is None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
                except (OSError, ValueError):
                    log.warning("no tokenizer at %s: checkpoints will not include one", cfg.model_name_or_path)
        if not isinstance(model, PeftModel):
            model = get_peft_model(
                model,
                LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                           target_modules=cfg.target_modules, task_type="CAUSAL_LM"),
            )  # fmt: skip
        if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        # Policy-gradient ratios compare scoring (eval mode) with the update pass (train mode, needed by gradient
        # checkpointing): any active dropout would make the ratio != 1 before the first step and bias every update.
        _disable_dropout(model)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("no trainable parameters: LoRA did not attach to any module")
        self.optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self._saved: list[str] = []
        os.makedirs(cfg.output_dir, exist_ok=True)

    # --- scoring ------------------------------------------------------------------------

    def _token_logprobs(self, ids):
        """log p(ids[t] | ids[:t]) for t >= 1, chunked so the full [seq, vocab] log-softmax never materializes."""
        torch = self.torch
        hidden = self.model(input_ids=ids[None], output_hidden_states=False).logits[0]  # [T, V]
        out = []
        for s in range(0, hidden.shape[0] - 1, self.cfg.logprob_chunk):
            logits = hidden[s : s + self.cfg.logprob_chunk].float()
            targets = ids[s + 1 : s + 1 + logits.shape[0]]
            logits = logits[: targets.shape[0]]
            out.append(logits.gather(-1, targets[:, None])[:, 0] - torch.logsumexp(logits, dim=-1))
        lp = torch.cat(out) if out else hidden.new_zeros(0, dtype=torch.float32)
        return torch.cat([lp.new_zeros(1), lp])  # position 0 has no prediction

    def logprobs(self, samples: Sequence[PackedSample]) -> list[np.ndarray]:
        torch = self.torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for s in samples:
                ids = torch.tensor(s.input_ids, device=self.device)
                out.append(self._token_logprobs(ids).double().cpu().numpy())
        return out

    # --- update ---------------------------------------------------------------------------

    def update(self, samples, advantages, old_logprobs, is_weights, masks, stage: StageConfig) -> dict:
        torch = self.torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        n_tokens = float(sum(m.sum() for m in masks))
        if n_tokens == 0:
            return {"loss": 0.0, "clipfrac": 0.0, "grad_norm": 0.0, "policy_tokens": 0.0, "stepped": 0.0}
        total_loss = clipped = approx_kl = 0.0
        for s, adv, old, w, m in zip(samples, advantages, old_logprobs, is_weights, masks, strict=True):
            if m.sum() == 0:
                continue
            ids = torch.tensor(s.input_ids, device=self.device)
            t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)  # noqa: E731
            m_t, adv_t, old_t, w_t = t(m), t(adv), t(old), t(w)
            new = self._token_logprobs(ids)
            obj, clip_ind = clipped_pg_terms(new, old_t, adv_t, stage.clip_low, stage.clip_high)
            loss = -(obj * w_t * m_t).sum() / n_tokens  # token-mean over the whole batch
            loss.backward()
            total_loss += float(loss.detach())
            clipped += float((clip_ind.float() * m_t).sum())
            approx_kl += float(((old_t - new.detach()) * m_t).sum())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.cfg.max_grad_norm))  # fmt: skip
        lr = self.cfg.learning_rate * stage.lr_scale
        stepped = lr > 0 and math.isfinite(grad_norm)
        if stepped:
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            self.optimizer.step()
        elif not math.isfinite(grad_norm):
            log.warning("non-finite grad norm %s: step skipped", grad_norm)
        self.optimizer.zero_grad(set_to_none=True)
        return {
            "loss": total_loss,
            "clipfrac": clipped / n_tokens,
            "approx_kl": approx_kl / n_tokens,
            "grad_norm": grad_norm,
            "lr": lr,
            "policy_tokens": n_tokens,
            "stepped": float(stepped),
        }

    # --- checkpoints ------------------------------------------------------------------------

    def save(self, version: int) -> str:
        path = os.path.join(self.cfg.output_dir, f"step-{version:06d}")
        if self.cfg.save_mode == "adapter":
            self.model.save_pretrained(path)
        else:  # full merged weights for /update_weights_from_disk
            self.model.merge_adapter()
            try:
                self.model.get_base_model().save_pretrained(path, safe_serialization=True)
            finally:
                self.model.unmerge_adapter()
            # merged checkpoints must not carry the LoRA modules' parameter names
            _strip_lora_keys(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        self._saved.append(path)
        self._prune()
        return path

    def _prune(self) -> None:
        import shutil

        while len(self._saved) > self.cfg.keep_checkpoints:
            shutil.rmtree(self._saved.pop(0), ignore_errors=True)


_DROPOUT_ATTRS = ("attention_dropout", "attn_dropout", "hidden_dropout", "dropout", "resid_pdrop", "attn_pdrop",
                  "embd_pdrop", "dropout_rate", "classifier_dropout")  # fmt: skip


def _disable_dropout(model) -> None:
    """Zero every dropout: nn.Dropout modules, float attributes read by functional dropout, and the configs."""
    import torch

    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        for name in _DROPOUT_ATTRS:
            if isinstance(getattr(mod, name, None), float):
                setattr(mod, name, 0.0)
        cfg = getattr(mod, "config", None)
        for name in _DROPOUT_ATTRS:
            if isinstance(getattr(cfg, name, None), float):
                setattr(cfg, name, 0.0)


def _strip_lora_keys(path: str) -> None:
    """get_base_model() still holds LoRA wrapper modules; rewrite the safetensors with plain base names."""
    import glob

    from safetensors.torch import load_file, save_file

    for f in glob.glob(os.path.join(path, "*.safetensors")):
        sd = load_file(f)
        clean = {}
        for k, v in sd.items():
            if "lora_" in k:
                continue
            clean[k.replace(".base_layer.", ".")] = v
        save_file(clean, f, metadata={"format": "pt"})
