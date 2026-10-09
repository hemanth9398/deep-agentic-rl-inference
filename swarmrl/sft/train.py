"""QLoRA SFT and bf16 merge. Heavy dependencies are imported lazily (pip install .[sft]).

QLoRA: 4-bit NF4 base, LoRA rank 32 on all linear layers, lr 1e-4, cosine, 1-3 epochs, no packing.
Merge the adapter in bf16 from the ORIGINAL base weights (never from the 4-bit quantized model).
Gate afterwards: eval path A on heldout; format errors must drop sharply and resolve rate must
beat the base model (swarmrl.lifecycle.gates).
"""

from __future__ import annotations

from dataclasses import dataclass

from swarmrl.sft.samples import IGNORE, read_jsonl


@dataclass
class QLoRAConfig:
    base_model: str
    output_dir: str
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    epochs: float = 2.0
    per_device_batch_size: int = 1
    grad_accum: int = 16
    max_len: int = 32_768
    gradient_checkpointing: bool = True
    quantize: bool = True  # 4-bit NF4 base (QLoRA); False = plain LoRA on a bf16/fp32 base
    seed: int = 0
    logging_steps: int = 10


def _collate(batch, pad_id: int):
    import torch

    n = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, x in enumerate(batch):
        k = len(x["input_ids"])
        ids[i, :k] = torch.tensor(x["input_ids"])
        labels[i, :k] = torch.tensor(x["labels"])
        attn[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def _training_kwargs(training_args_cls, cfg: QLoRAConfig, bf16: bool) -> dict:
    """TrainingArguments kwargs that work on transformers 4.x and 5.x (5.x folded warmup_ratio into
    warmup_steps and dropped group_by_length)."""
    import inspect

    accepted = set(inspect.signature(training_args_cls.__init__).parameters)
    kw = dict(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        bf16=bf16,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=bf16,  # only meaningful with an accelerator
        report_to=[],
    )
    if "warmup_ratio" in accepted:
        kw["warmup_ratio"] = cfg.warmup_ratio
    else:
        kw["warmup_steps"] = cfg.warmup_ratio  # a float in [0, 1) is a ratio in transformers 5
    if "group_by_length" in accepted:
        kw["group_by_length"] = True
    return kw


def train_qlora(cfg: QLoRAConfig, samples_path: str) -> str:
    """Train on pre-tokenized samples (exact rollout ids; no chat template is re-applied)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    cuda = torch.cuda.is_available()
    if cfg.quantize:
        if not cuda:
            raise RuntimeError("QLoRA needs a CUDA GPU (bitsandbytes 4-bit); pass quantize=False for plain LoRA")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.bfloat16 if cuda else torch.float32)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            task_type="CAUSAL_LM",
        ),  # fmt: skip
    )
    data = [
        {"input_ids": s.input_ids, "labels": s.labels}
        for s in read_jsonl(samples_path)
        if len(s.input_ids) <= cfg.max_len
    ]
    if not data:
        raise ValueError(f"no samples within max_len={cfg.max_len} in {samples_path}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    args = TrainingArguments(**_training_kwargs(TrainingArguments, cfg, bf16=cuda))
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=lambda b: _collate(b, pad))
    trainer.train()
    adapter_dir = f"{cfg.output_dir}/adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    return adapter_dir


def merge_adapter(base_model: str, adapter_dir: str, out_dir: str) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    try:
        AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    except (OSError, ValueError):
        AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    return out_dir
