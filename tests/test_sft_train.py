"""SFT training and merge for real on a tiny model (plain LoRA on CPU; QLoRA needs CUDA)."""

import asyncio

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")


def tiny_base(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    torch.manual_seed(0)
    transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=300,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=4096,
        )
    ).save_pretrained(path)
    vocab = {f"t{i}": i for i in range(300)}
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="t0")), pad_token="t0", eos_token="t1"
    )
    tok.save_pretrained(path)


def test_sft_train_and_merge(tmp_path, make_runner, task, factory):
    from swarmrl.sft.samples import build_sft_samples, read_jsonl, write_jsonl
    from swarmrl.sft.train import QLoRAConfig, merge_adapter, train_qlora
    from swarmrl.swarm.config import default_swarm_config
    from swarmrl.tasks.oracle_agent import oracle_file_states, oracle_script

    cfg = default_swarm_config()
    ep = asyncio.run(make_runner(oracle_script(oracle_file_states(task, factory), cfg), cfg).run(task))
    samples, _ = build_sft_samples([ep], max_len=4096)
    write_jsonl(samples, str(tmp_path / "sft.jsonl"))
    assert len(read_jsonl(str(tmp_path / "sft.jsonl"))) == len(samples)

    base = tmp_path / "base"
    tiny_base(base)
    qcfg = QLoRAConfig(
        str(base),
        str(tmp_path / "out"),
        lora_rank=4,
        lora_alpha=8,
        learning_rate=1e-3,
        epochs=1,
        grad_accum=1,
        max_len=4096,
        quantize=False,
        gradient_checkpointing=False,
        logging_steps=1,
    )
    adapter = train_qlora(qcfg, str(tmp_path / "sft.jsonl"))
    merged_dir = merge_adapter(str(base), adapter, str(tmp_path / "merged"))
    merged = transformers.AutoModelForCausalLM.from_pretrained(merged_dir)
    assert not any("lora" in k for k in merged.state_dict())
    orig = transformers.AutoModelForCausalLM.from_pretrained(base)
    diff = sum(
        (a - b).abs().sum().item()
        for a, b in zip(merged.state_dict().values(), orig.state_dict().values(), strict=True)
    )
    assert diff > 0  # the adapter changed the weights

    with pytest.raises(RuntimeError, match="CUDA"):
        if not torch.cuda.is_available():
            train_qlora(QLoRAConfig(str(base), str(tmp_path / "q")), str(tmp_path / "sft.jsonl"))
        else:
            raise RuntimeError("CUDA present: skip the negative check")
