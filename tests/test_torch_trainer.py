"""TorchPolicyTrainer on a tiny randomly initialized model (CPU, seconds)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from swarmrl.rl.packing import PackedSample
from swarmrl.rl.schedule import stage
from swarmrl.rl.trainer import TorchPolicyTrainer, TorchTrainerConfig


def tiny_model():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=300,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    return transformers.LlamaForCausalLM(cfg)


def trainer(tmp_path, **kw):
    cfg = TorchTrainerConfig(
        "tiny",
        str(tmp_path / "ckpt"),
        lora_rank=4,
        lora_alpha=8,
        learning_rate=5e-2,
        gradient_checkpointing=False,
        dtype="float32",
        logprob_chunk=7,
        **kw,
    )
    return TorchPolicyTrainer(cfg, model=tiny_model(), device="cpu")


def sample(ids, prompt_len):
    n = len(ids)
    mask = [0] * prompt_len + [1] * (n - prompt_len)
    return PackedSample("e", "t", "g", "coder", prompt_len, list(ids), mask, [0.0] * n)


def test_logprobs_match_direct_computation(tmp_path):
    tr = trainer(tmp_path)
    ids = list(range(5, 45))
    lp = tr.logprobs([sample(ids, 10)])[0]
    with torch.no_grad():
        logits = tr.model(input_ids=torch.tensor([ids])).logits[0].float()
        ref = torch.log_softmax(logits[:-1], -1).gather(-1, torch.tensor(ids[1:])[:, None])[:, 0]
    assert lp.shape == (len(ids),) and lp[0] == 0.0
    np.testing.assert_allclose(lp[1:], ref.numpy(), atol=1e-5)  # chunk size 7 crosses many boundaries


def _update(tr, s, adv_value, st, steps=1):
    mask = np.asarray(s.loss_mask, dtype=float)
    for _ in range(steps):
        old = tr.logprobs([s])
        m = tr.update([s], [mask * adv_value], old, [np.ones_like(mask)], [mask], st)
    return m


def test_positive_advantage_raises_logprob(tmp_path):
    tr = trainer(tmp_path)
    s = sample(list(range(5, 30)), 10)
    mask = np.asarray(s.loss_mask, dtype=bool)
    before = tr.logprobs([s])[0][mask].sum()
    m = _update(tr, s, 1.0, stage(1), steps=5)
    after = tr.logprobs([s])[0][mask].sum()
    assert m["stepped"] == 1.0 and m["policy_tokens"] == mask.sum() and np.isfinite(m["grad_norm"])
    assert after > before + 0.1
    tr2 = trainer(tmp_path / "neg")
    b2 = tr2.logprobs([s])[0][mask].sum()
    _update(tr2, s, -1.0, stage(1), steps=5)
    assert tr2.logprobs([s])[0][mask].sum() < b2 - 0.1


def test_stage0_lr_zero_changes_nothing(tmp_path):
    tr = trainer(tmp_path)
    s = sample(list(range(5, 30)), 10)
    before = tr.logprobs([s])[0]
    m = _update(tr, s, 1.0, stage(0), steps=2)
    assert m["stepped"] == 0.0 and m["lr"] == 0.0
    np.testing.assert_array_equal(tr.logprobs([s])[0], before)


def test_save_adapter_and_merged(tmp_path):
    from peft import PeftModel

    s = sample(list(range(5, 30)), 10)
    tr = trainer(tmp_path, save_mode="merged", keep_checkpoints=2)
    _update(tr, s, 1.0, stage(1), steps=2)  # make LoRA non-zero so merging matters
    want = tr.logprobs([s])[0]
    path = tr.save(1)
    merged = transformers.AutoModelForCausalLM.from_pretrained(path)
    assert not any("lora" in k for k in merged.state_dict())
    with torch.no_grad():
        logits = merged(input_ids=torch.tensor([s.input_ids])).logits[0].float()
        got = torch.log_softmax(logits[:-1], -1).gather(-1, torch.tensor(s.input_ids[1:])[:, None])[:, 0]
    np.testing.assert_allclose(got.numpy(), want[1:], atol=1e-4)
    tr.save(2), tr.save(3)
    import os

    assert sorted(os.listdir(tmp_path / "ckpt")) == ["step-000002", "step-000003"]  # pruned to keep_checkpoints

    ta = trainer(tmp_path / "a", save_mode="adapter")
    _update(ta, s, 1.0, stage(1), steps=2)
    p = ta.save(1)
    reloaded = PeftModel.from_pretrained(tiny_model(), p)
    with torch.no_grad():
        l1 = reloaded(input_ids=torch.tensor([s.input_ids])).logits
        l2 = ta.model(input_ids=torch.tensor([s.input_ids])).logits
    torch.testing.assert_close(l1, l2, atol=1e-5, rtol=1e-5)


def test_rl_loop_step_with_torch_trainer(tmp_path, make_runner, task, factory):
    from dataclasses import replace

    from swarmrl.inference.weight_sync import NullWeightSync
    from swarmrl.rl.curriculum import Curriculum
    from swarmrl.rl.loop import RLLoop
    from swarmrl.swarm.config import default_swarm_config
    from swarmrl.swarm.dispatcher import Dispatcher
    from swarmrl.tasks.oracle_agent import oracle_file_states, oracle_script

    cfg = default_swarm_config()
    st = replace(stage(1), group_size=2)
    runner = make_runner(oracle_script(oracle_file_states(task, factory), cfg), st.apply(cfg))
    tr = trainer(tmp_path)
    sync = NullWeightSync()
    rows = []
    loop = RLLoop(
        Dispatcher(runner),
        tr,
        sync,
        Curriculum({task.task_id: 0.5}),
        st,
        {task.task_id: task},
        batch_tasks=1,
        log=rows.append,
    )
    loop.run(2)
    assert len(rows) == 2 and rows[0]["policy_tokens"] > 0 and rows[0]["episodes"] == 2
    assert rows[0]["all_one_groups"] == 1  # oracle always wins: no signal, zero advantages
    assert rows[1]["train/policy_tokens"] == 0.0  # zero-advantage samples are kept but carry no gradient
    assert ("step-000001" in sync.calls[1][0]) and loop.checkpoint.endswith("step-000002")


def test_ratio_is_one_before_the_first_step(tmp_path):
    """Scoring and the update pass must agree exactly, even for a model configured with dropout."""
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=300,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0.5,
        )
    )
    cfg = TorchTrainerConfig(
        "tiny", str(tmp_path), lora_rank=4, lora_dropout=0.3, gradient_checkpointing=False, dtype="float32"
    )
    tr = TorchPolicyTrainer(cfg, model=model, device="cpu")
    s = sample(list(range(5, 30)), 10)
    old = tr.logprobs([s])
    mask = np.asarray(s.loss_mask, dtype=float)
    m = tr.update([s], [mask], old, [np.ones_like(mask)], [mask], stage(0))
    assert m["clipfrac"] == 0.0 and abs(m["approx_kl"]) < 1e-6
