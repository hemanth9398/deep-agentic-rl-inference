"""v2 training on tiny models: router calibration from outcomes, and one full step updating three separate models."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from test_v2_runtime import make_runner

from swarmrl.inference.chat import ByteTokenizer, ChatFormat
from swarmrl.rl.schedule import STAGES
from swarmrl.rl.trainer import TorchPolicyTrainer, TorchTrainerConfig
from swarmrl.tasks.oracle_agent import oracle_file_states
from swarmrl.v2.credit import VerifierTarget
from swarmrl.v2.loop import TriLoop
from swarmrl.v2.protocol import VerdictTokens
from swarmrl.v2.records import Call
from swarmrl.v2.router import GateConfig
from swarmrl.v2.train import VerifierTrainer


def tiny(tmp_path, name, hidden, cls=TorchPolicyTrainer, lr=1e-2):
    torch.manual_seed(0)
    path = tmp_path / name
    transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=300, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=8192)).save_pretrained(path)  # fmt: skip
    cfg = TorchTrainerConfig(str(path), str(tmp_path / f"{name}-out"), lora_rank=8, lora_alpha=16, learning_rate=lr,
                             gradient_checkpointing=False, dtype="float32")  # fmt: skip
    return cls(cfg, device="cpu")


def lora_snapshot(trainer):
    return {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}


def changed(before, trainer):
    return any(not torch.equal(before[n], p) for n, p in trainer.model.named_parameters() if p.requires_grad)


def _items(chat):
    good = Call("router", chat.start("judge", "a" * 60), [], [])
    bad = Call("router", chat.start("judge", "z" * 60), [], [])
    return [VerifierTarget(good, 1.0, 2.0, ("e", 0, 0), "patch"), VerifierTarget(bad, 0.0, 1.0, ("e", 0, 0), "patch")]


def test_router_gradient_is_exact(tmp_path):
    """The sequence-at-a-time backward equals autograd on the whole batch loss (BCE + ranking)."""
    router = tiny(tmp_path, "router", 32, VerifierTrainer)
    chat = ChatFormat(ByteTokenizer())
    vt = VerdictTokens.for_chat(chat)
    ts = _items(chat)
    router.optimizer.zero_grad(set_to_none=True)
    router.verifier_backward(ts, vt.yes, vt.no, rank_weight=0.5)
    ours = {n: p.grad.clone() for n, p in router.model.named_parameters() if p.grad is not None}
    router.optimizer.zero_grad(set_to_none=True)
    z = [router.verdict_logit(t.call.input_ids, vt.yes, vt.no, grad=True) for t in ts]
    w = torch.tensor([t.weight for t in ts])
    y = torch.tensor([t.label for t in ts])
    zz = torch.stack(z)
    loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(zz, y, reduction="none")).sum() / w.sum()
    loss = loss + 0.5 * torch.nn.functional.softplus(zz[1] - zz[0])
    loss.backward()
    ref = {n: p.grad.clone() for n, p in router.model.named_parameters() if p.grad is not None}
    assert ours.keys() == ref.keys() and ours
    for n in ref:
        assert torch.allclose(ours[n], ref[n], atol=1e-6, rtol=1e-4), n


def test_router_verifier_loss_calibrates_toward_outcomes(tmp_path):
    router = tiny(tmp_path, "router", 32, VerifierTrainer, lr=5e-2)
    chat = ChatFormat(ByteTokenizer())
    vt = VerdictTokens.for_chat(chat)
    ts = _items(chat)
    before = router.probabilities(ts, vt.yes, vt.no)
    for _ in range(30):
        out = router.update_verifier(ts, vt.yes, vt.no)
    after = router.probabilities(ts, vt.yes, vt.no)
    assert out["stepped"] == 1.0
    # A random 2-layer model with a frozen LM head can only move the YES/NO logit by about +-1 (p ~ 0.72):
    # assert the direction and the separation, which is what the loss controls.
    assert after[0] > before[0] + 0.15 and after[1] < before[1] - 0.15
    assert after[0] > 0.65 and after[1] < 0.35


def test_one_v2_step_updates_three_separate_models(tmp_path, task, factory, record):
    fixed = oracle_file_states(task, factory)["stats/core.py"]
    runner, _ = make_runner(task, factory, record, fixed, lambda good: 0.05 if good else 0.9,
                            GateConfig(epsilon=1.0), plans=("good", "bad"))  # fmt: skip
    planner, coder = tiny(tmp_path, "planner", 64), tiny(tmp_path, "coder", 32)  # different sizes, own LoRAs
    router = tiny(tmp_path, "router", 32, VerifierTrainer)
    snaps = {n: lora_snapshot(t) for n, t in (("planner", planner), ("coder", coder), ("router", router))}
    seen = []
    loop = TriLoop(runner, [task], planner, coder, router, STAGES[1], group_size=2, on_metrics=seen.append)
    m = loop.step(0)
    assert m["infra_errors"] == 0 and m["shadows"] >= 1 and seen == [m]
    assert m["targets/planner"] > 0 and m["targets/coder"] > 0 and m["targets/router"] > 0
    for role, trainer in (("planner", planner), ("coder", coder), ("router", router)):
        assert m[f"train/{role}/stepped"] == 1.0, role
        assert changed(snaps[role], trainer), f"{role} LoRA did not move"
    assert m["router/false_reject_rate"] == 1.0  # the router rejected the good plan; its shadow proved it wrong
    assert loop.version == 1 and (tmp_path / "router-out").exists()
