"""5D parallelism planner: Megatron's layout rules, memory/time models, remedies, flags, CLI."""

import json

import pytest

from swarmrl.cli import main
from swarmrl.parallel.planner import (
    PRESETS,
    ClusterSpec,
    Layout,
    ModelSpec,
    TrainSpec,
    enumerate_layouts,
    estimate,
    plan,
    plan_rollout,
    violations,
)

ONE_NODE = ClusterSpec(nodes=1)


@pytest.mark.parametrize(
    "name,billions",
    [("gemma3-27b", 27.0), ("qwen3-32b", 32.8), ("llama3.1-70b", 70.6), ("qwen3-30b-a3b", 30.5),
     ("qwen3-235b-a22b", 235.1)],
)  # fmt: skip
def test_preset_sizes_match_published_counts(name, billions):
    assert PRESETS[name].total_params / 1e9 == pytest.approx(billions, rel=0.02)


def test_moe_active_params_are_a_small_fraction():
    m = PRESETS["qwen3-30b-a3b"]
    assert 2.5e9 < m.active_params < 3.6e9  # "A3B"


def test_hf_config_loader_handles_gemma3_nesting_and_sliding_window():
    cfg = {"text_config": {"hidden_size": 5376, "intermediate_size": 21504, "num_hidden_layers": 62,
                           "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 128,
                           "vocab_size": 262208, "tie_word_embeddings": True, "sliding_window": 1024,
                           "sliding_window_pattern": 6}}  # fmt: skip
    m = ModelSpec.from_hf_config(cfg, name="gemma3-27b")
    assert m == PRESETS["gemma3-27b"]
    # 52 of 62 layers keep only a 1024-token window, so long contexts cost far less KV than all-global
    full = ModelSpec.from_hf_config({**cfg["text_config"], "sliding_window_pattern": 0})
    assert m.kv_bytes(32768) < full.kv_bytes(32768) / 4


def test_hf_config_loader_reads_moe_fields():
    m = ModelSpec.from_hf_config({"hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 48,
                                  "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
                                  "vocab_size": 151936, "num_experts": 128, "num_experts_per_tok": 8,
                                  "moe_intermediate_size": 768})  # fmt: skip
    assert (m.experts, m.topk, m.moe_ffn) == (128, 8, 768)
    assert m.total_params == PRESETS["qwen3-30b-a3b"].total_params


@pytest.mark.parametrize(
    "layout,reason",
    [
        (Layout(tp=16, dp=1), "NVLink"),  # TP across nodes
        (Layout(tp=8, cp=1, dp=1), ""),  # valid on one node for gemma3-27b
        (Layout(tp=32 // 4, pp=1, cp=1, dp=1, ep=2), "MoE"),  # EP on a dense model
        (Layout(tp=1, pp=3, dp=8), "world"),
    ],
)
def test_violations(layout, reason):
    c = ClusterSpec(nodes=2) if layout.tp == 16 else ONE_NODE
    v = violations(layout, PRESETS["gemma3-27b"], c, TrainSpec())
    assert (reason in " ".join(v)) if reason else not v


def test_violations_heads_experts_and_context_split():
    t = TrainSpec(seq_len=32768)
    assert any("KV heads" in v for v in violations(Layout(tp=8, dp=1), PRESETS["qwen3-30b-a3b"], ONE_NODE, t))
    qwen_moe = PRESETS["qwen3-30b-a3b"]
    assert any("experts" in v for v in violations(Layout(tp=1, dp=8, ep=3), qwen_moe, ONE_NODE, t))
    assert any("seq_len" in v for v in violations(Layout(cp=8), PRESETS["qwen3-8b"], ONE_NODE, TrainSpec(seq_len=8200)))
    assert any("layers" in v for v in violations(Layout(pp=8), PRESETS["gemma3-27b"], ONE_NODE, t))


def test_every_enumerated_layout_is_valid_and_covers_the_cluster():
    for name in ("gemma3-27b", "qwen3-30b-a3b"):
        m, t = PRESETS[name], TrainSpec()
        lays = enumerate_layouts(m, ONE_NODE, t)
        assert lays and all(lay.world == 8 and not violations(lay, m, ONE_NODE, t) for lay in lays)
        if m.is_moe:  # parallel folding: EP is free to differ from DP
            assert any(lay.ep == 8 and lay.dp == 1 for lay in lays)


def test_memory_model_directions():
    m, t = PRESETS["gemma3-27b"], TrainSpec()
    tp2, tp8 = estimate(Layout(tp=2, dp=4), m, ONE_NODE, t), estimate(Layout(tp=8), m, ONE_NODE, t)
    assert tp8.memory_gib["weights"] == pytest.approx(tp2.memory_gib["weights"] / 4)
    # the distributed optimizer shards Adam states over DP
    dp2 = estimate(Layout(tp=8, dp=2), m, ClusterSpec(nodes=2), t)
    assert dp2.memory_gib["optimizer"] == pytest.approx(tp8.memory_gib["optimizer"] / 2)
    assert dp2.memory_gib["grads"] == pytest.approx(tp8.memory_gib["grads"])  # grad buffers are not sharded
    full = estimate(Layout(tp=8), m, ONE_NODE, TrainSpec(recompute="full"))
    assert full.memory_gib["activations"] < tp8.memory_gib["activations"] / 5
    assert full.time_s["compute"] > tp8.time_s["compute"]  # recompute costs an extra forward
    lora = estimate(Layout(tp=8), m, ONE_NODE, TrainSpec(lora_rank=32))
    assert lora.memory_gib["optimizer"] < 1 and lora.memory_gib["grads"] < 1


def test_cross_node_groups_pay_rdma_bandwidth():
    m, c, t = PRESETS["qwen3-32b"], ClusterSpec(nodes=2), TrainSpec(recompute="full")
    intra = estimate(Layout(tp=8, cp=2), m, c, t)  # CP crosses nodes
    inter = estimate(Layout(tp=4, cp=2, dp=2), m, c, t)  # DP crosses nodes
    assert "DP spans nodes" in " ".join(inter.notes) and "DP spans" not in " ".join(intra.notes)


def test_plan_applies_remedies_only_when_needed():
    p = plan(PRESETS["gemma3-27b"], ONE_NODE, TrainSpec())
    assert p.remedies == ["full recompute"] and p.best.fits and p.train.recompute == "full"
    lora = plan(PRESETS["gemma3-27b"], ONE_NODE, TrainSpec(lora_rank=32))
    assert lora.remedies == [] and lora.best.fits
    with pytest.raises(ValueError, match="does not fit"):
        plan(PRESETS["llama3.1-70b"], ClusterSpec(nodes=1, gpu_mem_gib=24), TrainSpec())
    with pytest.raises(ValueError, match="does not fit"):
        plan(PRESETS["gemma3-27b"], ONE_NODE, TrainSpec(), auto_fit=False)


def test_plan_is_the_fastest_fitting_layout():
    p = plan(PRESETS["qwen3-30b-a3b"], ONE_NODE, TrainSpec())
    fitting = [e for e in p.ranked if e.fits]
    assert p.best.step_s == min(e.step_s for e in fitting)
    assert p.best.layout.etp == 1  # narrow 768-wide experts: expert-TP only adds communication


def test_megatron_args_match_the_layout():
    p = plan(PRESETS["qwen3-235b-a22b"], ClusterSpec(nodes=8), TrainSpec())
    a, lay = p.megatron_args(), p.best.layout

    def flag(name):
        return a[a.index(name) + 1]

    assert int(flag("--tensor-model-parallel-size")) * int(flag("--pipeline-model-parallel-size")) * \
        int(flag("--context-parallel-size")) * lay.dp == 64  # fmt: skip
    assert int(flag("--expert-model-parallel-size")) == lay.ep
    assert ("--sequence-parallel" in a) == (lay.tp > 1)
    assert "--moe-enable-deepep" in a  # EP crosses nodes on 8 nodes
    assert p.env()["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert "--enable-dp-attention" in p.sglang_args()


def test_rollout_engines_prefer_many_small_engines():
    r = plan_rollout(PRESETS["qwen3-8b"], ONE_NODE, gpus=8, context=16384, min_concurrent=16)
    assert r.engine_tp == 1 and r.engines == 8
    big = plan_rollout(PRESETS["llama3.1-70b"], ONE_NODE, gpus=8, context=16384, min_concurrent=32)
    assert big.engine_tp >= 4 and big.max_concurrent >= 32
    huge = plan_rollout(PRESETS["qwen3-235b-a22b"], ClusterSpec(nodes=4), gpus=32, min_concurrent=32)
    assert huge.engine_tp >= 16 and huge.dp_attention  # multi-node engine with DP attention + EP


def test_cli_plan(capsys, tmp_path):
    assert main(["plan", "--model", "qwen3-8b", "--seq-len", "16384", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["layout"]["tp"] * out["layout"]["pp"] * out["layout"]["cp"] * out["layout"]["dp"] == 8
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 8,
                               "num_attention_heads": 8, "num_key_value_heads": 8, "vocab_size": 32000}))  # fmt: skip
    assert main(["plan", "--model", str(tmp_path), "--seq-len", "4096", "--global-tokens", "262144"]) == 0
    assert "chosen:" in capsys.readouterr().out
    assert main(["plan", "--model", "llama3.1-70b", "--gpu-mem-gib", "24"]) == 1
