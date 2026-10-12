"""5D parallelism planner for Megatron training and SGLang rollouts on H100 nodes.

Training layouts are TP x CP x DP x PP for attention/dense layers, and ETP x EP x EDP x PP for MoE expert
layers (Megatron "parallel folding": both factorizations cover the same GPUs). The planner enumerates every
layout Megatron accepts for a model and cluster, estimates per-GPU memory and step time with the standard
analytic models, and returns the fastest layout that fits, with the Megatron flags to launch it.
Rollout engines (SGLang) are sized separately: the smallest tensor-parallel size that leaves enough KV cache
for the target concurrency, because more, smaller engines give more rollout throughput.

Rank order (Megatron's default "tp-cp-ep-dp-pp"): TP is innermost, so it stays on NVLink; a group spans
nodes only when stride x size exceeds the GPUs per node, and then runs at the NIC bandwidth.

The models are estimates for ranking layouts, not promises: check the chosen layout's memory with a short
real run (`swarmrl rl --stage 0`) before a long job. Constants below are documented where they are set.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

GIB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    hidden: int
    ffn: int  # dense MLP width (SwiGLU: gate, up, down); for MoE, dense-layer width if any
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied_embeddings: bool = False
    experts: int = 0  # routed experts per MoE layer (0 = dense model)
    topk: int = 0
    moe_ffn: int = 0  # width of one routed expert
    shared_ffn: int = 0  # width of the shared expert(s), if any
    sliding_window: int = 0  # local-attention window (Gemma 3: 1024)
    global_every: int = 0  # every n-th layer is global (Gemma 3: 6); 0 = all global

    @property
    def is_moe(self) -> bool:
        return self.experts > 0

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def attn_params(self) -> int:
        return 2 * self.hidden * self.q_dim + 2 * self.hidden * self.kv_dim

    @property
    def dense_mlp_params(self) -> int:
        """Per layer, replicated across EP: the dense MLP, or for MoE the shared expert and router."""
        if self.is_moe:
            return 3 * self.hidden * self.shared_ffn + self.hidden * self.experts
        return 3 * self.hidden * self.ffn

    @property
    def expert_params(self) -> int:
        return 3 * self.hidden * self.moe_ffn * self.experts if self.is_moe else 0

    @property
    def embedding_params(self) -> int:
        return self.vocab * self.hidden

    @property
    def total_params(self) -> int:
        per_layer = self.attn_params + self.dense_mlp_params + self.expert_params
        return self.layers * per_layer + self.embedding_params * (1 if self.tied_embeddings else 2)

    @property
    def active_params(self) -> int:
        """Parameters each token multiplies through, including the LM head (drives FLOPs)."""
        mlp = self.dense_mlp_params + (3 * self.hidden * self.moe_ffn * self.topk if self.is_moe else 0)
        return self.layers * (self.attn_params + mlp) + self.embedding_params

    @property
    def mlp_width_per_token(self) -> int:
        return self.topk * self.moe_ffn + self.shared_ffn if self.is_moe else self.ffn

    def attended(self, layer: int, context: int) -> int:
        """Tokens a query at position `context` attends to in `layer`."""
        local = self.sliding_window and self.global_every and (layer + 1) % self.global_every != 0
        return min(context, self.sliding_window) if local else context

    def kv_bytes(self, context: int, dtype_bytes: int = 2) -> int:
        """KV-cache bytes for one sequence of `context` tokens (sliding-window layers keep only the window)."""
        kept = sum(self.attended(i, context) for i in range(self.layers))
        return 2 * self.kv_dim * dtype_bytes * kept

    def attention_flops(self, seq: int) -> float:
        """Forward attention FLOPs per token, causal, averaged over positions of a `seq`-token sample."""
        avg = seq / 2
        return sum(4 * self.q_dim * self.attended(i, avg) for i in range(self.layers))

    @classmethod
    def from_hf_config(cls, cfg: dict, name: str = "") -> ModelSpec:
        cfg = cfg.get("text_config", cfg)  # Gemma 3 nests the language model's config
        hidden = cfg["hidden_size"]
        heads = cfg["num_attention_heads"]
        experts = cfg.get("num_experts") or cfg.get("num_local_experts") or cfg.get("n_routed_experts") or 0
        shared = cfg.get("shared_expert_intermediate_size") or 0
        if not shared and cfg.get("n_shared_experts"):
            shared = cfg["n_shared_experts"] * cfg["moe_intermediate_size"]
        pattern = cfg.get("sliding_window_pattern") or 0
        if not pattern and cfg.get("layer_types"):
            types = cfg["layer_types"]
            pattern = len(types) // max(1, types.count("full_attention")) if "sliding_attention" in types else 0
        return cls(
            name=name or cfg.get("_name_or_path", "model"),
            layers=cfg["num_hidden_layers"],
            hidden=hidden,
            ffn=cfg.get("intermediate_size", 0),
            heads=heads,
            kv_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or hidden // heads,
            vocab=cfg["vocab_size"],
            tied_embeddings=cfg.get("tie_word_embeddings", False),
            experts=experts,
            topk=cfg.get("num_experts_per_tok", 0) if experts else 0,
            moe_ffn=cfg.get("moe_intermediate_size", 0) if experts else 0,
            shared_ffn=shared,
            sliding_window=(cfg.get("sliding_window") or 0) if pattern else 0,
            global_every=pattern,
        )

    @classmethod
    def load(cls, name_or_path: str) -> ModelSpec:
        """A preset name, a model directory, or a config.json path."""
        if name_or_path in PRESETS:
            return PRESETS[name_or_path]
        p = Path(name_or_path)
        if p.is_dir():
            p = p / "config.json"
        if not p.exists():
            known = ", ".join(sorted(PRESETS))
            raise ValueError(f"unknown model {name_or_path!r}: use a config.json, a model dir, or one of {known}")
        return cls.from_hf_config(json.loads(p.read_text()), name=p.parent.name)


PRESETS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec("gemma3-12b", 48, 3840, 15360, 16, 8, 256, 262208, True, sliding_window=1024, global_every=6),
        ModelSpec("gemma3-27b", 62, 5376, 21504, 32, 16, 128, 262208, True, sliding_window=1024, global_every=6),
        ModelSpec("qwen3-8b", 36, 4096, 12288, 32, 8, 128, 151936),
        ModelSpec("qwen3-32b", 64, 5120, 25600, 64, 8, 128, 151936),
        ModelSpec("llama3.1-70b", 80, 8192, 28672, 64, 8, 128, 128256),
        ModelSpec("qwen3-30b-a3b", 48, 2048, 6144, 32, 4, 128, 151936, experts=128, topk=8, moe_ffn=768),
        ModelSpec("qwen3-235b-a22b", 94, 4096, 12288, 64, 4, 128, 151936, experts=128, topk=8, moe_ffn=1536),
    ]
}  # fmt: skip


@dataclass(frozen=True)
class ClusterSpec:
    nodes: int = 1
    gpus_per_node: int = 8
    gpu_mem_gib: float = 80.0  # H100 SXM
    intra_gbs: float = 450.0  # NVLink 4, per GPU per direction (GB/s)
    inter_gbs: float = 50.0  # one 400 Gb/s RoCE/IB NIC per GPU (GB/s)
    peak_tflops: float = 989.0  # dense BF16; FP8 doubles it
    reserve_gib: float = 6.0  # CUDA context, NCCL buffers, allocator fragmentation

    @property
    def world(self) -> int:
        return self.nodes * self.gpus_per_node


@dataclass(frozen=True)
class TrainSpec:
    seq_len: int = 32768  # longest training sample (tokens)
    tokens_per_microbatch: int = 0  # dynamic batching budget per micro-batch (0 = seq_len)
    global_tokens: int = 1 << 21  # tokens per optimizer step
    precision: str = "bf16"  # "bf16" | "fp8" (fp8 widens the rollout/trainer log-prob gap)
    recompute: str = "selective"  # "selective" | "full"
    lora_rank: int = 0  # 0 = full-parameter training
    optimizer_offload: bool = False  # Adam states on CPU (HybridDeviceOptimizer)
    bf16_moments: bool = False  # precision-aware optimizer: bf16 Adam moments (8 instead of 12 bytes/param)
    grad_fp32: bool = True  # fp32 main-grad buffers (Megatron default)

    @property
    def mb_tokens(self) -> int:
        return self.tokens_per_microbatch or self.seq_len


@dataclass(frozen=True)
class Layout:
    tp: int = 1
    pp: int = 1
    cp: int = 1
    dp: int = 1
    ep: int = 1
    etp: int = 1
    vpp: int = 1

    @property
    def world(self) -> int:
        return self.tp * self.pp * self.cp * self.dp

    @property
    def edp(self) -> int:
        return self.tp * self.cp * self.dp // (self.ep * self.etp)

    def __str__(self) -> str:
        s = f"TP{self.tp} PP{self.pp} CP{self.cp} DP{self.dp}"
        if self.ep > 1 or self.etp > 1:
            s += f" EP{self.ep} ETP{self.etp} EDP{self.edp}"
        return s + (f" VPP{self.vpp}" if self.vpp > 1 else "")


# Fractions of each communication left exposed after Megatron's overlap features
# (--tp-comm-overlap, CP's ring overlapping attention, --overlap-grad-reduce/--overlap-param-gather,
# DeepEP/alltoall overlap). Conservative values; they only shift the ranking between close layouts.
EXPOSED = {"tp": 0.5, "cp": 0.3, "ep": 0.6, "dp": 0.25, "pp": 1.0}
BASE_MFU = {"bf16": 0.50, "fp8": 0.35}  # fp8: 2x peak but lower utilization, ~1.4x net, as measured on H100


def _gemm_eff(width: float) -> float:
    """GEMM efficiency drops when a sharded weight gets narrow (e.g. a 768-wide expert split 4 ways)."""
    return min(1.0, width / 1024) ** 0.25


HEADROOM = 0.95  # keep 5% of HBM free beyond the reserve: the estimates are not exact


@dataclass
class Estimate:
    layout: Layout
    fits: bool
    memory_gib: dict[str, float]
    time_s: dict[str, float]
    microbatches: int
    notes: list[str] = field(default_factory=list)

    @property
    def peak_gib(self) -> float:
        return self.memory_gib["total"]

    @property
    def step_s(self) -> float:
        return self.time_s["total"]


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def violations(lay: Layout, m: ModelSpec, c: ClusterSpec, t: TrainSpec) -> list[str]:
    """Why Megatron (or good practice) rejects this layout; empty if it is valid."""
    out = []
    if lay.world != c.world:
        out.append(f"world {lay.world} != {c.world} GPUs")
    if lay.tp > c.gpus_per_node:
        out.append("TP must stay inside one NVLink domain")
    if m.heads % lay.tp or m.kv_heads % lay.tp:
        out.append(f"TP{lay.tp} must divide attention heads ({m.heads}) and KV heads ({m.kv_heads})")
    if m.layers % (lay.pp * lay.vpp):
        out.append(f"PP x VPP ({lay.pp * lay.vpp}) must divide {m.layers} layers")
    if lay.vpp > 1 and lay.pp == 1:
        out.append("virtual pipeline needs PP > 1")
    if lay.cp > 1 and t.seq_len % (2 * lay.cp):
        out.append(f"CP{lay.cp} needs seq_len divisible by {2 * lay.cp} (load-balanced causal split)")
    if lay.cp > 1 and t.mb_tokens % (2 * lay.cp):
        out.append(f"CP{lay.cp} needs tokens per micro-batch divisible by {2 * lay.cp}")
    if m.is_moe:
        if m.experts % lay.ep:
            out.append(f"EP{lay.ep} must divide {m.experts} experts")
        if (lay.tp * lay.cp * lay.dp) % (lay.ep * lay.etp):
            out.append("EP x ETP must divide TP x CP x DP (parallel folding)")
        if lay.etp > lay.tp:
            out.append("ETP > TP is never useful")
    elif lay.ep > 1 or lay.etp > 1:
        out.append("EP/ETP apply to MoE models only")
    if t.global_tokens % (lay.dp * t.mb_tokens):
        out.append(f"global tokens {t.global_tokens} not a multiple of DP x micro-batch tokens")
    return out


def _spans_nodes(stride: int, size: int, c: ClusterSpec) -> bool:
    return size > 1 and stride * size > c.gpus_per_node


def _bw(stride: int, size: int, c: ClusterSpec) -> float:
    return (c.inter_gbs if _spans_nodes(stride, size, c) else c.intra_gbs) * 1e9


def _lora_params_per_layer(m: ModelSpec, r: int) -> int:
    mlp = m.mlp_width_per_token if not m.is_moe else m.shared_ffn + m.moe_ffn * m.experts
    return r * (2 * (m.hidden + m.q_dim) + 2 * (m.hidden + m.kv_dim)) + r * 3 * (m.hidden + mlp)


def estimate(lay: Layout, m: ModelSpec, c: ClusterSpec, t: TrainSpec) -> Estimate:
    """Per-GPU memory (worst pipeline stage) and step time for one valid layout."""
    lps = m.layers // lay.pp
    mb = t.mb_tokens
    micro = max(1, t.global_tokens // (lay.dp * mb))
    tok_local = mb / lay.cp  # tokens one GPU holds per micro-batch

    # --- memory ---
    emb_copies = 1 if (lay.pp > 1 or m.tied_embeddings) else 2
    dense_local = lps * (m.attn_params + m.dense_mlp_params) / lay.tp
    emb_local = m.embedding_params / lay.tp
    expert_local = lps * m.expert_params / (lay.ep * lay.etp)
    if t.lora_rank:
        trainable_dense, trainable_expert = lps * _lora_params_per_layer(m, t.lora_rank) / lay.tp, 0.0
        emb_train = 0.0
    else:
        trainable_dense, trainable_expert, emb_train = dense_local, expert_local, emb_local
    grad_b = 4 if t.grad_fp32 else 2

    def stage_mem(first: bool, last: bool) -> dict[str, float]:
        n_emb = emb_copies if lay.pp == 1 else 1  # PP > 1: input embedding on the first stage, LM head on the last
        weights = 2 * (dense_local + expert_local + n_emb * emb_local)
        td, te = trainable_dense + n_emb * emb_train, trainable_expert
        grads = grad_b * (td + te)
        opt_b = 8 if t.bf16_moments else 12  # fp32 master weights + two Adam moments
        opt = 0.0 if t.optimizer_offload else opt_b * (td / (lay.dp * lay.cp) + te / max(1, lay.edp))
        if t.recompute == "full":
            per_tok_layer = 2 * m.hidden / lay.tp
            full_layer = 2 * (3 * m.hidden + 2 * m.q_dim + 2 * m.kv_dim + 3 * m.mlp_width_per_token) / lay.tp
            one_mb = tok_local * (lps * per_tok_layer + full_layer)
        else:  # flash attention: selective recompute keeps everything but the attention core
            per_tok_layer = 2 * (3 * m.hidden + 2 * m.q_dim + 2 * m.kv_dim + 3 * m.mlp_width_per_token) / lay.tp
            one_mb = tok_local * lps * per_tok_layer
        if first:  # 1F1B: the first stage holds pp micro-batches (more with interleaving)
            inflight = lay.pp * (1 + (lay.pp - 1) / (lay.pp * lay.vpp)) if lay.vpp > 1 else lay.pp
            inflight = min(inflight, micro)
        else:
            inflight = 1
        acts = one_mb * inflight
        logits = tok_local * m.vocab / lay.tp * 6 if last else 0.0  # bf16 logits + fp32 log-softmax
        return {
            "weights": weights / GIB, "grads": grads / GIB, "optimizer": opt / GIB,
            "activations": acts / GIB, "logits": logits / GIB, "reserve": c.reserve_gib,
        }  # fmt: skip

    stages = [stage_mem(True, lay.pp == 1)] + ([stage_mem(False, True)] if lay.pp > 1 else [])
    mem = max(stages, key=lambda s: sum(s.values()))
    mem["total"] = sum(mem.values())

    # --- time ---
    rep_tokens = t.global_tokens / lay.dp
    expert_fwd = 2 * m.layers * 3 * m.hidden * m.moe_ffn * m.topk if m.is_moe else 0.0
    dense_fwd = 2 * m.active_params - expert_fwd + m.attention_flops(t.seq_len)
    dense_width = (m.shared_ffn if m.is_moe else m.ffn) or m.q_dim
    fwd_per_tok = dense_fwd / _gemm_eff(dense_width / lay.tp) + expert_fwd / _gemm_eff(m.moe_ffn / lay.etp or 1)
    mult = (2 if t.lora_rank else 3) + (1 if t.recompute == "full" else 0)
    peak = c.peak_tflops * 1e12 * (2 if t.precision == "fp8" else 1) * BASE_MFU[t.precision]
    compute = rep_tokens * fwd_per_tok * mult / (lay.tp * lay.cp * lay.pp * peak)
    bubble = compute * (lay.pp - 1) / (lay.vpp * micro) if lay.pp > 1 else 0.0

    tok_gpu = rep_tokens / lay.cp  # tokens one GPU touches per step (per its pipeline stage)
    comm = {}
    comm["tp"] = (8 * (lay.tp - 1) / lay.tp * lps * tok_gpu * m.hidden * 2) / _bw(1, lay.tp, c) if lay.tp > 1 else 0.0
    comm["cp"] = (3 * (lay.cp - 1) / lay.cp * lps * tok_gpu * 2 * m.kv_dim * 2) / _bw(lay.tp, lay.cp, c) \
        if lay.cp > 1 else 0.0  # fmt: skip
    if m.is_moe and lay.ep > 1:
        a2a = 4 * m.topk * m.hidden * 2 * (lay.ep - 1) / lay.ep * lps * tok_gpu / lay.tp
        comm["ep"] = a2a / _bw(lay.etp, lay.ep, c)
    else:
        comm["ep"] = 0.0
    if m.is_moe and lay.etp > 1:  # expert TP: routed tokens all-gathered / reduce-scattered inside the ETP group
        comm["ep"] += 4 * (lay.etp - 1) / lay.etp * lps * tok_gpu / lay.tp * m.topk * m.hidden * 2 / _bw(1, lay.etp, c)
    dp_group = lay.dp * lay.cp
    # distributed optimizer: reduce-scatter grads, all-gather updated bf16 params
    dense_sync = (grad_b + 2) * (dp_group - 1) / dp_group * (trainable_dense + emb_train) if dp_group > 1 else 0.0
    expert_sync = (grad_b + 2) * (lay.edp - 1) / lay.edp * trainable_expert if lay.edp > 1 else 0.0
    comm["dp"] = dense_sync / _bw(lay.tp, dp_group, c) + expert_sync / _bw(lay.ep * lay.etp, lay.edp, c)
    comm["pp"] = (2 * micro * tok_local * m.hidden * 2 / lay.tp) / _bw(lay.tp * lay.cp * lay.dp, lay.pp, c) \
        if lay.pp > 1 else 0.0  # fmt: skip

    time = {"compute": compute, "pp_bubble": bubble}
    time.update({f"{k}_comm": v * EXPOSED[k] for k, v in comm.items()})
    if t.optimizer_offload:  # grads down, params up over PCIe (~20 GB/s effective), partly overlapped
        local = trainable_dense / (lay.dp * lay.cp) + trainable_expert / max(1, lay.edp)
        time["optimizer_offload"] = local * (grad_b + 2) / 20e9 + local * 16 / 50e9  # + CPU Adam at ~50 GB/s
    time["total"] = sum(time.values())

    usable = c.gpu_mem_gib * HEADROOM
    notes = []
    if lay.tp > 1:
        notes.append("sequence parallel on (TP > 1)")
    if _spans_nodes(lay.tp * lay.cp, lay.dp, c):
        notes.append("DP spans nodes: grad sync rides RDMA")
    if m.is_moe and _spans_nodes(lay.etp, lay.ep, c):
        notes.append("EP spans nodes: use DeepEP")
    return Estimate(lay, mem["total"] <= usable, mem, time, micro, notes)


def enumerate_layouts(m: ModelSpec, c: ClusterSpec, t: TrainSpec) -> list[Layout]:
    out = []
    n = c.world
    for tp, cp, pp in itertools.product(_divisors(c.gpus_per_node), _divisors(n), _divisors(n)):
        if n % (tp * cp * pp):
            continue
        dp = n // (tp * cp * pp)
        eps = _divisors(m.experts) if m.is_moe else [1]
        for ep, etp, vpp in itertools.product(eps, _divisors(tp) if m.is_moe else [1], (1, 2, 4)):
            lay = Layout(tp, pp, cp, dp, ep, etp, vpp)
            if not violations(lay, m, c, t):
                out.append(lay)
    return out


@dataclass
class RolloutPlan:
    engine_tp: int
    engines: int
    ep: int
    dp_attention: bool
    weights_gib: float
    kv_gib: float
    max_concurrent: int
    context: int
    mem_fraction: float


def plan_rollout(m: ModelSpec, c: ClusterSpec, gpus: int, context: int = 16384, min_concurrent: int = 32,
                 mem_fraction: float = 0.85) -> RolloutPlan:  # fmt: skip
    """Smallest engine (TP) whose KV cache holds `min_concurrent` sequences of `context` tokens."""
    best = None
    sizes = [d for d in _divisors(gpus) if d <= c.gpus_per_node or d % c.gpus_per_node == 0]  # multi-node engines
    for tp in sizes:
        if gpus % tp:
            continue
        budget = c.gpu_mem_gib * mem_fraction
        weights = 2 * m.total_params / tp / GIB
        kv_shards = min(tp, m.kv_heads)  # beyond kv_heads, KV heads are replicated
        kv_per_seq = m.kv_bytes(context) / kv_shards / GIB
        kv_room = budget - weights - 2.0  # activations / CUDA graphs
        if kv_room <= 0:
            continue
        conc = int(kv_room // kv_per_seq) * 1  # per engine (each GPU holds its KV shard)
        best = RolloutPlan(tp, gpus // tp, tp if m.is_moe and tp > 1 else 1, m.is_moe and tp > 1,
                           weights, kv_room, conc, context, mem_fraction)  # fmt: skip
        if conc >= min_concurrent:
            return best
    if best is None:
        raise ValueError(f"{m.name}: weights do not fit in {gpus} GPUs for inference")
    return best


@dataclass
class Plan:
    model: ModelSpec
    cluster: ClusterSpec
    train: TrainSpec
    best: Estimate
    ranked: list[Estimate]
    rollout: RolloutPlan
    colocated: bool
    remedies: list[str] = field(default_factory=list)

    def megatron_args(self) -> list[str]:
        lay, t, m = self.best.layout, self.train, self.model
        a = [
            "--tensor-model-parallel-size", str(lay.tp),
            "--pipeline-model-parallel-size", str(lay.pp),
            "--context-parallel-size", str(lay.cp),
            "--transformer-impl", "transformer_engine",
            "--use-distributed-optimizer",
            "--overlap-grad-reduce",
            "--overlap-param-gather",
            "--attention-backend", "auto",
            "--cross-entropy-loss-fusion",
            "--use-dynamic-batch-size", "--max-tokens-per-gpu", str(t.mb_tokens // lay.cp),
        ]  # fmt: skip
        if lay.tp > 1:
            a += ["--sequence-parallel"]
        if lay.vpp > 1:
            a += ["--num-layers-per-virtual-pipeline-stage", str(m.layers // (lay.pp * lay.vpp))]
        if m.is_moe:
            a += ["--expert-model-parallel-size", str(lay.ep), "--expert-tensor-parallel-size", str(lay.etp),
                  "--moe-grouped-gemm", "--moe-permute-fusion", "--moe-router-dtype", "fp32"]  # fmt: skip
            if _spans_nodes(lay.etp, lay.ep, self.cluster):
                a += ["--moe-token-dispatcher-type", "flex", "--moe-enable-deepep"]
            else:
                a += ["--moe-token-dispatcher-type", "alltoall"]
        if t.recompute == "full":
            a += ["--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1"]
        else:
            a += ["--recompute-granularity", "selective"]
        a += ["--bf16"]
        if t.precision == "fp8":
            a += ["--fp8-format", "hybrid", "--fp8-recipe", "blockwise"]
        if t.optimizer_offload or t.bf16_moments:
            a += ["--use-precision-aware-optimizer"]
        if t.bf16_moments:
            a += ["--exp-avg-dtype", "bf16", "--exp-avg-sq-dtype", "bf16"]
        if t.optimizer_offload:
            a += ["--optimizer-cpu-offload", "--overlap-cpu-optimizer-d2h-h2d"]
        if not t.grad_fp32:
            a += ["--grad-reduce-in-bf16"]
        return a

    def sglang_args(self) -> list[str]:
        r = self.rollout
        a = ["--tp-size", str(r.engine_tp), "--mem-fraction-static", str(r.mem_fraction),
             "--context-length", str(r.context)]  # fmt: skip
        if r.dp_attention:
            a += ["--enable-dp-attention", "--dp-size", str(r.engine_tp), "--ep-size", str(r.ep)]
        return a

    def env(self) -> dict[str, str]:
        # TP comm overlap needs one hardware queue so communication kernels are issued in order.
        return {"CUDA_DEVICE_MAX_CONNECTIONS": "1", "NVTE_FWD_LAYERNORM_SM_MARGIN": "16",
                "NVTE_BWD_LAYERNORM_SM_MARGIN": "16", "TORCH_NCCL_AVOID_RECORD_STREAMS": "1"}  # fmt: skip

    def to_dict(self) -> dict:
        return {
            "model": asdict(self.model), "cluster": asdict(self.cluster), "train": asdict(self.train),
            "layout": asdict(self.best.layout), "memory_gib": self.best.memory_gib, "time_s": self.best.time_s,
            "colocated": self.colocated, "rollout": asdict(self.rollout),
            "megatron_args": self.megatron_args(), "sglang_args": self.sglang_args(), "env": self.env(),
        }  # fmt: skip

    def report(self, top: int = 8) -> str:
        m, c, t, b = self.model, self.cluster, self.train, self.best
        tok_s = t.global_tokens / b.step_s / c.world
        useful = (2 if t.lora_rank else 3) * (2 * m.active_params + m.attention_flops(t.seq_len))
        mfu = tok_s * useful / (c.peak_tflops * 1e12)
        lines = [
            f"{m.name}: {m.total_params / 1e9:.1f}B params ({m.active_params / 1e9:.1f}B active), "
            f"{c.nodes} node(s) x {c.gpus_per_node} GPUs, seq {t.seq_len}, {t.global_tokens} tokens/step, "
            f"{'LoRA r=' + str(t.lora_rank) if t.lora_rank else 'full fine-tune'}, {t.precision}",
            f"chosen: {b.layout}  peak {b.peak_gib:.1f} GiB/GPU, step ~{b.step_s:.1f}s, "
            f"~{tok_s:,.0f} tok/s/GPU (~{mfu:.0%} MFU), {b.microbatches} micro-batches",
            "memory GiB: " + ", ".join(f"{k} {v:.1f}" for k, v in b.memory_gib.items()),
            "time s:     " + ", ".join(f"{k} {v:.2f}" for k, v in b.time_s.items()),
            f"rollout: {'colocated on the same GPUs' if self.colocated else 'separate GPUs'}; "
            f"{self.rollout.engines} SGLang engine(s) x TP{self.rollout.engine_tp}, "
            f"~{self.rollout.max_concurrent} concurrent {self.rollout.context}-token sequences per engine",
            "",
            f"{'layout':<36}{'fits':<6}{'GiB':>7}{'step s':>9}",
        ]
        for e in self.ranked[:top]:
            lines.append(f"{e.layout!s:<36}{'yes' if e.fits else 'NO':<6}{e.peak_gib:>7.1f}{e.step_s:>9.2f}")
        if self.remedies:
            lines.insert(1, "applied to fit: " + ", ".join(self.remedies))
        if b.notes:
            lines += ["", "notes: " + "; ".join(b.notes)]
        return "\n".join(lines)


# Memory remedies, cheapest first in step time; each is cumulative with the ones before it.
REMEDIES: list[tuple[str, dict]] = [
    ("full recompute", {"recompute": "full"}),
    ("bf16 Adam moments + bf16 grad reduce", {"bf16_moments": True, "grad_fp32": False}),
    ("optimizer CPU offload", {"optimizer_offload": True}),
]


def _rank(m: ModelSpec, c: ClusterSpec, t: TrainSpec) -> list[Estimate]:
    layouts = enumerate_layouts(m, c, t)
    if not layouts:
        raise ValueError(f"no valid layout for {m.name} on {c.world} GPUs (check seq_len / global_tokens)")
    ests = [estimate(lay, m, c, t) for lay in layouts]
    return sorted(ests, key=lambda e: (not e.fits, e.step_s if e.fits else e.peak_gib))


def plan(m: ModelSpec, c: ClusterSpec, t: TrainSpec, rollout_gpus: int | None = None, context: int = 16384,
         min_concurrent: int = 32, auto_fit: bool = True) -> Plan:  # fmt: skip
    """Best training layout plus rollout engines. `rollout_gpus=None` colocates rollouts on all GPUs
    (training and generation take turns; weights move by CUDA IPC). Otherwise the cluster spec describes
    the training GPUs only and rollouts get `rollout_gpus` separate GPUs (weights move over NCCL/RDMA).
    With `auto_fit`, memory remedies are applied in order until some layout fits."""
    ranked, applied = _rank(m, c, t), []
    for label, change in REMEDIES if auto_fit else []:
        if ranked[0].fits:
            break
        if all(getattr(t, k) == v for k, v in change.items()):
            continue
        t = replace(t, **change)
        applied.append(label)
        ranked = _rank(m, c, t)
    if not ranked[0].fits:
        smallest = min(ranked, key=lambda e: e.peak_gib)
        raise ValueError(f"{m.name} does not fit on {c.world} GPUs even with {', '.join(applied) or 'no remedies'}: "
                         f"best is {smallest.layout} at {smallest.peak_gib:.0f} GiB/GPU; use LoRA, a shorter "
                         f"--tokens-per-microbatch, or more nodes")  # fmt: skip
    colocated = rollout_gpus is None
    r = plan_rollout(m, c, c.world if colocated else rollout_gpus, context, min_concurrent)
    return Plan(m, c, t, ranked[0], ranked, r, colocated, applied)
