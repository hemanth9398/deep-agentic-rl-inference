"""Deployment profile: everything about the cluster that the manifests need, in one file.

Load from YAML/JSON with `DeployProfile.load(path)`; `deploy/profiles/h100.yaml` is a worked example.
Values that only the cluster can tell you (PF names, node labels) are marked in that file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class Rail:
    """One RDMA NIC per GPU ("rail"). `pf` is the physical function's netdev name on every GPU node
    (see `kubectl get sriovnetworknodestates -o yaml`)."""

    name: str
    pf: str


def _default_rails() -> list[Rail]:
    return [Rail(f"rail{i}", f"ens{i + 1}f0np0") for i in range(8)]


@dataclass(frozen=True)
class DeployProfile:
    name: str = "h100"
    # --- nodes -------------------------------------------------------------------------------
    gpu_product: str = "NVIDIA-H100-80GB-HBM3"  # nvidia.com/gpu.product (GPU Feature Discovery)
    gpus_per_node: int = 8
    cpu_per_node: str = "160"  # what one full-node pod requests
    memory_per_node: str = "1600Gi"
    shm_size: str = "256Gi"
    # --- RDMA fabric (SR-IOV VFs on Multus, Cilium stays primary on eth0) -------------------------
    link_type: str = "eth"  # "eth" (RoCEv2) | "ib"
    rails: list[Rail] = field(default_factory=_default_rails)
    nic_device_id: str = "1021"  # ConnectX-7 PF
    vfs_per_pf: int = 8
    mtu: int = 9000
    ipam_prefix: str = "10.200"  # rail i: <prefix>.<i*16>.0/20 via whereabouts
    rdma_mode: str = "exclusive"  # VF RDMA devices move into the pod's netns
    roce_gid_index: int | None = 3  # RoCEv2 IPv4 GID; None = NCCL auto-select
    roce_traffic_class: int | None = None  # NCCL_IB_TC, must match switch QoS (e.g. 106)
    # --- scheduling ---------------------------------------------------------------------------
    scheduler: str = "kueue"  # "kueue" (quota + TAS) | "volcano" (gang + HyperNode)
    topology_levels: list[str] = field(default_factory=lambda: [
        "network.topology/block", "network.topology/rack", "kubernetes.io/hostname"])  # fmt: skip
    racks: list[str] = field(default_factory=list)  # Volcano HyperNodes: values of the rack label
    train_gpus: int = 8  # nominal quota
    inference_gpus: int = 8
    # --- namespaces ---------------------------------------------------------------------------
    sriov_namespace: str = "sriov-network-operator"
    training_namespace: str = "rl"
    inference_namespace: str = "inference"
    sandbox_namespace: str = "swarmrl-sandbox"
    # --- images (pin by digest in production) -----------------------------------------------------
    swarmrl_image: str = "ghcr.io/your-org/swarmrl-slime:0.1.0"  # deploy/images/Dockerfile.slime
    epp_tag: str = "v0.11.0"
    # --- model and data -------------------------------------------------------------------------
    model: str = "Qwen/Qwen3-30B-A3B"  # HF id served and trained
    planner_model: str = "qwen3-30b-a3b"  # planner preset or config.json path
    models_pvc: str = "models"  # RWX volume: HF + torch_dist checkpoints, tasks
    hf_checkpoint: str = "/models/hf/Qwen3-30B-A3B"
    torch_dist: str = "/models/torch_dist/Qwen3-30B-A3B"
    save_dir: str = "/models/rl/Qwen3-30B-A3B"
    tasks_dir: str = "/models/tasks"
    prompt_data: str = "/models/tasks/slime-train.jsonl"
    hf_token_secret: str = "hf-token"
    # --- RL run -------------------------------------------------------------------------------------
    stage: int = 1
    train_nodes: int = 1
    rollout: str = "colocated"  # "colocated" | "pool" (llm-d pool, NCCL/RDMA weight sync)
    pool_gpus: int = 8
    seq_len: int = 32768
    rollout_batch_size: int = 32
    num_rollout: int = 200
    sandbox_in_flight: int = 64

    def __post_init__(self):
        if self.link_type not in ("eth", "ib"):
            raise ValueError("link_type must be eth or ib")
        if self.scheduler not in ("kueue", "volcano"):
            raise ValueError("scheduler must be kueue or volcano")
        if self.rollout not in ("colocated", "pool"):
            raise ValueError("rollout must be colocated or pool")
        if len(self.rails) > 16:
            raise ValueError("at most 16 rails (IPAM layout)")

    @classmethod
    def from_dict(cls, d: dict) -> DeployProfile:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown profile keys: {sorted(unknown)}")
        d = dict(d)
        if "rails" in d:
            d["rails"] = [r if isinstance(r, Rail) else Rail(**r) for r in d["rails"]]
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> DeployProfile:
        text = Path(path).read_text()
        if str(path).endswith((".yaml", ".yml")):
            import yaml

            return cls.from_dict(yaml.safe_load(text) or {})
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict:
        return asdict(self)
