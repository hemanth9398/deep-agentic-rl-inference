"""RDMA fabric: SR-IOV VFs per rail, attached to pods through Multus next to Cilium's eth0.

Cluster prerequisites (installed once, not rendered here):
  - Cilium with `cni.exclusive=false` (cilium-values.yaml), so Multus can add secondary interfaces.
  - Multus thick plugin (ghcr.io/k8snetworkplumbingwg/multus-cni:v4.3.1-thick) and whereabouts IPAM.
  - SR-IOV Network Operator v1.6 (oci://ghcr.io/k8snetworkplumbingwg/sriov-network-operator-chart).
  - GPU Operator with GPUDirect RDMA over DMA-BUF (open kernel module), rdma-core on the hosts.

Each rail gets a SriovNetworkNodePolicy (VFs with RDMA on that PF) and, per workload namespace, a
SriovNetwork (eth/RoCEv2) or SriovIBNetwork (InfiniBand) that renders the NetworkAttachmentDefinition.
In exclusive RDMA mode a pod sees only its own VFs' RDMA devices, so NCCL needs no HCA list.
"""

from __future__ import annotations

import json

from swarmrl.deploy.profile import DeployProfile

SRIOV_API = "sriovnetwork.openshift.io/v1"
RESOURCE_PREFIX = "openshift.io"  # SR-IOV operator default resourcePrefix
GPU_NODE = {"nvidia.com/gpu.present": "true"}


def _ipam(p: DeployProfile, i: int) -> str:
    a, b = p.ipam_prefix.split(".")
    return json.dumps({"type": "whereabouts", "range": f"{a}.{b}.{i * 16}.0/20"})


def sriov_objects(p: DeployProfile, workload_namespaces: list[str]) -> list[dict]:
    ns = p.sriov_namespace
    out: list[dict] = [
        {"apiVersion": SRIOV_API, "kind": "SriovOperatorConfig", "metadata": {"name": "default", "namespace": ns},
         "spec": {"enableInjector": True, "enableOperatorWebhook": True, "configDaemonNodeSelector": GPU_NODE,
                  "logLevel": 2}},
        {"apiVersion": SRIOV_API, "kind": "SriovNetworkPoolConfig", "metadata": {"name": "gpu-rdma", "namespace": ns},
         "spec": {"nodeSelector": {"matchLabels": GPU_NODE}, "maxUnavailable": 1, "rdmaMode": p.rdma_mode}},
    ]  # fmt: skip
    for i, r in enumerate(p.rails):
        out.append({
            "apiVersion": SRIOV_API, "kind": "SriovNetworkNodePolicy",
            "metadata": {"name": r.name, "namespace": ns},
            "spec": {
                "resourceName": r.name, "nodeSelector": GPU_NODE, "priority": 90, "numVfs": p.vfs_per_pf,
                "mtu": p.mtu, "deviceType": "netdevice", "isRdma": True, "linkType": p.link_type,
                "nicSelector": {"vendor": "15b3", "deviceID": p.nic_device_id, "pfNames": [r.pf]},
            },
        })  # fmt: skip
        for wns in workload_namespaces:
            meta = {"name": f"{r.name}-{wns}", "namespace": ns}
            if p.link_type == "ib":
                out.append({"apiVersion": SRIOV_API, "kind": "SriovIBNetwork", "metadata": meta, "spec": {
                    "resourceName": r.name, "networkNamespace": wns, "linkState": "enable", "ipam": _ipam(p, i),
                    "capabilities": json.dumps({"ips": True})}})  # fmt: skip
            else:
                out.append({"apiVersion": SRIOV_API, "kind": "SriovNetwork", "metadata": meta, "spec": {
                    "resourceName": r.name, "networkNamespace": wns, "ipam": _ipam(p, i), "spoofChk": "off",
                    "trust": "on", "linkState": "auto", "capabilities": json.dumps({"ips": True}),
                    "metaPlugins": json.dumps({"type": "rdma"})}})  # fmt: skip
    return out


def cilium_values() -> dict:
    """Cilium Helm values for coexisting with Multus (merge into your Cilium release)."""
    return {"cni": {"exclusive": False}}


def nad_name(p: DeployProfile, rail_index: int, namespace: str) -> str:
    return f"{p.rails[rail_index].name}-{namespace}"


def rdma_pod_patch(p: DeployProfile, namespace: str, rails: int) -> dict:
    """What a pod needs to use `rails` RDMA rails: Multus annotation, VF resources, NCCL env, IPC_LOCK, /dev/shm.

    Returns {"annotations", "resources", "env", "security_context", "volumes", "volume_mounts"} for the
    workload builders to merge."""
    rails = max(0, min(rails, len(p.rails)))
    nets = [{"name": nad_name(p, i, namespace), "namespace": namespace, "interface": f"net{i + 1}"}
            for i in range(rails)]  # fmt: skip
    env = {
        "NCCL_SOCKET_IFNAME": "eth0",  # bootstrap over Cilium; data over the VFs' verbs devices
        "NCCL_CROSS_NIC": "0",  # rail-optimized fabric: GPU i talks to GPU i on other nodes over rail i
        "NCCL_IB_QPS_PER_CONNECTION": "2",
        "NCCL_NVLS_ENABLE": "1",  # NVLink SHARP on HGX H100 (NVSwitch)
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    if p.link_type == "eth":
        env["NCCL_IB_ROCE_VERSION_NUM"] = "2"
        if p.roce_gid_index is not None:
            env["NCCL_IB_GID_INDEX"] = str(p.roce_gid_index)
        if p.roce_traffic_class is not None:
            env["NCCL_IB_TC"] = str(p.roce_traffic_class)
    return {
        "annotations": {"k8s.v1.cni.cncf.io/networks": json.dumps(nets)} if rails else {},
        "resources": {f"{RESOURCE_PREFIX}/{p.rails[i].name}": "1" for i in range(rails)},
        "env": env,
        "security_context": {"capabilities": {"add": ["IPC_LOCK"]}},
        "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": p.shm_size}}],
        "volume_mounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
    }
