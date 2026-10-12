"""Scheduling: Kueue (default) or Volcano, never both on one workload.

Kueue v0.20 (`kueue.x-k8s.io/v1beta2`) owns quota, priority, preemption and topology-aware placement:
  - one cohort with three ClusterQueues: `serving` (never borrows; lends idle GPUs and reclaims them by
    preemption), `rl-train` (borrows idle serving GPUs) and `eval` (no quota, borrows only);
  - all-or-nothing admission for the RayJob/Deployment, plus waitForPodsReady for gang at runtime;
  - TAS keeps a training job inside one block/rack (`podset-required-topology`).
Kueue and Volcano have no official integration (Kueue issues #1204/#1985, closed). The `volcano` profile is
for clusters that want Volcano's gang scheduling and HyperNode network topology instead; it renders
Volcano Queues, HyperNodes and the scheduler plugin config, and the workloads use `schedulerName: volcano`.
"""

from __future__ import annotations

from swarmrl.deploy.profile import DeployProfile

KUEUE = "kueue.x-k8s.io/v1beta2"
COHORT = "gpu"
FLAVOR = "h100"
TOPOLOGY = "gpu-topology"
PRIORITIES = {"serving": 1000, "rl-train": 500, "eval": 100}
QUEUES = {"inference": "serving", "rl": "rl-train", "eval": "eval"}  # LocalQueue -> ClusterQueue


def _cq(name: str, gpus: int, borrow: int | None, lend: int | None, preempt: dict) -> dict:
    gpu = {"name": "nvidia.com/gpu", "nominalQuota": gpus}
    if borrow is not None:
        gpu["borrowingLimit"] = borrow
    if lend is not None:
        gpu["lendingLimit"] = lend
    return {
        "apiVersion": KUEUE, "kind": "ClusterQueue", "metadata": {"name": name},
        "spec": {
            "cohortName": COHORT, "namespaceSelector": {}, "queueingStrategy": "BestEffortFIFO",
            "preemption": preempt,
            "resourceGroups": [{"coveredResources": ["nvidia.com/gpu"],
                                "flavors": [{"name": FLAVOR, "resources": [gpu]}]}],
        },
    }  # fmt: skip


def kueue_objects(p: DeployProfile) -> list[dict]:
    total = p.train_gpus + p.inference_gpus
    out = [
        {"apiVersion": KUEUE, "kind": "Topology", "metadata": {"name": TOPOLOGY},
         "spec": {"levels": [{"nodeLabel": lvl} for lvl in p.topology_levels]}},
        {"apiVersion": KUEUE, "kind": "ResourceFlavor", "metadata": {"name": FLAVOR},
         "spec": {"nodeLabels": {"nvidia.com/gpu.product": p.gpu_product}, "topologyName": TOPOLOGY,
                  "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]}},
        # serving never borrows and may lend idle GPUs; it takes them back by preempting the borrower
        _cq("serving", p.inference_gpus, borrow=0, lend=None,
            preempt={"reclaimWithinCohort": "Any", "withinClusterQueue": "LowerPriority"}),
        _cq("rl-train", p.train_gpus, borrow=p.inference_gpus, lend=None,
            preempt={"reclaimWithinCohort": "Any", "withinClusterQueue": "LowerPriority",
                     "borrowWithinCohort": {"policy": "LowerPriority",
                                            "maxPriorityThreshold": PRIORITIES["rl-train"]}}),
        _cq("eval", 0, borrow=total, lend=None, preempt={"withinClusterQueue": "LowerPriority"}),
    ]  # fmt: skip
    for name, value in PRIORITIES.items():
        out.append({"apiVersion": KUEUE, "kind": "WorkloadPriorityClass", "metadata": {"name": name},
                    "value": value, "description": f"swarmrl {name}"})  # fmt: skip
    for ns, (lq, cq) in {
        p.inference_namespace: ("inference", QUEUES["inference"]),
        p.training_namespace: ("rl", QUEUES["rl"]),
    }.items():
        out.append({"apiVersion": KUEUE, "kind": "LocalQueue", "metadata": {"name": lq, "namespace": ns},
                    "spec": {"clusterQueue": cq}})  # fmt: skip
    out.append({"apiVersion": KUEUE, "kind": "LocalQueue",
                "metadata": {"name": "eval", "namespace": p.training_namespace},
                "spec": {"clusterQueue": QUEUES["eval"]}})  # fmt: skip
    return out


def kueue_configuration() -> dict:
    """Fragment for Kueue's manager Configuration (merge into the kueue-manager-config ConfigMap)."""
    return {
        "apiVersion": "config.kueue.x-k8s.io/v1beta2",
        "kind": "Configuration",
        # gang semantics at runtime: a job whose pods are not all Ready in time is evicted and requeued
        "waitForPodsReady": {"timeout": "15m", "recoveryTimeout": "5m", "blockAdmission": False,
                             "requeuingStrategy": {"timestamp": "Eviction", "backoffLimitCount": 5,
                                                   "backoffBaseSeconds": 60}},
        # SR-IOV VF resources are node-local plumbing, not quota: without this they block admission
        "resources": {"excludeResourcePrefixes": ["openshift.io/"]},
        "integrations": {"frameworks": ["batch/job", "ray.io/rayjob", "ray.io/raycluster", "jobset.x-k8s.io/jobset",
                                        "pod", "deployment", "leaderworkerset.x-k8s.io/leaderworkerset"]},
    }  # fmt: skip


def volcano_objects(p: DeployProfile) -> list[dict]:
    out = [
        {"apiVersion": "scheduling.volcano.sh/v1beta1", "kind": "Queue", "metadata": {"name": "serving"},
         "spec": {"weight": 4, "reclaimable": False, "priority": PRIORITIES["serving"],
                  "guarantee": {"resource": {"nvidia.com/gpu": str(p.inference_gpus)}}}},
        {"apiVersion": "scheduling.volcano.sh/v1beta1", "kind": "Queue", "metadata": {"name": "rl-train"},
         "spec": {"weight": 2, "reclaimable": True, "priority": PRIORITIES["rl-train"],
                  "deserved": {"nvidia.com/gpu": str(p.train_gpus)}}},
    ]  # fmt: skip
    rack_label = p.topology_levels[-2] if len(p.topology_levels) >= 2 else "kubernetes.io/hostname"
    for rack in p.racks:
        out.append({
            "apiVersion": "topology.volcano.sh/v1alpha1", "kind": "HyperNode", "metadata": {"name": f"rack-{rack}"},
            "spec": {"tier": 1, "tierName": "rack",
                     "members": [{"type": "Node", "selector": {"labelMatch": {"matchLabels": {rack_label: rack}}}}]},
        })  # fmt: skip
    if p.racks:
        out.append({
            "apiVersion": "topology.volcano.sh/v1alpha1", "kind": "HyperNode", "metadata": {"name": "cluster"},
            "spec": {"tier": 2, "tierName": "block", "members": [
                {"type": "HyperNode", "selector": {"exactMatch": {"name": f"rack-{r}"}}} for r in p.racks]},
        })  # fmt: skip
    return out


VOLCANO_SCHEDULER_CONF = """actions: "enqueue, allocate, backfill, preempt, reclaim"
tiers:
- plugins:
  - name: priority
  - name: gang
    enablePreemptable: false
  - name: conformance
- plugins:
  - name: drf
  - name: predicates
  - name: proportion
  - name: nodeorder
  - name: binpack
  - name: network-topology-aware
"""


def volcano_scheduler_config(namespace: str = "volcano-system") -> dict:
    return {"apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": "volcano-scheduler-configmap", "namespace": namespace},
            "data": {"volcano-scheduler.conf": VOLCANO_SCHEDULER_CONF}}  # fmt: skip


def workload_meta(p: DeployProfile, local_queue: str, priority: str, topology: str | None = None) -> dict:
    """Labels/annotations/scheduler for a workload under the profile's scheduler.

    Returns {"labels", "pod_annotations", "pod_labels", "scheduler_name"}."""
    if p.scheduler == "kueue":
        ann = {"kueue.x-k8s.io/podset-required-topology": topology} if topology else {}
        return {"labels": {"kueue.x-k8s.io/queue-name": local_queue, "kueue.x-k8s.io/priority-class": priority},
                "pod_annotations": ann, "pod_labels": {}, "scheduler_name": None}  # fmt: skip
    queue = QUEUES.get(local_queue, local_queue)
    return {"labels": {"ray.io/scheduler-name": "volcano", "volcano.sh/queue-name": queue},
            "pod_annotations": {"scheduling.volcano.sh/queue-name": queue}, "pod_labels": {},
            "scheduler_name": "volcano"}  # fmt: skip
