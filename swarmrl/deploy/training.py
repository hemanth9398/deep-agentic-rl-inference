"""The RL job: slime (Megatron + SGLang) under KubeRay, swarm rollouts in k8s sandbox pods.

One RayJob per stage. GPU workers are full nodes (8 GPUs, every RDMA rail); the Ray head runs on CPU.
Colocated: training and SGLang take turns on the same GPUs and weights move by CUDA IPC.
Pool: slime trains on its own nodes and NCCL-broadcasts weights over RDMA to the llm-d rollout pool, whose
addresses are resolved when the job starts (`swarmrl pool-endpoints`); rollouts go through the gateway, where
the EPP routes each request to the engine that already holds its prefix in KV cache.
"""

from __future__ import annotations

import json
import shlex

from swarmrl.deploy.fabric import rdma_pod_patch
from swarmrl.deploy.profile import DeployProfile
from swarmrl.deploy.scheduling import workload_meta
from swarmrl.integrations.slime import SlimeSwarmSettings, slime_args
from swarmrl.parallel.planner import Plan
from swarmrl.rl.schedule import stage as get_stage
from swarmrl.sandbox.kubernetes import namespace_manifest, network_policy_manifest

SA = "swarmrl-rollout"
CONFIG_PATH = "/etc/swarmrl/slime.json"
POOL_LABEL = {"app.kubernetes.io/name": "rollout-pool"}
POOL_PORT = 8000
ENGINES_PLACEHOLDER = "__SWARMRL_ENGINES__"


def gateway_url(p: DeployProfile) -> str:
    return f"http://rollout-gateway.{p.inference_namespace}.svc.cluster.local"


def sandbox_objects(p: DeployProfile) -> list[dict]:
    return [namespace_manifest(p.sandbox_namespace), network_policy_manifest(p.sandbox_namespace)]


def rbac_objects(p: DeployProfile) -> list[dict]:
    """The rollout process creates, execs into and deletes sandbox pods; in pool mode it also lists pool pods."""
    ns, sb = p.training_namespace, p.sandbox_namespace
    out = [
        {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": SA, "namespace": ns}},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "Role", "metadata": {"name": SA, "namespace": sb},
         "rules": [
             {"apiGroups": [""], "resources": ["pods"],
              "verbs": ["create", "get", "list", "watch", "delete", "patch"]},
             {"apiGroups": [""], "resources": ["pods/exec"], "verbs": ["create", "get"]},
             {"apiGroups": [""], "resources": ["events", "serviceaccounts"], "verbs": ["get", "list"]},
             {"apiGroups": [""], "resources": ["namespaces"], "verbs": ["get"]},
             {"apiGroups": ["networking.k8s.io"], "resources": ["networkpolicies"],
              "verbs": ["get", "create", "patch"]},
         ]},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding", "metadata": {"name": SA, "namespace": sb},
         "subjects": [{"kind": "ServiceAccount", "name": SA, "namespace": ns}],
         "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": SA}},
    ]  # fmt: skip
    if p.rollout == "pool":
        inf = p.inference_namespace
        out += [
            {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "Role",
             "metadata": {"name": f"{SA}-pool", "namespace": inf},
             "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]}]},
            {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding",
             "metadata": {"name": f"{SA}-pool", "namespace": inf},
             "subjects": [{"kind": "ServiceAccount", "name": SA, "namespace": ns}],
             "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": f"{SA}-pool"}},
        ]  # fmt: skip
    return out


def slime_settings(p: DeployProfile) -> SlimeSwarmSettings:
    pool = p.rollout == "pool"
    return SlimeSwarmSettings(
        tasks_dir=p.tasks_dir, sandbox="k8s", namespace=p.sandbox_namespace, traces=f"{p.save_dir}/traces",
        tokenizer=p.hf_checkpoint, stage=p.stage, max_in_flight=p.sandbox_in_flight,
        # pool: through the gateway, where the EPP routes on exact KV-cache contents; colocated: slime's router
        policy_url=gateway_url(p) if pool else None,
        routing_header=None if pool else "X-SMG-Routing-Key",
    )  # fmt: skip


def config_map(p: DeployProfile) -> dict:
    s = slime_settings(p)
    from dataclasses import asdict

    return {"apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": "swarmrl-slime", "namespace": p.training_namespace},
            "data": {"slime.json": json.dumps(asdict(s), indent=1)}}  # fmt: skip


def entrypoint(p: DeployProfile, plan: Plan) -> str:
    st = get_stage(p.stage)
    pool = p.rollout == "pool"
    args = slime_args(
        plan, st, hf_checkpoint=p.hf_checkpoint, load=p.torch_dist, save=p.save_dir, prompt_data=p.prompt_data,
        rollout_batch_size=p.rollout_batch_size, num_rollout=p.num_rollout,
        external_engines=[ENGINES_PLACEHOLDER] if pool else (),
    )  # fmt: skip
    cmd = "python3 /root/slime/train.py " + " ".join(shlex.quote(a) for a in args)
    if not pool:
        return cmd
    discover = (f"ENGINES=$(swarmrl pool-endpoints --namespace {p.inference_namespace} "
                f"--selector {','.join(f'{k}={v}' for k, v in POOL_LABEL.items())} --port {POOL_PORT})")  # fmt: skip
    return f"{discover} && " + cmd.replace(ENGINES_PLACEHOLDER, "$ENGINES")


def _container(p: DeployProfile, name: str, resources: dict, env: dict, mounts: list, sc: dict | None) -> dict:
    c = {
        "name": name, "image": p.swarmrl_image, "imagePullPolicy": "IfNotPresent",
        "resources": resources,
        "env": [{"name": k, "value": v} for k, v in env.items()]
        + [{"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": p.hf_token_secret, "key": "HF_TOKEN",
                                                                "optional": True}}}],
        "volumeMounts": mounts,
    }  # fmt: skip
    if sc:
        c["securityContext"] = sc
    return c


def rayjob(p: DeployProfile, plan: Plan, name: str = "swarmrl-rl") -> dict:
    """RayJob running slime's train.py; Kueue (or Volcano) admits the whole Ray cluster at once."""
    st = get_stage(p.stage)
    # a multi-node job stays inside one rack: its DP/EP/CP groups cross nodes and need the shortest RDMA path
    rack = p.topology_levels[-2] if p.train_nodes > 1 and len(p.topology_levels) > 1 else None
    meta = workload_meta(p, "rl", "rl-train", topology=rack)
    rdma = rdma_pod_patch(p, p.training_namespace, len(p.rails))
    common_mounts = [{"name": "models", "mountPath": "/models"},
                     {"name": "swarmrl-config", "mountPath": "/etc/swarmrl", "readOnly": True}]  # fmt: skip
    common_vols = [{"name": "models", "persistentVolumeClaim": {"claimName": p.models_pvc}},
                   {"name": "swarmrl-config", "configMap": {"name": "swarmrl-slime"}}]  # fmt: skip
    env = {"SWARMRL_SLIME_CONFIG": CONFIG_PATH, "SWARMRL_LOG_JSON": "1"}

    def pod(container: dict, gpu: bool) -> dict:
        spec = {"serviceAccountName": SA, "containers": [container],
                "volumes": common_vols + (rdma["volumes"] if gpu else [])}  # fmt: skip
        if meta["scheduler_name"]:
            spec["schedulerName"] = meta["scheduler_name"]
        if gpu:
            spec["tolerations"] = [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
            spec["nodeSelector"] = {"nvidia.com/gpu.product": p.gpu_product}
        ann = {**(rdma["annotations"] if gpu else {}), **(meta["pod_annotations"] if gpu else {})}
        return {"metadata": {"annotations": ann}, "spec": spec} if ann else {"spec": spec}

    head = _container(p, "ray-head", {"requests": {"cpu": "8", "memory": "32Gi"}, "limits": {"memory": "32Gi"}},
                      env, common_mounts, None)  # fmt: skip
    head["ports"] = [{"containerPort": 6379, "name": "gcs"}, {"containerPort": 8265, "name": "dashboard"},
                     {"containerPort": 10001, "name": "client"}]  # fmt: skip
    gpu_res = {"nvidia.com/gpu": str(p.gpus_per_node), "cpu": p.cpu_per_node, "memory": p.memory_per_node,
               **rdma["resources"]}  # fmt: skip
    worker = _container(p, "ray-worker", {"requests": gpu_res, "limits": gpu_res}, {**env, **rdma["env"]},
                        common_mounts + rdma["volume_mounts"], rdma["security_context"])  # fmt: skip
    runtime_env = {"env_vars": {"PYTHONPATH": "/root/Megatron-LM", **plan.env(), **env}}
    return {
        "apiVersion": "ray.io/v1", "kind": "RayJob",
        "metadata": {"name": name, "namespace": p.training_namespace, "labels": meta["labels"],
                     "annotations": {"swarmrl.dev/stage": st.name, "swarmrl.dev/layout": str(plan.best.layout)}},
        "spec": {
            "entrypoint": entrypoint(p, plan),
            "runtimeEnvYAML": json.dumps(runtime_env),
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 3600,
            "submissionMode": "K8sJobMode",
            "rayClusterSpec": {
                "headGroupSpec": {"rayStartParams": {"dashboard-host": "0.0.0.0", "num-gpus": "0"},
                                  "template": pod(head, gpu=False)},
                "workerGroupSpecs": [{
                    "groupName": "gpu", "replicas": p.train_nodes, "minReplicas": p.train_nodes,
                    "maxReplicas": p.train_nodes, "rayStartParams": {"num-gpus": str(p.gpus_per_node)},
                    "template": pod(worker, gpu=True),
                }],
            },
        },
    }  # fmt: skip
