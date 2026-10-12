"""Inference: the llm-d rollout pool for RL, the gateway, and KServe serving of trained checkpoints.

Rollout pool (RL, `rollout: pool`): SGLang engines (the same SGLang as the trainer, so NCCL weight updates
match) publishing KV-cache events over ZMQ. llm-d's router chart (v0.11, `llm-d-router-gateway`) deploys the
EPP and the InferencePool; its precise-prefix producer indexes those events, so each multi-turn request goes
to the engine that already holds the longest prefix of its context. Our client sends SGLang `/generate` with
`input_ids`, which the EPP's `sglanghttp-parser` reads directly: no tokenizer service in the loop.
Gateway: Envoy Gateway with the Envoy AI Gateway extension (plain Envoy Gateway does not implement
InferencePool). Serving: KServe `LLMInferenceService` (v1alpha2, vLLM) with the same precise-prefix EPP.
"""

from __future__ import annotations

import json

from swarmrl.deploy.fabric import rdma_pod_patch
from swarmrl.deploy.profile import DeployProfile
from swarmrl.deploy.scheduling import workload_meta
from swarmrl.deploy.training import POOL_LABEL, POOL_PORT
from swarmrl.parallel.planner import Plan

KV_EVENTS_PORT = 5556
BLOCK_TOKENS = 64  # SGLang --page-size / vLLM --block-size; the EPP rehashes, but keep pages aligned
GATEWAY_CLASS = "envoy-ai-gateway"


def epp_config(engine: str = "sglang", block_tokens: int = BLOCK_TOKENS, kv_port: int = KV_EVENTS_PORT,
               speculative: bool = True) -> dict:  # fmt: skip
    """EndpointPickerConfig: precise prefix-cache routing from engine KV events, plus load balance."""
    parser = "sglanghttp-parser" if engine == "sglang" else "vllmhttp-parser"
    return {
        "apiVersion": "llm-d.ai/v1alpha1",
        "kind": "EndpointPickerConfig",
        "plugins": [
            {"type": "single-profile-handler"},
            {"type": parser},
            {"type": "openai-parser"},
            {"type": "token-producer"},  # estimate backend: pre-tokenized input_ids pass through as real tokens
            {"type": "endpoint-notification-source"},
            {"type": "precise-prefix-cache-producer", "parameters": {
                "tokenProcessorConfig": {"blockSizeTokens": block_tokens},
                "speculativeIndexing": speculative,  # count a just-routed prefix as cached before its events arrive
                "indexerConfig": {"kvBlockIndexConfig": {"enableMetrics": True}},
                "kvEventsConfig": {"engineType": engine, "topicFilter": "kv@", "concurrency": 8, "discoverPods": True,
                                   "podDiscoveryConfig": {"socketPort": kv_port}},
            }},
            {"type": "prefix-cache-scorer",
             "parameters": {"prefixMatchInfoProducerName": "precise-prefix-cache-producer"}},
            {"type": "kv-cache-utilization-scorer"},
            {"type": "queue-scorer"},
            {"type": "max-score-picker"},
        ],
        "dataLayer": {"sources": [{"pluginRef": "endpoint-notification-source",
                                   "extractors": [{"pluginRef": "precise-prefix-cache-producer"}]}]},
        "schedulingProfiles": [{"name": "default", "plugins": [
            {"pluginRef": "prefix-cache-scorer", "weight": 3.0},
            {"pluginRef": "kv-cache-utilization-scorer", "weight": 1.0},
            {"pluginRef": "queue-scorer", "weight": 1.0},
            {"pluginRef": "max-score-picker"},
        ]}],
        "requestHandler": {"parsers": [{"pluginRef": parser}, {"pluginRef": "openai-parser"}]},
    }  # fmt: skip


def router_values(p: DeployProfile, release: str = "rollout") -> dict:
    """Helm values for oci://ghcr.io/llm-d/charts/llm-d-router-gateway (creates the EPP, the InferencePool
    named after the release, and the HTTPRoute to our Gateway)."""
    import yaml

    return {
        "router": {
            "epp": {
                "replicas": 1,  # in-flight load is per EPP process; keep one active replica with this config
                "image": {"registry": "ghcr.io/llm-d", "repository": "llm-d-router-endpoint-picker",
                          "tag": p.epp_tag, "pullPolicy": "IfNotPresent"},
                "pluginsConfigFile": "swarmrl-rollout.yaml",
                "pluginsCustomConfig": {"swarmrl-rollout.yaml": yaml.safe_dump(epp_config("sglang"), sort_keys=False)},
                "resources": {"requests": {"cpu": "4", "memory": "8Gi"}, "limits": {"memory": "16Gi"}},
            },
            "inferencePool": {"create": True, "failureMode": "FailClose"},
            "modelServers": {"type": "sglang", "protocol": "http", "targetPorts": [{"number": POOL_PORT}],
                             "matchLabels": dict(POOL_LABEL)},
        },
        "httpRoute": {"create": True, "inferenceGatewayName": "rollout-gateway", "requestTimeout": "900s"},
    }  # fmt: skip


def envoy_gateway_values() -> dict:
    """Envoy Gateway (oci://docker.io/envoyproxy/gateway-helm) + Envoy AI Gateway hooks + InferencePool backend."""
    return {"config": {"envoyGateway": {
        "gateway": {"controllerName": "gateway.envoyproxy.io/gatewayclass-controller"},
        "provider": {"type": "Kubernetes", "kubernetes": {"deploy": {"type": "GatewayNamespace"}}},
        "extensionApis": {"enableEnvoyPatchPolicy": True, "enableBackend": True},
        "extensionManager": {
            "backendResources": [{"group": "inference.networking.k8s.io", "kind": "InferencePool", "version": "v1"}],
            "hooks": {"xdsTranslator": {
                "translation": {k: {"includeAll": True} for k in ("listener", "route", "cluster", "secret")},
                "post": ["Translation", "Cluster", "Route"]}},
            "service": {"fqdn": {"hostname": "ai-gateway-controller.envoy-ai-gateway-system.svc.cluster.local",
                                 "port": 1063}},
        },
    }}}  # fmt: skip


def gateway_objects(p: DeployProfile) -> list[dict]:
    return [
        {"apiVersion": "gateway.networking.k8s.io/v1", "kind": "GatewayClass", "metadata": {"name": GATEWAY_CLASS},
         "spec": {"controllerName": "gateway.envoyproxy.io/gatewayclass-controller"}},
        {"apiVersion": "gateway.networking.k8s.io/v1", "kind": "Gateway",
         "metadata": {"name": "rollout-gateway", "namespace": p.inference_namespace},
         "spec": {"gatewayClassName": GATEWAY_CLASS,
                  "listeners": [{"name": "http", "protocol": "HTTP", "port": 80,
                                 "allowedRoutes": {"namespaces": {"from": "Same"}}}]}},
    ]  # fmt: skip


def _sglang_args(plan: Plan, p: DeployProfile) -> list[str]:
    kv = {"publisher": "zmq", "endpoint": "$(KV_EVENTS_ENDPOINT)", "topic": f"kv@$(POD_IP):$(POD_PORT)@{p.model}"}
    return [
        "--model-path", p.hf_checkpoint, "--served-model-name", p.model, "--host", "0.0.0.0", "--port", str(POOL_PORT),
        *plan.sglang_args(), "--page-size", str(BLOCK_TOKENS), "--enable-metrics",
        "--kv-events-config", json.dumps(kv, separators=(",", ":")),
    ]  # fmt: skip


def rollout_pool(p: DeployProfile, plan: Plan) -> dict:
    """SGLang engines for RL rollouts, one pod per engine, sized by the planner (`plan.rollout`)."""
    r = plan.rollout
    if r.engine_tp > p.gpus_per_node:
        raise ValueError(f"engines of TP{r.engine_tp} span nodes: deploy them as a LeaderWorkerSet (not rendered)")
    meta = workload_meta(p, "inference", "serving")
    rdma = rdma_pod_patch(p, p.inference_namespace, min(r.engine_tp, len(p.rails)))
    res = {"nvidia.com/gpu": str(r.engine_tp), "cpu": str(16 * r.engine_tp), "memory": f"{160 * r.engine_tp}Gi",
           **rdma["resources"]}  # fmt: skip
    env = [{"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
           {"name": "POD_PORT", "value": str(POOL_PORT)},
           {"name": "KV_EVENTS_ENDPOINT", "value": f"tcp://*:{KV_EVENTS_PORT}"}]  # fmt: skip
    env += [{"name": k, "value": v} for k, v in rdma["env"].items()]
    probe = {"httpGet": {"path": "/health", "port": POOL_PORT}}
    spec = {
        "containers": [{
            "name": "sglang", "image": p.swarmrl_image, "imagePullPolicy": "IfNotPresent",
            "command": ["python3", "-m", "sglang.launch_server"], "args": _sglang_args(plan, p),
            "ports": [{"containerPort": POOL_PORT, "name": "http"},
                      {"containerPort": KV_EVENTS_PORT, "name": "kv-events"}],
            "env": env, "resources": {"requests": res, "limits": res},
            "securityContext": rdma["security_context"],
            "volumeMounts": [{"name": "models", "mountPath": "/models", "readOnly": True}] + rdma["volume_mounts"],
            "startupProbe": {**probe, "periodSeconds": 10, "failureThreshold": 180},
            "readinessProbe": {**probe, "periodSeconds": 5, "failureThreshold": 3},
            "livenessProbe": {**probe, "periodSeconds": 10, "failureThreshold": 6},
        }],
        "volumes": [{"name": "models", "persistentVolumeClaim": {"claimName": p.models_pvc}}] + rdma["volumes"],
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        "nodeSelector": {"nvidia.com/gpu.product": p.gpu_product},
        "terminationGracePeriodSeconds": 30,
    }  # fmt: skip
    if meta["scheduler_name"]:
        spec["schedulerName"] = meta["scheduler_name"]
    labels = {**POOL_LABEL, "llm-d.ai/engine-type": "sglang", "llm-d.ai/model": p.model.split("/")[-1]}
    return {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": "rollout-pool", "namespace": p.inference_namespace,
                     "labels": {**labels, **meta["labels"]}},
        "spec": {
            "replicas": r.engines, "selector": {"matchLabels": dict(POOL_LABEL)},
            "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxUnavailable": 1, "maxSurge": 0}},
            "template": {"metadata": {"labels": {**labels, **meta["labels"]},
                                      "annotations": {**rdma["annotations"], **meta["pod_annotations"]}},
                         "spec": spec},
        },
    }  # fmt: skip


def llm_inference_service(
    p: DeployProfile, plan: Plan, name: str = "swarm-policy", checkpoint: str | None = None
) -> dict:
    """KServe LLMInferenceService (llm-d) serving a trained checkpoint with vLLM and precise-prefix routing."""
    import yaml

    tp = plan.rollout.engine_tp
    # KServe's vLLM preset owns the command; extra flags go through VLLM_ADDITIONAL_ARGS (as in KServe's own
    # precise-prefix sample). The engine binds its KV-event socket; the EPP discovers pods and dials it.
    kv = {"enable_kv_cache_events": True, "publisher": "zmq", "endpoint": f"tcp://*:{KV_EVENTS_PORT}",
          "topic": f"kv@${{POD_IP}}:8000@{p.model}"}  # fmt: skip
    extra = f"--block-size {BLOCK_TOKENS} --kv-events-config '{json.dumps(kv, separators=(',', ':'))}'"
    rdma = rdma_pod_patch(p, p.inference_namespace, 0)
    return {
        "apiVersion": "serving.kserve.io/v1alpha2", "kind": "LLMInferenceService",
        "metadata": {"name": name, "namespace": p.inference_namespace},
        "spec": {
            "model": {"uri": f"pvc://{p.models_pvc}/{(checkpoint or p.save_dir).removeprefix('/models/')}",
                      "name": p.model},
            "replicas": max(1, plan.rollout.engines),
            "parallelism": {"tensor": tp},
            "template": {"containers": [{
                "name": "main",
                "env": [{"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
                        {"name": "VLLM_ADDITIONAL_ARGS", "value": extra}],
                "ports": [{"containerPort": KV_EVENTS_PORT, "name": "kv-events"}],
                "resources": {"limits": {"nvidia.com/gpu": str(tp)}, "requests": {"nvidia.com/gpu": str(tp)}},
                "volumeMounts": rdma["volume_mounts"],
            }], "volumes": rdma["volumes"]},
            "router": {
                "route": {}, "gateway": {},
                "scheduler": {"config": {"inline": yaml.safe_load(yaml.safe_dump(epp_config("vllm")))}},
            },
        },
    }  # fmt: skip
