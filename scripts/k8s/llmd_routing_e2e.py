"""End-to-end check of the llm-d routing path on a real cluster: Envoy -> EPP (precise prefix cache) -> engines.

Deploys three llm-d-inference-sim engines (vLLM mode) that publish real KV-cache events over ZMQ, an
InferencePool, the llm-d EPP loaded with the config `swarmrl.deploy.inference.epp_config()` generates, and Envoy
with the ext_proc filter, as llm-d's standalone router does. It then plays multi-turn sessions whose context
grows every turn, like a swarm role's, and checks that

  1. every later turn of a session goes back to the engine that served its first turn (prefix affinity).
     Speculative indexing is OFF here, so the EPP can only know where a prefix lives from the engines' KV events;
  2. different sessions spread over more than one engine;
  3. pre-tokenized requests on vLLM's native /inference/v1/generate are parsed and routed too;
  4. the EPP's KV-block index was filled from engine events and answered lookups (its own metrics);
  5. the SGLang flavour of our EPP config (sglanghttp-parser, engineType sglang) is accepted by the same EPP.

Needs the GAIE InferencePool CRD and the llm-d.ai CRDs installed, and the three images available to the cluster:
  python scripts/k8s/llmd_routing_e2e.py --epp-image IMG --sim-image IMG --envoy-image IMG
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swarmrl.deploy.inference import epp_config

NS = "llmd-e2e"
MODEL = "e2e-model"
BLOCK = 16
KV_PORT = 5557
POOL_LABEL = {"app": "llmd-e2e-sim"}

ENVOY = """
admin: {address: {socket_address: {address: 127.0.0.1, port_value: 19000}}}
static_resources:
  listeners:
  - name: router
    address: {socket_address: {address: 0.0.0.0, port_value: 8081}}
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: router
          route_config:
            virtual_hosts:
            - name: all
              domains: ["*"]
              routes:
              - match: {prefix: "/"}
                route: {cluster: original_destination_cluster, timeout: 300s}
          http_filters:
          - name: envoy.filters.http.ext_proc
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
              failure_mode_allow: false
              grpc_service: {envoy_grpc: {cluster_name: ext_proc, authority: "localhost:9002"}, timeout: 10s}
              processing_mode:
                request_header_mode: SEND
                response_header_mode: SEND
                request_body_mode: FULL_DUPLEX_STREAMED
                response_body_mode: FULL_DUPLEX_STREAMED
                request_trailer_mode: SEND
                response_trailer_mode: SEND
              message_timeout: 300s
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
              suppress_envoy_headers: true
  clusters:
  - name: original_destination_cluster
    type: ORIGINAL_DST
    connect_timeout: 30s
    lb_policy: CLUSTER_PROVIDED
    original_dst_lb_config: {use_http_header: true, http_header_name: x-gateway-destination-endpoint}
  - name: ext_proc
    type: STATIC
    connect_timeout: 30s
    typed_extension_protocol_options:
      envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
        "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
        explicit_http_config: {http2_protocol_options: {}}
    load_assignment:
      cluster_name: ext_proc
      endpoints:
      - lb_endpoints:
        - endpoint: {address: {socket_address: {address: 127.0.0.1, port_value: 9002}}}
"""


def kubectl(*args, input: str | None = None, check: bool = True) -> str:
    r = subprocess.run(["kubectl", *args], input=input, capture_output=True, text=True, timeout=600)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def apply(objs: list[dict]) -> None:
    kubectl("apply", "-f", "-", input="\n".join(json.dumps(o) for o in objs))


def epp_container(image: str, config_key: str) -> dict:
    return {
        "name": "epp", "image": image, "imagePullPolicy": "IfNotPresent",
        "args": ["--pool-name", "sim-pool", "--pool-namespace", NS, "--pool-group", "inference.networking.k8s.io",
                 "--config-file", f"/config/{config_key}", "--zap-encoder", "json", "--grpc-health-port", "9003",
                 "--secure-serving=false", "--tracing=false", "--metrics-endpoint-auth=false", "-v=2"],
        "env": [{"name": "NAMESPACE", "value": NS},
                {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}}],
        "ports": [{"containerPort": 9002, "name": "grpc"}, {"containerPort": 9003, "name": "grpc-health"},
                  {"containerPort": 9090, "name": "metrics"}],
        "readinessProbe": {"grpc": {"port": 9003, "service": "inference-extension"}, "periodSeconds": 2},
        "volumeMounts": [{"name": "config", "mountPath": "/config"}],
    }  # fmt: skip


def manifests(a) -> list[dict]:
    import yaml

    vllm_cfg = epp_config("vllm", block_tokens=BLOCK, kv_port=KV_PORT, speculative=False)
    sglang_cfg = epp_config("sglang", block_tokens=BLOCK, kv_port=KV_PORT)
    sim = {
        "name": "sim", "image": a.sim_image, "imagePullPolicy": "IfNotPresent",
        "args": ["--model", MODEL, "--port", "8000", "--mode", "random", "--enable-kvcache", "--kv-cache-size", "4096",
                 "--block-size", str(BLOCK), "--hash-seed", "42", "--zmq-endpoint", f"tcp://*:{KV_PORT}",
                 "--event-batch-size", "1", "--max-num-seqs", "32", "--max-model-len", "8192"],
        "env": [{"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
                {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                {"name": "POD_NAMESPACE", "value": NS}],
        "ports": [{"containerPort": 8000, "name": "http"}, {"containerPort": KV_PORT, "name": "kv-events"}],
        "readinessProbe": {"httpGet": {"path": "/health", "port": 8000}, "periodSeconds": 2},
    }  # fmt: skip
    rules = [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch"]},
        {"apiGroups": ["inference.networking.k8s.io"], "resources": ["inferencepools"],
         "verbs": ["get", "list", "watch"]},
        {"apiGroups": ["llm-d.ai", "inference.networking.x-k8s.io"],
         "resources": ["inferenceobjectives", "inferencemodelrewrites"], "verbs": ["get", "list", "watch"]},
    ]  # fmt: skip
    cfg = {"epp-vllm.yaml": yaml.safe_dump(vllm_cfg, sort_keys=False),
           "epp-sglang.yaml": yaml.safe_dump(sglang_cfg, sort_keys=False), "envoy.yaml": ENVOY}  # fmt: skip
    vol = [{"name": "config", "configMap": {"name": "router-config"}}]
    return [
        {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": "epp", "namespace": NS}},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "Role", "metadata": {"name": "epp", "namespace": NS},
         "rules": rules},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding",
         "metadata": {"name": "epp", "namespace": NS},
         "subjects": [{"kind": "ServiceAccount", "name": "epp", "namespace": NS}],
         "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": "epp"}},
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "router-config", "namespace": NS}, "data": cfg},
        {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "sim", "namespace": NS},
         "spec": {"replicas": 3, "selector": {"matchLabels": POOL_LABEL},
                  "template": {"metadata": {"labels": {**POOL_LABEL, "llm-d.ai/engine-type": "vllm"}},
                               "spec": {"containers": [sim]}}}},
        {"apiVersion": "inference.networking.k8s.io/v1", "kind": "InferencePool",
         "metadata": {"name": "sim-pool", "namespace": NS},
         "spec": {"selector": {"matchLabels": POOL_LABEL}, "targetPorts": [{"number": 8000}],
                  "endpointPickerRef": {"name": "router", "port": {"number": 9002}, "failureMode": "FailClose"}}},
        {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "router", "namespace": NS},
         "spec": {"replicas": 1, "selector": {"matchLabels": {"app": "router"}},
                  "template": {"metadata": {"labels": {"app": "router"}}, "spec": {
                      "serviceAccountName": "epp", "volumes": vol,
                      "containers": [epp_container(a.epp_image, "epp-vllm.yaml"), {
                          "name": "envoy", "image": a.envoy_image, "imagePullPolicy": "IfNotPresent",
                          "args": ["-c", "/config/envoy.yaml", "--log-level", "warn"],
                          "ports": [{"containerPort": 8081, "name": "http"}],
                          "volumeMounts": [{"name": "config", "mountPath": "/config"}]}]}}}},
        {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "router", "namespace": NS},
         "spec": {"selector": {"app": "router"}, "ports": [{"name": "http", "port": 8081, "targetPort": 8081},
                                                          {"name": "grpc", "port": 9002, "targetPort": 9002}]}},
        # The SGLang flavour of the same config must load in the same EPP (it is what the RL pool runs).
        {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "epp-sglang-config", "namespace": NS},
         "spec": {"replicas": 1, "selector": {"matchLabels": {"app": "epp-sglang-config"}},
                  "template": {"metadata": {"labels": {"app": "epp-sglang-config"}}, "spec": {
                      "serviceAccountName": "epp", "volumes": vol,
                      "containers": [epp_container(a.epp_image, "epp-sglang.yaml")]}}}},
    ]  # fmt: skip


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, body: dict) -> tuple[int, str, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.headers.get("x-inference-pod", ""), json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("x-inference-pod", ""), {"error": e.read().decode(errors="replace")[:300]}


def epp_metrics(port: int) -> dict[str, float]:
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("kvcache_index_") and " " in line and "{" not in line:
            name, value = line.rsplit(" ", 1)
            out[name] = float(value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epp-image", default="ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.11.0")
    ap.add_argument("--sim-image", default="ghcr.io/llm-d/llm-d-inference-sim:v0.12.0")
    ap.add_argument("--envoy-image", default="docker.io/envoyproxy/envoy:distroless-v1.33.2")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--keep", action="store_true", help="leave the namespace for inspection")
    a = ap.parse_args()

    kubectl("create", "namespace", NS, check=False)
    apply(manifests(a))
    for dep in ("sim", "router", "epp-sglang-config"):
        kubectl("rollout", "status", f"deployment/{dep}", "-n", NS, "--timeout=300s")
    results: dict[str, str] = {}
    port = free_port()
    pf = subprocess.Popen(["kubectl", "port-forward", "-n", NS, "svc/router", f"{port}:8081"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=2)
                break
            except Exception:
                time.sleep(1)
        rng = random.Random(0)
        time.sleep(3)  # let the EPP subscribe to every engine's KV-event socket
        first: dict[int, str] = {}
        later: list[tuple[int, int, str]] = []
        contexts = {s: [rng.randrange(1000, 30000) for _ in range(256)] for s in range(a.sessions)}
        for turn in range(a.turns):
            for s in range(a.sessions):
                if turn:
                    contexts[s] += [rng.randrange(1000, 30000) for _ in range(48)]  # model turn + observation
                code, pod, body = post(base + "/v1/completions",
                                       {"model": MODEL, "prompt": contexts[s], "max_tokens": 4})  # fmt: skip
                if code != 200 or not pod:
                    raise RuntimeError(f"session {s} turn {turn}: HTTP {code} pod={pod!r} {body}")
                if turn == 0:
                    first[s] = pod
                else:
                    later.append((s, turn, pod))
            time.sleep(1.0)  # KV events from this turn reach the EPP's index
        sticky = sum(pod == first[s] for s, _, pod in later)
        spread = Counter(first.values())
        results["prefix affinity"] = f"{sticky}/{len(later)} later turns on the engine holding the session's prefix"
        results["session spread"] = ", ".join(f"{p.split('-')[-1]}:{n}" for p, n in sorted(spread.items()))
        native = {"model": MODEL, "token_ids": contexts[0], "sampling_params": {"max_tokens": 4}}
        code, pod, body = post(base + "/inference/v1/generate", native)
        results["/inference/v1/generate"] = f"HTTP {code}, routed to {pod.split('-')[-1] if pod else '-'}" + (
            ", same engine as the session" if pod == first[0] else ""
        )
        mport = free_port()
        mpf = subprocess.Popen(["kubectl", "port-forward", "-n", NS, "deploy/router", f"{mport}:9090"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        try:
            time.sleep(3)
            m = epp_metrics(mport)
        finally:
            mpf.terminate()
        admitted, hits = m.get("kvcache_index_admissions_total", 0), m.get("kvcache_index_lookup_hits_total", 0)
        results["KV-event index"] = f"{admitted:.0f} blocks admitted from engine events, {hits:.0f} lookup hits"
        sg = json.loads(kubectl("get", "deploy", "epp-sglang-config", "-n", NS, "-o", "json"))
        results["sglang EPP config"] = f"{sg['status'].get('readyReplicas', 0)}/1 EPP ready with engineType sglang"
        ok = (sticky == len(later) and len(spread) >= 2 and code == 200 and pod == first[0] and admitted > 0
              and hits > 0 and sg["status"].get("readyReplicas", 0) == 1)  # fmt: skip
    finally:
        pf.terminate()
        if not a.keep:
            kubectl("delete", "namespace", NS, "--wait=false", check=False)
    for k, v in results.items():
        print(f"{k:28s} {v}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
