# Deployment: profile `h100-colocated`

Model `Qwen/Qwen3-30B-A3B`; training on 1 node(s) x 8 GPUs, layout **TP4 PP1 CP2 DP1 EP8 ETP1 EDP1** (~76 GiB/GPU); rollouts colocated: 2 SGLang engine(s) x TP4 on the training GPUs, weights by CUDA IPC.
Scheduler: **kueue**. RDMA: 8 SR-IOV rails (eth).

## Install order

1. Cluster add-ons (once): Cilium with `10-fabric/values/cilium.yaml`; Multus thick (`ghcr.io/k8snetworkplumbingwg/multus-cni:v4.3.1-thick`) and whereabouts; NVIDIA GPU Operator (open kernel module, GPUDirect RDMA via DMA-BUF); SR-IOV Network Operator:
   `helm install -n sriov-network-operator --create-namespace --version 1.6.0 sriov-network-operator oci://ghcr.io/k8snetworkplumbingwg/sriov-network-operator-chart`
2. Scheduling: Kueue v0.20 (merge `20-scheduling/values/kueue-manager-config.yaml` into `kueue-manager-config`) and KubeRay v1.7.
3. Gateway: Envoy Gateway + Envoy AI Gateway:
   `helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm -n envoy-gateway-system --create-namespace -f 40-inference/values/envoy-gateway.yaml` and `helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm -n envoy-ai-gateway-system --create-namespace`; Gateway API Inference Extension CRDs v1.6 (`v1-manifests.yaml`).
4. `kubectl apply -f 00-namespaces.yaml -f 10-fabric/sriov.yaml` and wait for `kubectl get sriovnetworknodestates -A` to report `Succeeded` (nodes drain to create VFs).
5. `kubectl apply -f 20-scheduling/ -f 40-inference/gateway.yaml`
6. RL job: `kubectl apply -f 30-training/`. One RayJob per stage; re-render with the next `stage` in the profile when a stage's gate passes.
7. Serving a checkpoint: `kubectl apply -f 50-serving/` (KServe v0.21 with llm-d).

## Files

`values/` folders hold Helm values and config fragments; every other file is `kubectl apply`-able.

- `00-namespaces.yaml`
- `10-fabric/sriov.yaml`
- `10-fabric/values/cilium.yaml`
- `20-scheduling/kueue.yaml`
- `20-scheduling/values/kueue-manager-config.yaml`
- `30-training/rbac.yaml`
- `30-training/swarmrl-config.yaml`
- `30-training/rayjob.yaml`
- `40-inference/gateway.yaml`
- `40-inference/values/envoy-gateway.yaml`
- `50-serving/llminferenceservice.yaml`
