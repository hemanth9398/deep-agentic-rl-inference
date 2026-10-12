#!/usr/bin/env bash
# A single-node Kubernetes cluster (k3s in Docker) for running swarmrl sandboxes locally / in CI.
#
#   scripts/k8s/local_cluster.sh up                 start the cluster, write ./.k8s/kubeconfig.yaml
#   scripts/k8s/local_cluster.sh load IMAGE[:TAG]   copy a local Docker image into the cluster's containerd
#   scripts/k8s/local_cluster.sh down               delete the cluster
#   eval "$(scripts/k8s/local_cluster.sh env)"      export KUBECONFIG (and NO_PROXY when a proxy is set)
#
# k3s ships a NetworkPolicy controller, so swarmrl's deny-all policy is actually enforced.
#
# Flags only needed on restricted hosts (all safe elsewhere):
#   fail-cgroupv1=false           kubelet >= 1.35 refuses cgroup v1 hosts by default
#   restrict_oom_score_adj=true   (containerd-config-v3.toml.tmpl) hosts without CAP_SYS_RESOURCE cannot lower
#                                 oom_score_adj; without it every pod sandbox fails to start
#   image-gc / eviction-hard      hosts whose disk *looks* nearly full (quota'd VMs): stops the kubelet from
#                                 garbage-collecting pre-loaded images and evicting pods
#   host network + HTTPS_PROXY    lets the node pull system images through an egress proxy on 127.0.0.1
set -euo pipefail

NAME=${SWARMRL_K3S_NAME:-swarmrl-k3s}
IMAGE=${SWARMRL_K3S_IMAGE:-rancher/k3s:v1.37.1-k3s1}  # containerd-config-v3.toml.tmpl matches this version
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${SWARMRL_K8S_DIR:-$PWD/.k8s}
CONTAINERD_TMPL=/var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.tmpl
NODE_CIDRS="192.0.2.0/24,10.42.0.0/16,10.43.0.0/16"

up() {
  mkdir -p "$OUT"
  rm -f "$OUT/kubeconfig.yaml"
  local args=(-d --name "$NAME" --privileged --network host --tmpfs /run --tmpfs /var/run
              -e K3S_KUBECONFIG_OUTPUT=/output/kubeconfig.yaml -e K3S_KUBECONFIG_MODE=644
              -v "$OUT:/output" -v "$HERE/containerd-config-v3.toml.tmpl:$CONTAINERD_TMPL:ro")
  if [[ -n "${HTTPS_PROXY:-}" ]]; then
    args+=(-e "HTTPS_PROXY=$HTTPS_PROXY" -e "NO_PROXY=${NO_PROXY:-localhost,127.0.0.1},$NODE_CIDRS")
  fi
  if [[ -n "${SWARMRL_EXTRA_CA:-}" ]]; then  # e.g. an egress proxy's CA bundle
    args+=(-v "$SWARMRL_EXTRA_CA:/etc/ssl/certs/ca-certificates.crt:ro")
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run "${args[@]}" "$IMAGE" server \
    --disable=traefik --disable=metrics-server --disable=local-storage \
    --kubelet-arg=fail-cgroupv1=false \
    --kubelet-arg=image-gc-high-threshold=100 --kubelet-arg=image-gc-low-threshold=99 \
    '--kubelet-arg=eviction-hard=imagefs.available<1%,nodefs.available<1%' >/dev/null
  export KUBECONFIG="$OUT/kubeconfig.yaml"
  for _ in $(seq 1 90); do
    [[ -f "$KUBECONFIG" ]] && kubectl get nodes 2>/dev/null | grep -q " Ready" && break
    sleep 2
  done
  for _ in $(seq 1 60); do kubectl get serviceaccount default >/dev/null 2>&1 && break; sleep 2; done
  kubectl get nodes
}

load() {
  # --all-platforms keeps multi-arch indexes intact; a pulled image holds only the host's platform, so retry without
  docker save "$1" | docker exec -i "$NAME" ctr -n k8s.io images import --all-platforms - >/dev/null 2>&1 ||
    docker save "$1" | docker exec -i "$NAME" ctr -n k8s.io images import - >/dev/null
  docker exec "$NAME" crictl images | grep -F "${1%%:*}"
}

case "${1:-}" in
  up) up ;;
  load) load "${2:?image}" ;;
  down) docker rm -f "$NAME" >/dev/null && echo "deleted $NAME" ;;
  env)
    echo "export KUBECONFIG=$OUT/kubeconfig.yaml"
    if [[ -n "${HTTPS_PROXY:-}" ]]; then
      echo "export NO_PROXY=\"\${NO_PROXY},$NODE_CIDRS\" no_proxy=\"\${no_proxy},$NODE_CIDRS\""
    fi
    ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac
