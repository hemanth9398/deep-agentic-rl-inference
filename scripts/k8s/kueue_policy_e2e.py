"""Run the rendered Kueue policy against a real Kueue controller: quota, borrowing, reclaim by preemption.

Installs Kueue from its kustomize output with our manager-config fragment merged in, gives the node fake
GPUs (an extended resource, as the Kubernetes docs describe), applies the ClusterQueues/LocalQueues/priorities
that `swarmrl deploy` renders for the profile, and plays the scenario the policy exists for (profile h100:
8 GPUs of rl-train quota, 8 of serving):

  1. RL job A (8 GPUs)                 -> admitted on rl-train's own quota
  2. RL job B (8 GPUs)                 -> admitted by borrowing serving's idle GPUs
  3. eval job (1 GPU)                  -> waits: nothing idle left
  4. serving workload (8 GPUs)         -> admitted by reclaiming: B (the borrower) is preempted and requeued
  5. A finishes                        -> B is admitted again before eval (rl-train outranks eval)

  kubectl kustomize <kueue-src>/config/default > kueue.yaml
  python scripts/k8s/kueue_policy_e2e.py --kueue-manifests kueue.yaml --kueue-image IMG [--profile P]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from swarmrl.deploy.profile import DeployProfile  # noqa: E402
from swarmrl.deploy.scheduling import kueue_configuration, kueue_objects  # noqa: E402

TEST_IMAGE = "python:3.12-slim"


def kubectl(*args, input: str | None = None, check: bool = True) -> str:
    r = subprocess.run(["kubectl", *args], input=input, capture_output=True, text=True, timeout=600)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args[:4])}: {r.stderr.strip()[:500]}")
    return r.stdout


def apply(objs, server_side: bool = False) -> None:
    extra = ["--server-side", "--force-conflicts"] if server_side else []
    kubectl("apply", *extra, "-f", "-", input="\n---\n".join(yaml.safe_dump(o) for o in objs))


def merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        base[k] = merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def install_kueue(manifests: Path, image: str) -> None:
    docs = [d for d in yaml.safe_load_all(manifests.read_text()) if d]
    for d in docs:
        if d["kind"] == "Deployment" and d["metadata"]["name"] == "kueue-controller-manager":
            for c in d["spec"]["template"]["spec"]["containers"]:
                if c["name"] == "manager":
                    c["image"], c["imagePullPolicy"] = image, "IfNotPresent"
        if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "kueue-manager-config":
            key = "controller_manager_config.yaml"
            cfg = merge(yaml.safe_load(d["data"][key]), {k: v for k, v in kueue_configuration().items()
                                                          if k not in ("apiVersion", "kind")})  # fmt: skip
            d["data"][key] = yaml.safe_dump(cfg, sort_keys=False)
    apply(docs, server_side=True)
    kubectl("rollout", "status", "deployment/kueue-controller-manager", "-n", "kueue-system", "--timeout=300s")


def fake_gpus(p: DeployProfile, gpus: int) -> str:
    node = kubectl("get", "nodes", "-o", "jsonpath={.items[0].metadata.name}")
    labels = {"nvidia.com/gpu.product": p.gpu_product, "nvidia.com/gpu.present": "true"}
    labels.update({lvl: f"{lvl.split('/')[-1]}-1" for lvl in p.topology_levels if lvl != "kubernetes.io/hostname"})
    kubectl("label", "node", node, "--overwrite", *[f"{k}={v}" for k, v in labels.items()])
    patch = [{"op": "add", "path": "/status/capacity/nvidia.com~1gpu", "value": str(gpus)}]
    kubectl("patch", "node", node, "--subresource=status", "--type=json", "-p", json.dumps(patch))
    return node


def job(ns: str, name: str, queue: str, priority: str, gpus: int) -> dict:
    res = {"nvidia.com/gpu": str(gpus), "cpu": "50m", "memory": "32Mi"}
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": ns,
                     "labels": {"kueue.x-k8s.io/queue-name": queue, "kueue.x-k8s.io/priority-class": priority}},
        "spec": {"suspend": True, "backoffLimit": 0, "template": {"spec": {
            "restartPolicy": "Never", "terminationGracePeriodSeconds": 0,
            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
            "containers": [{"name": "work", "image": TEST_IMAGE, "imagePullPolicy": "IfNotPresent",
                            "command": ["sleep", "3600"], "resources": {"requests": res, "limits": res}}]}}},
    }  # fmt: skip


def workload(ns: str, name: str) -> dict:
    uid = kubectl("get", "job", name, "-n", ns, "-o", "jsonpath={.metadata.uid}")
    items = json.loads(kubectl("get", "workloads", "-n", ns, "-l", f"kueue.x-k8s.io/job-uid={uid}", "-o", "json"))
    return items["items"][0] if items["items"] else {}


def state(ns: str, name: str) -> tuple[bool, set[str], str]:
    w = workload(ns, name)
    conds = {c["type"]: c for c in w.get("status", {}).get("conditions", [])}
    true = {t for t, c in conds.items() if c["status"] == "True"}
    why = conds.get("Evicted", conds.get("Preempted", conds.get("QuotaReserved", {}))).get("message", "")
    return "Admitted" in true, true, why


def wait(ns: str, name: str, admitted: bool, timeout: float = 90) -> tuple[set[str], str]:
    deadline = time.monotonic() + timeout
    while True:
        ok, conds, why = state(ns, name)
        if ok == admitted:
            return conds, why
        if time.monotonic() > deadline:
            raise TimeoutError(f"{ns}/{name}: expected admitted={admitted}, conditions={sorted(conds)} {why}")
        time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kueue-manifests", required=True, type=Path)
    ap.add_argument("--kueue-image", required=True)
    ap.add_argument("--profile", default=str(ROOT / "deploy" / "profiles" / "h100.yaml"))
    a = ap.parse_args()
    p = DeployProfile.load(a.profile)
    total = p.train_gpus + p.inference_gpus

    install_kueue(a.kueue_manifests, a.kueue_image)
    fake_gpus(p, total)
    for ns in (p.training_namespace, p.inference_namespace):
        kubectl("create", "namespace", ns, check=False)
    for attempt in range(30):  # the webhook takes a moment to serve after the rollout
        try:
            apply(kueue_objects(p))
            break
        except RuntimeError:
            if attempt == 29:
                raise
            time.sleep(4)

    rl, inf = p.training_namespace, p.inference_namespace
    steps: list[tuple[str, str]] = []
    try:
        apply([job(rl, "train-a", "rl", "rl-train", p.train_gpus)])
        wait(rl, "train-a", True)
        steps.append(("RL job A, rl-train quota", "admitted"))
        apply([job(rl, "train-b", "rl", "rl-train", p.inference_gpus)])
        wait(rl, "train-b", True)
        steps.append(("RL job B, borrows serving's idle GPUs", "admitted"))
        apply([job(rl, "eval-c", "eval", "eval", 1)])
        time.sleep(10)
        assert not state(rl, "eval-c")[0], "eval was admitted with no idle GPUs"
        steps.append(("eval job, nothing idle", "pending"))
        apply([job(inf, "serve", "inference", "serving", p.inference_gpus)])
        wait(inf, "serve", True)
        conds, why = wait(rl, "train-b", False)
        assert "Evicted" in conds or "Preempted" in conds, conds
        steps.append(("serving reclaims its GPUs", f"admitted; job B preempted ({why[:70]})"))
        assert state(rl, "train-a")[0], "A must keep its own quota"
        kubectl("delete", "job", "train-a", "-n", rl, "--wait=true")
        wait(rl, "train-b", True)
        assert not state(rl, "eval-c")[0], "eval must not jump ahead of rl-train"
        steps.append(("A finishes", "B re-admitted before eval"))
        ok = True
    except (AssertionError, TimeoutError) as e:
        steps.append(("FAILED", str(e)))
        ok = False
    finally:
        for ns, name in ((rl, "train-a"), (rl, "train-b"), (rl, "eval-c"), (inf, "serve")):
            kubectl("delete", "job", name, "-n", ns, "--ignore-not-found", "--wait=false", check=False)
    for what, result in steps:
        print(f"{what:42s} {result}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
