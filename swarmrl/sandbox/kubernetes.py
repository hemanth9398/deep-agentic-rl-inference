"""Kubernetes sandbox: one pod per episode, for sandbox pools larger than one Docker host.

Uses kubectl (in-cluster service account or a kubeconfig). Network isolation comes from a
deny-all NetworkPolicy on the sandbox label: apply `network_policy_manifest()` once per
namespace (`KubernetesSandboxFactory.ensure_network_policy()`); the cluster's CNI must enforce
NetworkPolicy (Calico, Cilium, ...). Task images must already be in a registry the cluster can
pull from: set `docker_image` in task.toml or pass `image_for` to the factory.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from swarmrl.sandbox.base import ExecResult, InfraError, Sandbox, SandboxFactory
from swarmrl.sandbox.docker import _cli

LABEL = {"app.kubernetes.io/name": "swarmrl-sandbox"}


@dataclass(frozen=True)
class PodLimits:
    cpu: str = "2"
    memory: str = "4Gi"
    ephemeral_storage: str = "10Gi"


def pod_manifest(name: str, image: str, workdir: str, limits: PodLimits, namespace: str) -> dict:
    res = {"cpu": limits.cpu, "memory": limits.memory, "ephemeral-storage": limits.ephemeral_storage}
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace, "labels": dict(LABEL)},
        "spec": {
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "terminationGracePeriodSeconds": 0,
            "containers": [{
                "name": "sandbox",
                "image": image,
                "command": ["sleep", "infinity"],
                "workingDir": workdir,
                "env": [{"name": "PYTHONDONTWRITEBYTECODE", "value": "1"}],
                "resources": {"requests": res, "limits": res},
                "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
            }],
        },
    }  # fmt: skip


def network_policy_manifest(namespace: str) -> dict:
    """Deny all ingress and egress for sandbox pods (no network, as the design requires)."""
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": "swarmrl-sandbox-deny-all", "namespace": namespace},
        "spec": {"podSelector": {"matchLabels": dict(LABEL)}, "policyTypes": ["Ingress", "Egress"]},
    }


class KubernetesSandbox(Sandbox):
    binary = "kubectl"

    def __init__(self, image: str, namespace: str, workdir: str = "/workspace", limits: PodLimits = PodLimits(),
                 ready_timeout_s: int = 300):  # fmt: skip
        super().__init__()
        self.workdir = workdir
        self.namespace = namespace
        self.name = f"swarmrl-{uuid.uuid4().hex[:12]}"
        manifest = json.dumps(pod_manifest(self.name, image, workdir, limits, namespace))
        r = _cli(self.binary, "apply", "-f", "-", input=manifest)
        if r.returncode != 0:
            self._closed = True
            raise InfraError(f"pod create failed: {r.stderr.strip()}")
        try:
            w = _cli(self.binary, "wait", "-n", namespace, "--for=condition=Ready", f"pod/{self.name}",
                     f"--timeout={ready_timeout_s}s", timeout=ready_timeout_s + 30)  # fmt: skip
            if w.returncode != 0:
                raise InfraError(f"pod {self.name} not ready: {w.stderr.strip()}")
            self.init_baseline()
        except BaseException:
            self.close()
            raise

    def _run(self, cmd: str, timeout: float, input: str | None = None) -> ExecResult:
        limit = max(int(timeout), 1)
        inner = f"cd {shlex.quote(self.workdir)} && timeout --kill-after=5 {limit} bash -c {shlex.quote(cmd)}"
        args = [
            "exec",
            *(["-i"] if input is not None else []),
            "-n",
            self.namespace,
            self.name,
            "--",
            "bash",
            "-c",
            inner,
        ]
        try:
            p = subprocess.run(
                [self.binary, *args],
                capture_output=True,
                text=True,
                timeout=timeout + 60,
                input=input,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", f"command timed out after {timeout}s", timed_out=True)
        if p.returncode != 0 and ("NotFound" in p.stderr or "error dialing backend" in p.stderr):
            raise InfraError(p.stderr.strip())
        return ExecResult(p.returncode, p.stdout, p.stderr, timed_out=p.returncode == 124)

    def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        return self._run(cmd, timeout)

    def write_file(self, path: str, content: str) -> None:
        q = shlex.quote(path)
        r = self._run(f'mkdir -p "$(dirname {q})" && cat > {q}', timeout=120, input=content)
        if r.returncode != 0:
            raise OSError(f"write {path} failed: {r.output}")

    def copy_in(self, src: str | os.PathLike[str], dest: str = ".") -> None:
        dest_abs = dest if dest.startswith("/") else os.path.normpath(os.path.join(self.workdir, dest))
        r = _cli(self.binary, "cp", f"{os.fspath(src)}/.", f"{self.namespace}/{self.name}:{dest_abs}")
        if r.returncode != 0:
            raise InfraError(f"kubectl cp failed: {r.stderr.strip()}")

    def _teardown(self) -> None:
        _cli(self.binary, "delete", "pod", "-n", self.namespace, self.name, "--wait=false", "--grace-period=0",
             "--ignore-not-found", timeout=60)  # fmt: skip


class KubernetesSandboxFactory(SandboxFactory):
    def __init__(self, namespace: str = "swarmrl", limits: PodLimits = PodLimits(),
                 image_for: Callable[[object], str] | None = None):  # fmt: skip
        self.namespace = namespace
        self.limits = limits
        self._image_for = image_for

    def ensure_network_policy(self) -> None:
        r = _cli("kubectl", "apply", "-f", "-", input=json.dumps(network_policy_manifest(self.namespace)))
        if r.returncode != 0:
            raise InfraError(f"NetworkPolicy apply failed: {r.stderr.strip()}")

    def image_for(self, task) -> str:
        if self._image_for is not None:
            return self._image_for(task)
        if task.docker_image:
            return task.docker_image
        raise InfraError(f"task {task.task_id}: no registry image; set docker_image in task.toml or pass image_for")

    def create(self, task) -> KubernetesSandbox:
        return KubernetesSandbox(self.image_for(task), self.namespace, task.workdir, self.limits)
