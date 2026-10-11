"""DockerSandbox and KubernetesSandbox end to end against fake CLIs (no daemon or cluster needed)."""

import asyncio
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from swarmrl.grading.grader import CleanRoomGrader
from swarmrl.inference.chat import ByteTokenizer, ChatFormat
from swarmrl.inference.client import ScriptedPolicy
from swarmrl.sandbox.base import InfraError
from swarmrl.sandbox.docker import DockerSandboxFactory
from swarmrl.sandbox.kubernetes import KubernetesSandboxFactory
from swarmrl.swarm.config import default_swarm_config
from swarmrl.swarm.runtime import EpisodeRunner
from swarmrl.tasks.oracle_agent import oracle_file_states, oracle_script
from swarmrl.tasks.validate import TaskValidator

FAKE = Path(__file__).parent / "fakes" / "fake_cli.py"


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("docker", "kubectl"):
        p = bin_dir / tool
        p.write_text(f'#!/bin/sh\nFAKE_TOOL={tool} exec {sys.executable} {FAKE} "$@"\n')
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    state = tmp_path / "state"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLI_STATE", str(state))
    return state


def calls(state, tool):
    return [c["argv"] for c in map(json.loads, (state / "calls.jsonl").read_text().splitlines()) if c["tool"] == tool]


def run_oracle(task, factory):
    rec = TaskValidator(factory).validate(task)
    assert rec.valid, rec.reason
    cfg = default_swarm_config()
    chat = ChatFormat(ByteTokenizer())
    policy = ScriptedPolicy(chat, oracle_script(oracle_file_states(task, factory), cfg))
    runner = EpisodeRunner(cfg, policy, chat, factory, CleanRoomGrader(factory, cfg.protected_paths), lambda t: rec)
    return asyncio.run(runner.run(task))


def live(state):
    return [p.name for p in state.iterdir() if p.is_dir()]


def test_docker_sandbox_end_to_end(task, fake_cli):
    ep = run_oracle(task, DockerSandboxFactory())
    assert ep.resolved and ep.team_reward == 1.0
    runs = calls(fake_cli, "docker")
    starts = [a for a in runs if a[0] == "run"]
    assert starts and all("--network" in a and a[a.index("--network") + 1] == "none" for a in starts)
    assert all("--pids-limit" in a and "--memory" in a and "no-new-privileges" in a for a in starts)
    assert sum(a[0] == "build" for a in runs) == 1  # one image build per task, reused by every container
    assert live(fake_cli) == []  # every container removed


def test_docker_write_file_beyond_argv_limit(task, fake_cli):
    sb = DockerSandboxFactory().create(task)
    try:
        big = "x = 1\n" * 60_000  # 360 KB: larger than the kernel's 128 KB single-argument limit
        sb.write_file("big.py", big)
        assert sb.read_file("big.py") == big
    finally:
        sb.close()


def test_docker_missing_image_is_infra_error(task, fake_cli):
    task.docker_image = "does-not-exist"
    with pytest.raises(InfraError):
        DockerSandboxFactory().create(task)


def test_kubernetes_sandbox_end_to_end(task, fake_cli):
    f = KubernetesSandboxFactory(namespace="swarm", image_for=lambda t: str(t.repo_dir))
    f.ensure_network_policy()
    ep = run_oracle(task, f)
    assert ep.resolved and ep.team_reward == 1.0
    pol = json.loads((fake_cli / "NetworkPolicy-swarmrl-sandbox-deny-all.json").read_text())
    assert pol["spec"]["policyTypes"] == ["Ingress", "Egress"]
    assert (fake_cli / "Namespace-swarm.json").exists()  # created when missing
    pods = [json.loads(p.read_text()) for p in fake_cli.glob("Pod-*.json")]
    assert pods and all(p["spec"]["automountServiceAccountToken"] is False for p in pods)
    c = pods[0]["spec"]["containers"][0]
    assert c["imagePullPolicy"] == "IfNotPresent"  # pre-loaded / air-gapped images work even when untagged
    assert c["resources"]["requests"]["cpu"] == "500m" and c["resources"]["limits"]["cpu"] == "2"
    assert all(p["metadata"]["labels"] == pol["spec"]["podSelector"]["matchLabels"] for p in pods)
    assert live(fake_cli) == []


def test_kubernetes_requires_registry_image(task):
    task.docker_image = None
    with pytest.raises(InfraError):
        KubernetesSandboxFactory().image_for(task)
