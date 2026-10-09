"""Every interface is an abstract base class, and every backend actually implements it."""

import inspect

import pytest

from swarmrl.inference.chat import ByteTokenizer, HFTokenizer, Tokenizer
from swarmrl.inference.client import PolicyClient, ScriptedPolicy, SGLangClient
from swarmrl.inference.weight_sync import DiskWeightSync, NullWeightSync, WeightSync
from swarmrl.integrations.skyrl import SkyRLEnginePolicy
from swarmrl.rl.trainer import TorchPolicyTrainer, Trainer
from swarmrl.sandbox.base import Sandbox, SandboxFactory
from swarmrl.sandbox.docker import DockerSandbox, DockerSandboxFactory
from swarmrl.sandbox.kubernetes import KubernetesSandbox, KubernetesSandboxFactory
from swarmrl.sandbox.local import LocalSandbox, LocalSandboxFactory

IMPLEMENTATIONS = [
    (Sandbox, [LocalSandbox, DockerSandbox, KubernetesSandbox]),
    (SandboxFactory, [LocalSandboxFactory, DockerSandboxFactory, KubernetesSandboxFactory]),
    (Tokenizer, [ByteTokenizer, HFTokenizer]),
    (PolicyClient, [SGLangClient, ScriptedPolicy, SkyRLEnginePolicy]),
    (WeightSync, [DiskWeightSync, NullWeightSync]),
    (Trainer, [TorchPolicyTrainer]),
]


@pytest.mark.parametrize("base,impls", IMPLEMENTATIONS, ids=lambda x: getattr(x, "__name__", ""))
def test_interfaces_are_abstract_and_fully_implemented(base, impls):
    assert inspect.isabstract(base)
    with pytest.raises(TypeError):
        base()
    for impl in impls:
        assert issubclass(impl, base)
        assert not inspect.isabstract(impl), f"{impl.__name__} leaves {impl.__abstractmethods__} unimplemented"
