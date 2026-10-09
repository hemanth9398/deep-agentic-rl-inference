import json
import shutil
from pathlib import Path

import pytest

from swarmrl.grading.grader import CleanRoomGrader
from swarmrl.inference.chat import ByteTokenizer, ChatFormat
from swarmrl.inference.client import ScriptedPolicy
from swarmrl.sandbox.local import LocalSandboxFactory
from swarmrl.swarm.config import default_swarm_config
from swarmrl.swarm.runtime import EpisodeRunner
from swarmrl.tasks.harbor import HarborTask
from swarmrl.tasks.validate import TaskValidator

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tasks" / "stats-empty-input"


@pytest.fixture
def task(tmp_path) -> HarborTask:
    d = tmp_path / "tasks" / EXAMPLE.name
    shutil.copytree(EXAMPLE, d)
    return HarborTask.load(d)


@pytest.fixture
def factory():
    return LocalSandboxFactory()


@pytest.fixture
def record(task, factory):
    rec = TaskValidator(factory).validate(task)
    assert rec.valid, rec.reason
    return rec


@pytest.fixture
def chat():
    return ChatFormat(ByteTokenizer())


def call(name, **arguments):
    return f"thinking...\n<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


@pytest.fixture
def make_runner(factory, record, chat):
    def make(script, config=None, **kw):
        cfg = config or default_swarm_config()
        policy = ScriptedPolicy(chat, script)
        grader = CleanRoomGrader(factory, cfg.protected_paths)
        return EpisodeRunner(cfg, policy, chat, kw.pop("sandbox_factory", factory), grader, lambda t: record, **kw)

    return make
