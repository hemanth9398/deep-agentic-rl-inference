"""Validation step 4: a scripted oracle agent must resolve every task through the real tools.

It plays every role with the same tools and protocol as the model: the planner delegates,
the coder writes the oracle's post-fix files with create_file, the tester runs the tests,
the reviewer accepts. A task the oracle agent cannot resolve has a broken harness, tool or
grader path, and must not enter any split.
"""

from __future__ import annotations

import json

from swarmrl.grading.protected import changed_paths
from swarmrl.sandbox.base import SandboxFactory


def oracle_file_states(task, factory: SandboxFactory) -> dict[str, str | None]:
    """Post-fix contents of every file the oracle touches (None = deleted)."""
    paths = changed_paths(task.oracle_patch)
    sb = factory.create(task)
    try:
        r = sb.apply_patch(task.oracle_patch)
        if r.returncode != 0:
            raise ValueError(f"oracle patch does not apply: {r.output}")
        out: dict[str, str | None] = {}
        for p in paths:
            try:
                out[p] = sb.read_file(p)
            except (FileNotFoundError, OSError):
                out[p] = None
        return out
    finally:
        sb.close()


def _call(name: str, **arguments) -> str:
    return f"Applying the reference step.\n<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def oracle_script(files: dict[str, str | None], config):
    """Returns script(role, role_turn, context_text) -> reply for ScriptedPolicy."""
    edits = []
    for path, content in files.items():
        if content is None:
            edits.append(_call("bash", command=f"rm -f -- '{path}'"))
        else:
            edits.append(_call("create_file", path=path, content=content, overwrite=True))
    roles = set(config.role_names)

    def script(role: str, turn: int, context: str) -> str:
        if role == "planner":
            return _call("delegate", to="coder", plan="Apply the reference fix.")
        if role == "coder":
            if turn < len(edits):
                return edits[turn]
            return _call("handoff", to="tester" if "tester" in roles else "reviewer", note="Fix applied.")
        if role == "tester":
            if turn == 0:
                return _call("run_tests")
            return _call("handoff", to="reviewer", note="Tests ran.")
        if role == "reviewer":
            return _call("accept", note="Matches the issue.")
        if role == "agent":  # single-agent config
            if turn < len(edits):
                return edits[turn]
            return _call("submit")
        raise KeyError(role)

    return script


async def run_oracle_agent(task, make_runner, factory: SandboxFactory, config, chat):
    """make_runner(policy) -> EpisodeRunner wired to the real tools, grader and validation for `config`."""
    from swarmrl.inference.client import ScriptedPolicy

    policy = ScriptedPolicy(chat, oracle_script(oracle_file_states(task, factory), config))
    return await make_runner(policy).run(task, seed=0)
