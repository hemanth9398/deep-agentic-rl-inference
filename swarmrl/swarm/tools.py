"""Tools. Coordination happens only through tools, so hand-offs are learnable actions."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from swarmrl.grading.protected import is_protected
from swarmrl.sandbox.base import BASELINE_TAG, Sandbox
from swarmrl.swarm.config import RoleConfig, SwarmConfig
from swarmrl.traces.schema import ErrorKind


@dataclass
class ToolContext:
    sandbox: Sandbox
    config: SwarmConfig
    role: RoleConfig


@dataclass
class ToolOutcome:
    observation: str
    error_kind: ErrorKind | None = None
    handoff_to: str | None = None
    note: str = ""
    submit: bool = False
    is_edit: bool = False
    ran_tests: bool = False


@dataclass(frozen=True)
class Param:
    type: type | tuple[type, ...]
    required: bool = True


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: dict[str, Param]
    fn: Callable[[ToolContext, dict[str, Any]], ToolOutcome] = field(compare=False)

    def check_args(self, args: dict[str, Any]) -> str | None:
        for k, p in self.params.items():
            if k not in args:
                if p.required:
                    return f"missing argument {k!r}"
                continue
            if not isinstance(args[k], p.type):
                return f"argument {k!r} has the wrong type"
        extra = set(args) - set(self.params)
        if extra:
            return f"unexpected argument(s): {', '.join(sorted(extra))}"
        return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} chars truncated] ...\n{text[-half:]}"


def _fail(msg: str) -> ToolOutcome:
    return ToolOutcome(msg, ErrorKind.TOOL_FAILED)


# --- environment tools ---------------------------------------------------------------


def _view(ctx: ToolContext, a: dict) -> ToolOutcome:
    path = a["path"]
    q = shlex.quote(path)
    r = ctx.sandbox.exec(f"if [ -d {q} ]; then find {q} -maxdepth 2 -not -path '*/.git*' | sort | head -200; "
                         f"else cat -n -- {q}; fi")  # fmt: skip
    if r.returncode != 0:
        return _fail(f"Cannot view {path}: {r.stderr.strip() or 'not found'}")
    out = r.stdout
    if "start_line" in a or "end_line" in a:
        lines = out.splitlines()
        s = max(int(a.get("start_line", 1)), 1)
        e = int(a.get("end_line", len(lines)))
        out = "\n".join(lines[s - 1 : e])
    return ToolOutcome(_truncate(out, ctx.config.observation_max_chars))


def _search(ctx: ToolContext, a: dict) -> ToolOutcome:
    path = a.get("path", ".")
    r = ctx.sandbox.exec(
        f"grep -rnI --exclude-dir=.git --exclude-dir=.swarmrl -e {shlex.quote(a['pattern'])} -- {shlex.quote(path)} | head -50"
    )
    return ToolOutcome(r.stdout or "No matches.")


def _str_replace(ctx: ToolContext, a: dict) -> ToolOutcome:
    path, old, new = a["path"], a["old"], a["new"]
    if is_protected(path, ctx.config.protected_paths):
        return _fail(f"{path} is a protected test/config path; edits there are discarded and zero the reward.")
    try:
        text = ctx.sandbox.read_file(path)
    except (FileNotFoundError, OSError):
        return _fail(f"{path} does not exist.")
    n = text.count(old)
    if n == 0:
        return _fail(f"`old` was not found in {path}. View the file and copy the exact text.")
    if n > 1:
        return _fail(f"`old` occurs {n} times in {path}; include more context so it is unique.")
    ctx.sandbox.write_file(path, text.replace(old, new, 1))
    return ToolOutcome(f"Edited {path}.", is_edit=True)


def _create_file(ctx: ToolContext, a: dict) -> ToolOutcome:
    path = a["path"]
    if is_protected(path, ctx.config.protected_paths):
        return _fail(f"{path} is a protected test/config path; edits there are discarded and zero the reward.")
    exists = ctx.sandbox.exec(f"test -e {shlex.quote(path)}").returncode == 0
    if exists and not a.get("overwrite", False):
        return _fail(f"{path} exists. Use str_replace, or pass overwrite=true.")
    ctx.sandbox.write_file(path, a["content"])
    return ToolOutcome(f"Wrote {path}.", is_edit=True)


def _run_tests(ctx: ToolContext, a: dict) -> ToolOutcome:
    targets = " ".join(shlex.quote(t) for t in a.get("targets", []))
    r = ctx.sandbox.exec(f"python -m pytest -q -p no:cacheprovider -x {targets}", timeout=600)
    head = "Tests timed out." if r.timed_out else f"pytest exit code {r.returncode}."
    return ToolOutcome(_truncate(f"{head}\n{r.output}", ctx.config.observation_max_chars), ran_tests=True)


def _bash(ctx: ToolContext, a: dict) -> ToolOutcome:
    cmd = a["command"]
    r = ctx.sandbox.exec(cmd, timeout=120)
    ran_tests = "pytest" in cmd
    head = "Command timed out." if r.timed_out else f"exit code {r.returncode}"
    return ToolOutcome(_truncate(f"{head}\n{r.output}", ctx.config.observation_max_chars), ran_tests=ran_tests)


def _git_diff(ctx: ToolContext, a: dict) -> ToolOutcome:
    r = ctx.sandbox.exec(f"git add -A && git diff --cached {BASELINE_TAG} -- . ':(exclude).swarmrl'")
    return ToolOutcome(_truncate(r.stdout or "No changes.", ctx.config.observation_max_chars))


# --- coordination tools -------------------------------------------------------------


def _handoff_to(ctx: ToolContext, target: str, note: str) -> ToolOutcome:
    if target not in ctx.role.handoff_targets:
        allowed = ", ".join(ctx.role.handoff_targets) or "none"
        return ToolOutcome(f"Cannot hand off to {target!r}. Allowed: {allowed}.", ErrorKind.BAD_ARGS)
    return ToolOutcome(f"Handed off to {target}.", handoff_to=target, note=note)


def _delegate(ctx: ToolContext, a: dict) -> ToolOutcome:
    return _handoff_to(ctx, a["to"], a["plan"])


def _handoff(ctx: ToolContext, a: dict) -> ToolOutcome:
    return _handoff_to(ctx, a["to"], a["note"])


def _send_back(ctx: ToolContext, a: dict) -> ToolOutcome:
    if "coder" not in ctx.config.role_names:
        return _fail("There is no coder to send back to.")
    return ToolOutcome("Sent back to coder.", handoff_to="coder", note=a["note"])


def _accept(ctx: ToolContext, a: dict) -> ToolOutcome:
    return ToolOutcome("Change accepted and submitted.", submit=True, note=a.get("note", ""))


def _submit(ctx: ToolContext, a: dict) -> ToolOutcome:
    return ToolOutcome("Submitted.", submit=True)


S = (str,)
TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("view", "Show a file with line numbers, or list a directory.",
             {"path": Param(S), "start_line": Param(int, False), "end_line": Param(int, False)}, _view),
        Tool("search", "grep the repository for a pattern.", {"pattern": Param(S), "path": Param(S, False)}, _search),
        Tool("str_replace", "Replace one exact, unique occurrence of `old` with `new` in a file.",
             {"path": Param(S), "old": Param(S), "new": Param(S)}, _str_replace),
        Tool("create_file", "Create a file (overwrite=true to replace one).",
             {"path": Param(S), "content": Param(S), "overwrite": Param(bool, False)}, _create_file),
        Tool("run_tests", "Run pytest, optionally on given targets.", {"targets": Param(list, False)}, _run_tests),
        Tool("bash", "Run a shell command in the repository (no network).", {"command": Param(S)}, _bash),
        Tool("git_diff", "Show the current change against the starting commit.", {}, _git_diff),
        Tool("delegate", "Planner: hand the plan to another role.", {"to": Param(S), "plan": Param(S)}, _delegate),
        Tool("handoff", "Hand control to another role with a note.", {"to": Param(S), "note": Param(S)}, _handoff),
        Tool("send_back", "Reviewer: return the change to the coder with the unmet requirement.",
             {"note": Param(S)}, _send_back),
        Tool("accept", "Reviewer: accept and submit the change.", {"note": Param(S, False)}, _accept),
        Tool("submit", "Single agent: submit the change.", {}, _submit),
    ]
}  # fmt: skip


def execute(call, ctx: ToolContext) -> ToolOutcome:
    tool = TOOLS[call.name]
    err = tool.check_args(call.arguments)
    if err:
        return ToolOutcome(f"Bad arguments for {call.name}: {err}.", ErrorKind.BAD_ARGS)
    return tool.fn(ctx, call.arguments)


def describe_tools(names: tuple[str, ...]) -> str:
    lines = []
    for n in names:
        t = TOOLS[n]
        params = ", ".join(f"{k}{'' if p.required else '?'}" for k, p in t.params.items())
        lines.append(f"- {n}({params}): {t.description}")
    return "\n".join(lines)
