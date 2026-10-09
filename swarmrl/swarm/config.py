"""Swarm configuration and its fingerprint.

One config drives evals, data collection, RL and deployment. The fingerprint covers
everything that changes the protocol the model sees (prompts, tools, budgets, parser),
and excludes sampling settings, so rejection sampling at temperature 0.8 and RL at 1.0
produce data with the same fingerprint as the deployed swarm.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

TOOL_CALL_FORMAT = (
    "Reply with brief reasoning, then exactly one tool call in this form:\n"
    '<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>'
)


@dataclass(frozen=True)
class RoleConfig:
    name: str
    system_prompt: str
    tools: tuple[str, ...]
    handoff_targets: tuple[str, ...] = ()
    turn_budget: int = 10


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0  # RL: 1.0 / 1.0 so the trainer scores the distribution SGLang sampled
    top_p: float = 1.0
    max_new_tokens: int = 2048


@dataclass(frozen=True)
class SwarmConfig:
    roles: tuple[RoleConfig, ...]
    entry_role: str = "planner"
    episode_turn_budget: int = 40
    episode_token_budget: int = 48_000  # generated tokens across all roles
    context_limit: int = 32_768  # per-role context (input + new tokens)
    warn_turns_remaining: int = 3
    max_consecutive_format_errors: int = 3
    max_handoffs_per_pair: int = 6  # more hand-offs than this between two roles = ping-pong
    repeat_call_window: int = 4  # identical call within the last N calls of a role = repeat
    observation_max_chars: int = 8_000
    protected_paths: tuple[str, ...] = (
        "tests/**",
        "test/**",
        "**/test_*.py",
        "**/*_test.py",
        "**/conftest.py",
        "conftest.py",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        ".github/**",
    )
    parser_version: str = "tool_call_v1"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def role(self, name: str) -> RoleConfig:
        for r in self.roles:
            if r.name == name:
                return r
        raise KeyError(name)

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.roles)

    def fingerprint(self) -> str:
        from swarmrl.swarm.tools import describe_tools

        d = asdict(self)
        d.pop("sampling")
        d["tool_specs"] = {r.name: describe_tools(r.tools) for r in self.roles}
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def with_sampling(self, **kw: float) -> SwarmConfig:
        from dataclasses import replace

        return replace(self, sampling=replace(self.sampling, **kw))


_PLANNER = f"""You are the Planner in a four-role software team (Planner, Coder, Tester, Reviewer).
Read the issue and explore the repository. Identify the files to change and the approach.
Do not edit files. When you know the plan, call `delegate` with to="coder" and a concise plan
naming files, functions and the intended change.
{TOOL_CALL_FORMAT}"""

_CODER = f"""You are the Coder in a four-role software team.
Make the minimal source change that fixes the issue. Never edit tests or test configuration:
those edits are discarded at grading. When the fix is in place, call `handoff` with to="tester"
and say what you changed.
{TOOL_CALL_FORMAT}"""

_TESTER = f"""You are the Tester in a four-role software team.
Reproduce the bug, check edge cases, and run the test suite against the Coder's change.
If something fails, call `handoff` with to="coder" and the failing output. If everything
passes, call `handoff` with to="reviewer" and summarise the evidence. Put reproduction
scripts under /tmp, never in the repository's tests: edits to tests zero the team reward.
{TOOL_CALL_FORMAT}"""

_REVIEWER = f"""You are the Reviewer in a four-role software team.
Check the diff against every requirement in the issue. Call `accept` to submit the change,
or `send_back` with the unmet requirement to return it to the Coder.
{TOOL_CALL_FORMAT}"""


def default_swarm_config() -> SwarmConfig:
    return SwarmConfig(
        roles=(
            RoleConfig("planner", _PLANNER, ("view", "search", "delegate"), ("coder",), turn_budget=8),
            RoleConfig(
                "coder",
                _CODER,
                ("view", "search", "str_replace", "create_file", "run_tests", "handoff"),
                ("tester",),
                turn_budget=15,
            ),
            RoleConfig(
                "tester", _TESTER, ("view", "bash", "run_tests", "handoff"), ("coder", "reviewer"), turn_budget=10
            ),
            RoleConfig("reviewer", _REVIEWER, ("view", "git_diff", "accept", "send_back"), (), turn_budget=6),
        ),
    )


_SINGLE = f"""You are a software engineer. Fix the issue with the minimal source change.
Never edit tests or test configuration. Reproduce the bug, fix it, run the tests, then call
`submit`.
{TOOL_CALL_FORMAT}"""


def single_agent_config() -> SwarmConfig:
    """The deployment-time comparison candidate: one role with every tool."""
    return SwarmConfig(
        roles=(
            RoleConfig(
                "agent",
                _SINGLE,
                ("view", "search", "str_replace", "create_file", "bash", "run_tests", "git_diff", "submit"),
                (),
                turn_budget=40,
            ),
        ),
        entry_role="agent",
    )
