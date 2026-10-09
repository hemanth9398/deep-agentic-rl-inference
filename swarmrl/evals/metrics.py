"""Metrics on every report: outcome, format, process, swarm health, failure taxonomy, cost."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from swarmrl.traces.schema import Episode, ErrorKind, Termination

EDIT_TOOLS = ("str_replace", "create_file")
HANDOFF_TOOLS = ("delegate", "handoff", "send_back")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c successes (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def scored(episodes: Iterable[Episode]) -> list[Episode]:
    """Infrastructure errors are never scored against the model."""
    return [e for e in episodes if e.termination is not Termination.INFRA_ERROR]


def per_task(episodes: Sequence[Episode]) -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in scored(episodes):
        out[e.task_id][0] += int(e.resolved)
        out[e.task_id][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


# --- process ------------------------------------------------------------------------


def _is_edit(t) -> bool:
    return bool(t.tool_call) and t.tool_call["name"] in EDIT_TOOLS and t.error_kind is None


def _ran_tests(t) -> bool:
    if not t.tool_call or t.error_kind is not None:
        return False
    name = t.tool_call["name"]
    return name == "run_tests" or (name == "bash" and "pytest" in str(t.tool_call["arguments"].get("command", "")))


def tester_ran_tests_after_last_edit(ep: Episode, tester_roles: tuple[str, ...] = ("tester", "agent")) -> bool:
    last_edit = max((i for i, t in enumerate(ep.transitions) if _is_edit(t)), default=None)
    if last_edit is None:
        return False
    return any(_ran_tests(t) and t.role in tester_roles for t in ep.transitions[last_edit + 1 :])


def reviewer_verdict(ep: Episode) -> str | None:
    for t in reversed(ep.transitions):
        if t.role == "reviewer" and t.tool_call and t.error_kind is None and t.tool_call["name"] in ("accept", "send_back"):
            return t.tool_call["name"]
    return None


def handoffs(ep: Episode) -> int:
    return sum(1 for t in ep.transitions if t.tool_call and t.error_kind is None and t.tool_call["name"] in HANDOFF_TOOLS)


# --- failure taxonomy -----------------------------------------------------------------


def classify_failure(ep: Episode, oracle_files: Sequence[str] | None = None) -> str | None:
    """Per-role failure label for an unresolved, scored episode (None if resolved or unscored)."""
    if ep.termination is Termination.INFRA_ERROR or ep.resolved:
        return None
    if ep.protected_edits:
        return "hack_attempt"
    if ep.termination is Termination.FORMAT_ERRORS:
        return "format_errors"
    if ep.termination is Termination.PING_PONG:
        return "role_ping_pong"
    edits = [t for t in ep.transitions if t.tool_call and t.tool_call["name"] in EDIT_TOOLS]
    if oracle_files:
        plans = " ".join(
            str(t.tool_call["arguments"].get("plan", "")) for t in ep.transitions
            if t.role == "planner" and t.tool_call and t.tool_call["name"] == "delegate"
        )  # fmt: skip
        if plans and not any(f in plans or f.rsplit("/", 1)[-1] in plans for f in oracle_files):
            return "wrong_file_planned"
    if edits and not any(_is_edit(t) for t in edits):
        return "edit_failed"
    if not ep.final_diff.strip():
        return "max_turns" if ep.termination is not Termination.SUBMIT else "no_change"
    if any(t.role == "tester" for t in ep.transitions) and not any(
        t.role == "tester" and (_ran_tests(t) or (t.tool_call and t.tool_call["name"] == "bash")) for t in ep.transitions
    ):
        return "bug_never_reproduced"
    if reviewer_verdict(ep) == "accept":
        return "reviewer_rubber_stamped"
    if ep.termination in (Termination.TURN_BUDGET, Termination.TOKEN_BUDGET, Termination.CONTEXT_LIMIT):
        return "max_turns"
    return "wrong_fix"


# --- aggregate --------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute_metrics(
    episodes: Sequence[Episode], k_values: Sequence[int] = (1, 4, 8), oracle_files: dict[str, list[str]] | None = None
) -> dict:
    eps = scored(episodes)
    n, k = len(eps), sum(e.resolved for e in eps)
    tasks = per_task(eps)
    m: dict = {
        "episodes": n,
        "infra_errors": len(episodes) - n,
        "resolve_rate": k / n if n else 0.0,
        "resolve_ci95": wilson(k, n),
        "per_task": {t: c / m_ for t, (c, m_) in sorted(tasks.items())},
    }
    for kk in k_values:
        vals = [pass_at_k(m_, c, kk) for c, m_ in tasks.values() if m_ >= kk]
        if vals:
            m[f"pass@{kk}"] = _mean(vals)

    # format and tool errors per turn, by role
    turns: Counter[str] = Counter()
    fmt: Counter[str] = Counter()
    tool: Counter[str] = Counter()
    for e in eps:
        for t in e.transitions:
            turns[t.role] += 1
            if t.error_kind:
                if ErrorKind(t.error_kind).is_format_error:
                    fmt[t.role] += 1
                else:
                    tool[t.role] += 1
    m["format_error_rate"] = {r: fmt[r] / turns[r] for r in sorted(turns)}
    m["tool_error_rate"] = {r: tool[r] / turns[r] for r in sorted(turns)}

    # process
    m["tester_ran_tests_after_last_edit"] = _mean([float(tester_ran_tests_after_last_edit(e)) for e in eps])
    verdicts = [(reviewer_verdict(e), e.resolved) for e in eps]
    judged = [(v, r) for v, r in verdicts if v is not None]
    m["reviewer_agreement"] = _mean([float((v == "accept") == r) for v, r in judged]) if judged else None

    # swarm health
    m["handoffs_per_episode"] = _mean([handoffs(e) for e in eps])
    m["ping_pong_rate"] = _mean([float(e.ping_pong_events > 0) for e in eps])
    m["repeat_call_rate"] = _mean([float(e.repeat_calls > 0) for e in eps])
    role_tokens: dict[str, list[int]] = defaultdict(list)
    for e in eps:
        for r, c in e.tokens_by_role().items():
            role_tokens[r].append(c)
    m["tokens_per_role"] = {r: _mean(v) for r, v in sorted(role_tokens.items())}

    # failure taxonomy
    of = oracle_files or {}
    m["failures"] = dict(Counter(f for e in eps if (f := classify_failure(e, of.get(e.task_id)))))
    m["terminations"] = dict(Counter(e.termination.value for e in episodes))

    # cost
    m["cost"] = {
        "turns": _mean([e.num_turns for e in eps]),
        "tokens": _mean([sum(e.tokens_by_role().values()) for e in eps]),
        "wall_time_s": _mean([e.wall_time_s for e in eps]),
        "peak_context": max((e.peak_context() for e in eps), default=0),
    }
    return m
