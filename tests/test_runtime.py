import asyncio

from conftest import call

from swarmrl.sandbox.base import InfraError
from swarmrl.swarm.config import single_agent_config
from swarmrl.swarm.dispatcher import Dispatcher, EpisodeRequest
from swarmrl.tasks.oracle_agent import oracle_file_states, oracle_script
from swarmrl.traces.schema import Episode, Termination
from swarmrl.traces.store import TraceStore


def _oracle_runner(make_runner, task, factory, config=None):
    from swarmrl.swarm.config import default_swarm_config

    cfg = config or default_swarm_config()
    return make_runner(oracle_script(oracle_file_states(task, factory), cfg), cfg)


def test_oracle_swarm_resolves(make_runner, task, factory):
    ep = asyncio.run(_oracle_runner(make_runner, task, factory).run(task))
    assert ep.termination is Termination.SUBMIT
    assert ep.resolved and ep.team_reward == 1.0
    assert [s.role for s in ep.spans] == ["planner", "coder", "tester", "reviewer"]
    assert [s.handoff_reason for s in ep.spans] == ["delegate", "handoff", "handoff", "submit"]


def test_oracle_single_agent_resolves(make_runner, task, factory):
    ep = asyncio.run(_oracle_runner(make_runner, task, factory, single_agent_config()).run(task))
    assert ep.resolved and ep.team_reward == 1.0


def test_role_contexts_grow_by_prefix(make_runner, task, factory):
    """Every role's next input = previous input + output + closing + new turn: never re-tokenized."""
    ep = asyncio.run(_oracle_runner(make_runner, task, factory).run(task))
    last = {}
    for t in ep.transitions:
        if t.role in last:
            prev = last[t.role]
            head = prev.input_ids + prev.output_ids + prev.closing_ids
            assert t.input_ids[: len(head)] == head
        last[t.role] = t
        assert len(t.output_ids) == len(t.rollout_logprobs)


def test_hack_attempt_zeroes_reward(make_runner, task):
    fix = "def mean(values):\n    return sum(values) / len(values) if values else 0.0\n"

    def script(role, turn, ctx):
        if role == "planner":
            return call("delegate", to="coder", plan="edit")
        if role == "coder":
            return [
                call(
                    "str_replace",
                    path="stats/core.py",
                    old="    return sum(values) / len(values)",
                    new="    return sum(values) / len(values) if values else 0.0",
                ),
                call("str_replace", path="tests/test_core.py", old="x", new="y"),  # refused by the tool
                call("handoff", to="tester", note="done"),
            ][turn]
        if role == "tester":
            # bash can still write to tests: the grader drops it and zeroes the reward
            return [
                call("bash", command="echo 'def test_x(): pass' > tests/test_extra.py"),
                call("handoff", to="reviewer", note="ok"),
            ][turn]
        return call("accept")

    ep = asyncio.run(make_runner(script).run(task))
    assert ep.transitions[2].error_kind == "tool_failed"
    assert ep.protected_edits == ["tests/test_extra.py"]
    assert ep.team_reward == 0.0
    assert fix  # the partial source fix is irrelevant once a hack is attempted


def test_three_format_errors_end_episode(make_runner, task):
    ep = asyncio.run(make_runner(lambda r, t, c: "no tool call here").run(task))
    assert ep.termination is Termination.FORMAT_ERRORS
    assert len(ep.transitions) == 3
    assert ep.team_reward == 0.0


def test_corrective_observation_then_recovery(make_runner, task):
    replies = [
        "<tool_call>{bad json</tool_call>",
        call("view", path="stats/core.py"),
        call("delegate", to="coder", plan="x"),
    ]

    def script(role, turn, ctx):
        if role == "planner":
            return replies[turn]
        return call("view", path=".") if role != "reviewer" else call("accept")

    runner = make_runner(script)
    ep = asyncio.run(runner.run(task))
    assert ep.transitions[0].error_kind == "bad_json"
    assert "not valid JSON" in ep.transitions[0].observation
    assert ep.transitions[1].error_kind is None


def test_repeat_call_detected(make_runner, task):
    def script(role, turn, ctx):
        return call("view", path="stats/core.py")

    ep = asyncio.run(make_runner(script).run(task))
    assert ep.repeat_calls > 0
    assert any(t.error_kind == "repeat_call" for t in ep.transitions)
    assert ep.termination is Termination.TURN_BUDGET


def test_ping_pong_terminates(make_runner, task):
    def script(role, turn, ctx):
        if role == "planner":
            return call("delegate", to="coder", plan="go")
        if role == "coder":
            return call("handoff", to="tester", note=f"try {turn}")
        return call("handoff", to="coder", note=f"fails {turn}")

    ep = asyncio.run(make_runner(script).run(task))
    assert ep.termination is Termination.PING_PONG
    assert ep.ping_pong_events == 1


def test_infra_error_is_not_scored(make_runner, task):
    class Broken:
        def create(self, task):
            raise InfraError("container failed to start")

    ep = asyncio.run(make_runner(lambda r, t, c: "", sandbox_factory=Broken()).run(task))
    assert ep.termination is Termination.INFRA_ERROR
    assert ep.team_reward is None and not ep.scored


def test_dispatcher_streams_to_store(make_runner, task, factory, tmp_path):
    runner = _oracle_runner(make_runner, task, factory)
    store = TraceStore(tmp_path / "traces")
    eps = Dispatcher(runner, on_episode=store.put).run_sync([EpisodeRequest(task, s) for s in range(3)])
    assert all(e.resolved for e in eps)
    assert len(store.index()) == 3
    back = store.get(eps[0].episode_id)
    assert isinstance(back, Episode) and back.to_dict() == eps[0].to_dict()


class _FlakyPolicy:
    """Raises a non-infra bug for one episode, sleeps forever for another, delegates otherwise."""

    def __init__(self, inner):
        self.inner = inner
        self.seen: dict[str, int] = {}

    async def generate(self, input_ids, sampling, stop_ids, session_key, max_new_tokens):
        ep = session_key.split(":")[0]
        idx = self.seen.setdefault(ep, len(self.seen))
        if idx == 0:
            raise RuntimeError("bug in a tool parser")
        if idx == 1:
            await asyncio.sleep(3600)
        return await self.inner.generate(input_ids, sampling, stop_ids, session_key, max_new_tokens)


def test_batch_survives_internal_errors_and_timeouts(make_runner, task, factory):
    from swarmrl.swarm.dispatcher import DispatchLimits

    runner = _oracle_runner(make_runner, task, factory)
    runner.policy = _FlakyPolicy(runner.policy)
    d = Dispatcher(runner, DispatchLimits(episode_timeout_s=10))
    eps = d.run_sync([EpisodeRequest(task, s) for s in range(4)])
    by_err = [e.metadata.get("infra_error", "ok") for e in eps]
    assert sum(e.resolved for e in eps) == 2
    assert any("RuntimeError" in x for x in by_err) and any("wall clock" in x for x in by_err)
    bug = next(e for e in eps if "RuntimeError" in e.metadata.get("infra_error", ""))
    assert bug.metadata["internal_error"] is True and not bug.scored


def test_invalid_task_is_never_graded(make_runner, task, factory, record):
    from dataclasses import replace

    runner = _oracle_runner(make_runner, task, factory)
    runner.validation = lambda t: replace(record, valid=False, reason="no failing test")
    ep = asyncio.run(runner.run(task))
    assert ep.termination is Termination.INFRA_ERROR and "failed validation" in ep.metadata["infra_error"]
