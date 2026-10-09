from swarmrl.swarm.config import default_swarm_config
from swarmrl.swarm.harness import Harness
from swarmrl.swarm.parser import ParseError, parse_tool_call
from swarmrl.traces.schema import ErrorKind, Termination


def test_parse_errors():
    tools = ("view", "delegate")
    assert parse_tool_call("hi", tools).kind is ErrorKind.NO_TOOL_CALL
    assert parse_tool_call("<tool_call>{nope</tool_call>", tools).kind is ErrorKind.BAD_JSON
    assert (
        parse_tool_call('<tool_call>{"name": "bash", "arguments": {}}</tool_call>', tools).kind
        is ErrorKind.UNKNOWN_TOOL
    )
    assert parse_tool_call('<tool_call>{"name": "view", "arguments": []}</tool_call>', tools).kind is ErrorKind.BAD_ARGS
    ok = parse_tool_call('x <tool_call>{"name": "view", "arguments": {"path": "a"}}</tool_call>', tools)
    assert not isinstance(ok, ParseError) and ok.arguments == {"path": "a"}


def test_budgets_and_warning():
    cfg = default_swarm_config()
    h = Harness(cfg)
    budget = cfg.role("planner").turn_budget
    for _ in range(budget - cfg.warn_turns_remaining):
        h.post_turn("planner", 10)
    assert h.warning("planner") is not None
    for _ in range(cfg.warn_turns_remaining):
        h.post_turn("planner", 10)
    assert h.pre_turn("planner", 100)[0] is Termination.TURN_BUDGET
    assert h.pre_turn("planner", cfg.context_limit - 5)[0] is Termination.TURN_BUDGET
    h2 = Harness(cfg)
    assert h2.pre_turn("coder", cfg.context_limit - 5)[0] is Termination.CONTEXT_LIMIT
    term, max_new = h2.pre_turn("coder", 1000)
    assert term is None and max_new == cfg.sampling.max_new_tokens


def test_fingerprint_ignores_sampling_but_not_prompts():
    from dataclasses import replace

    cfg = default_swarm_config()
    assert cfg.fingerprint() == cfg.with_sampling(temperature=0.8).fingerprint()
    assert cfg.fingerprint() != replace(cfg, episode_turn_budget=41).fingerprint()
