"""v2 episodes on the real example task: probes, gating, shadows, coder patches, router ranking, hidden grading."""

import asyncio
import math
import random
import re

import pytest

from swarmrl.grading.grader import CleanRoomGrader
from swarmrl.inference.chat import ByteTokenizer, ChatFormat
from swarmrl.inference.client import Generation, PolicyClient
from swarmrl.swarm.config import default_swarm_config
from swarmrl.tasks.oracle_agent import oracle_file_states
from swarmrl.v2 import protocol as P
from swarmrl.v2.credit import decision_stats, planner_targets, router_targets
from swarmrl.v2.router import Gate, GateConfig
from swarmrl.v2.runtime import RoleModel, TriConfig, TriRunner

GOOD_PLAN = """<hypothesis>mean and median in stats/core.py do not handle an empty list</hypothesis>
<files>stats/core.py</files>
<steps>return 0.0 early from mean and median when values is empty</steps>
<probe>python -c "import stats.core as c; c.mean([])"</probe>
<expect>fail</expect>"""

BAD_PLAN = """<hypothesis>the package __init__ hides the functions</hypothesis>
<files>stats/__init__.py</files>
<steps>re-export mean and median</steps>
<probe>python -c "import stats"</probe>
<expect>fail</expect>"""


class FnPolicy(PolicyClient):
    """Replies chosen by fn(session_key, prompt_text) -> (reply, first_token_logprob). Records every prompt."""

    def __init__(self, chat, fn):
        self.chat, self.fn, self.prompts = chat, fn, []

    async def generate(self, input_ids, sampling, stop_ids, session_key, max_new_tokens) -> Generation:
        text = self.chat.decode(input_ids)
        self.prompts.append((session_key, text))
        reply, lp0 = self.fn(session_key, text)
        ids = self.chat.tok.encode(reply) + [self.chat.end_of_turn_id]
        return Generation(ids, [lp0] + [-0.1] * (len(ids) - 1), "stop")


@pytest.fixture
def fixed_core(task, factory):
    return oracle_file_states(task, factory)["stats/core.py"]


def make_runner(task, factory, record, fixed_core, plan_router_p, gate_cfg, plans=("good", "bad", "junk")):
    chat = ChatFormat(ByteTokenizer())

    def planner(key, text):
        k = int(re.search(r"-r\d+p(\d+):", key).group(1))
        kind = plans[k % len(plans)]
        return {"good": GOOD_PLAN, "bad": BAD_PLAN, "junk": "I would refactor everything."}[kind], -0.5

    def coder(key, text):
        j = int(re.search(r"c(\d+):coder", key).group(1))
        if "stats/core.py" in text and j % 2 == 0:
            return f'<file path="stats/core.py">\n{fixed_core}</file>', -0.3
        if "stats/core.py" in text:
            return '<file path="stats/core.py">\ndef mean(v):\n    return sum(v) / len(v)\n</file>', -0.3
        return '<file path="stats/__init__.py">\nfrom .core import mean, median\n</file>', -0.3

    def router(key, text):
        if "Patch:" in text:  # patch verdict: trust the visible checks
            ok = "probe now exits 0; visible tests exit 0" in text
            return ("YES the checks pass", math.log(0.9)) if ok else ("NO the checks fail", math.log(0.9))
        p = plan_router_p("stats/core.py" in text)
        return ("YES looks right", math.log(p)) if p >= 0.5 else ("NO the hypothesis is not supported",
                                                                     math.log(1 - p))  # fmt: skip

    roles = {n: FnPolicy(chat, f) for n, f in (("planner", planner), ("coder", coder), ("router", router))}
    cfg = TriConfig(plans_per_round=len(plans), max_rounds=2, patches_per_plan=2, gate=gate_cfg)
    runner = TriRunner(*(RoleModel(roles[n], chat, 4096) for n in ("planner", "coder", "router")), factory,
                       CleanRoomGrader(factory, default_swarm_config().protected_paths), lambda t: record, cfg,
                       Gate(gate_cfg, random.Random(0)))  # fmt: skip
    return runner, roles


def test_parsers_and_verdict_probability():
    p = P.parse_plan(GOOD_PLAN)
    assert p.valid and p.files == ["stats/core.py"] and p.expect == "fail"
    assert not P.parse_plan("<hypothesis>x</hypothesis>").valid
    assert not P.parse_plan(GOOD_PLAN.replace("stats/core.py", "../etc/passwd")).valid
    assert P.parse_patch('<file path="a.py">\nx = 1\n</file>') == ({"a.py": "x = 1\n"}, "")
    assert P.parse_patch("nothing")[1]
    chat = ChatFormat(ByteTokenizer())
    vt = P.VerdictTokens.for_chat(chat)
    yes = Generation(chat.tok.encode("YES ok"), [math.log(0.9), 0, 0, 0, 0, 0], "stop")
    no = Generation(chat.tok.encode("NO"), [math.log(0.8), 0], "stop")
    assert P.verdict_probability(yes, vt) == (pytest.approx(0.9), True)
    assert P.verdict_probability(no, vt) == (pytest.approx(0.2), True)
    assert P.verdict_probability(Generation(chat.tok.encode("maybe"), [0] * 5, "stop"), vt) == (0.0, False)


def test_router_accepts_the_good_plan_and_the_best_patch_is_submitted(task, factory, record, fixed_core):
    runner, roles = make_runner(task, factory, record, fixed_core, lambda good: 0.9 if good else 0.1,
                                GateConfig(epsilon=0.0, temperature=0.01, min_temperature=0.01))  # fmt: skip
    ep = asyncio.run(runner.run(task, seed=1, group_id="g"))
    assert ep.error is None and len(ep.rounds) == 1
    rd = ep.rounds[0]
    good, bad, junk = rd.plans
    assert good.probe_result["confirmed"] is True and bad.probe_result["confirmed"] is False  # sandbox evidence
    assert not junk.valid and not rd.gate.executed[2]
    assert rd.gate.main == [0] and not rd.gate.executed[1]
    assert [p.y for p in good.patches] == [1.0, 0.0] and good.value == 0.5  # fixed vs broken patch, hidden tests
    assert ep.final == (0, 0, 0) and ep.reward == 1.0  # the router ranked the passing patch first
    assert ep.tokens["planner"] > 0 and ep.tokens["coder"] > 0 and ep.tokens["router"] > 0


def test_a_wrongly_rejected_good_plan_is_caught_by_its_shadow(task, factory, record, fixed_core):
    # the router is wrong both ways: it rejects the good plan and accepts the bad one; eps = 1 runs everything
    runner, _ = make_runner(task, factory, record, fixed_core, lambda good: 0.05 if good else 0.9,
                            GateConfig(epsilon=1.0), plans=("good", "bad"))  # fmt: skip
    ep = asyncio.run(runner.run(task, seed=2, group_id="g"))
    rd = ep.rounds[0]
    assert rd.gate.main == [1] and rd.gate.shadows == [0]
    assert ep.reward == 0.0  # the submitted answer came from the bad plan
    assert rd.plans[0].value == 0.5 and rd.plans[1].value == 0.0  # but the shadow shows the good idea works
    adv = {t.call.text: t.advantage for t in planner_targets([ep])}
    assert adv[rd.plans[0].call.text] > 0 > adv[rd.plans[1].call.text]  # planner credited by real outcomes
    labels = {t.call.text: t.label for t in router_targets([ep]) if t.kind == "plan"}
    assert labels[rd.plans[0].verdict.call.text] == 0.5  # the router learns what its rejection was worth
    assert decision_stats([ep])["router/false_reject_rate"] == 1.0


def test_rejections_go_back_to_the_planner_with_reasons(task, factory, record, fixed_core):
    runner, roles = make_runner(task, factory, record, fixed_core, lambda good: 0.02,
                                GateConfig(epsilon=0.0, temperature=0.01, min_temperature=0.01))  # fmt: skip
    ep = asyncio.run(runner.run(task, seed=3, group_id="g"))
    assert len(ep.rounds) == 2 and not any(ep.rounds[0].gate.executed)
    assert ep.rounds[1].gate.forced  # last round: the best plan runs regardless
    second = [t for k, t in roles["planner"].prompts if "-r1p" in k]
    assert second and "rejected (p=0.02)" in second[0] and "router: the hypothesis is not supported" in second[0]
    assert "contradicts the hypothesis" in second[0]  # the probe evidence travels with the rejection
