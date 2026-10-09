from swarmrl.lifecycle.gates import DeployCandidate, choose_deployment, gate_environments, gate_rl0, gate_rl2, gate_sft
from swarmrl.rl.loop import Stage0Verdict


def m(rate, lo, hi, fmt=0.1):
    return {"resolve_rate": rate, "resolve_ci95": (lo, hi), "format_error_rate": {"coder": fmt}}


def test_gates():
    assert gate_environments([True, True]).next_stage == "harness"
    g = gate_environments([True, False])
    assert not g.passed and g.next_stage == "environments"
    assert gate_sft(m(0.4, 0.35, 0.45, 0.02), m(0.2, 0.15, 0.25, 0.1)).passed
    g = gate_sft(m(0.4, 0.35, 0.45, 0.09), m(0.2, 0.15, 0.25, 0.1))
    assert not g.passed and g.next_stage == "harness"  # back one stage
    assert not gate_rl0(Stage0Verdict(False, ["x"])).passed
    assert not gate_rl2(m(0.6, 0.55, 0.65), m(0.4, 0.35, 0.45), [1e-4] * 4 + [1e-3] * 4).passed


def test_choose_deployment():
    c = [DeployCandidate("swarm", 0.62, 40_000), DeployCandidate("single_agent", 0.60, 12_000)]
    assert choose_deployment(c, budget=50_000).name == "swarm"
    assert choose_deployment(c, budget=20_000).name == "single_agent"
