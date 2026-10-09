import pytest

from swarmrl.evals.compare import FreezeTracker, compare_runs
from swarmrl.evals.metrics import classify_failure, compute_metrics, pass_at_k, wilson
from swarmrl.traces.schema import Episode, GradeSummary, Termination, Transition


def ep(task, resolved, term=Termination.SUBMIT, transitions=(), diff="x", protected=()):
    e = Episode(
        "e" + str(id(object())),
        task,
        "dev",
        "fp",
        "ck",
        0,
        term,
        float(resolved),
        GradeSummary(resolved=resolved),
        list(protected),
        diff,
    )
    e.transitions = list(transitions)
    return e


def tr(role, name, **args):
    return Transition("e", role, 0, 0, [1], [2], [-0.1], {"name": name, "arguments": args})


def test_wilson_and_pass_at_k():
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert pass_at_k(8, 0, 1) == 0.0 and pass_at_k(8, 8, 4) == 1.0
    assert pass_at_k(8, 2, 1) == pytest.approx(0.25)


def test_metrics_process_and_taxonomy():
    good = ep("t1", True, transitions=[tr("coder", "str_replace"), tr("tester", "run_tests"), tr("reviewer", "accept")])
    stamp = ep(
        "t1", False, transitions=[tr("coder", "str_replace"), tr("tester", "run_tests"), tr("reviewer", "accept")]
    )
    hack = ep("t2", False, protected=["tests/x.py"])
    infra = ep("t2", False, term=Termination.INFRA_ERROR)
    m = compute_metrics([good, stamp, hack, infra], k_values=(1,))
    assert m["episodes"] == 3 and m["infra_errors"] == 1
    assert m["resolve_rate"] == pytest.approx(1 / 3)
    assert m["tester_ran_tests_after_last_edit"] == pytest.approx(2 / 3)
    assert m["reviewer_agreement"] == pytest.approx(0.5)
    assert m["failures"] == {"reviewer_rubber_stamped": 1, "hack_attempt": 1}
    assert (
        classify_failure(
            ep("t", False, transitions=[tr("planner", "delegate", plan="look at utils.py")]), ["stats/core.py"]
        )
        == "wrong_file_planned"
    )


def test_compare_runs_keep_rule():
    a = [ep(f"t{i}", r) for i in range(10) for r in (True, False, False, False)]
    b = [ep(f"t{i}", r) for i in range(10) for r in (True, True, True, False)]
    c = compare_runs(a, b)
    assert c.keep and c.tasks_improved == 10
    flat = compare_runs(a, a)
    assert not flat.keep
    f = FreezeTracker(2)
    assert not f.record(c) and not f.record(flat) and f.record(flat)
