from swarmrl.grading.protected import filter_diff, is_protected
from swarmrl.swarm.config import default_swarm_config
from swarmrl.tasks.harbor import load_tasks
from swarmrl.tasks.mutants import apply_mutation, find_mutations, generate_mutant_tasks
from swarmrl.tasks.splits import assign_repo_splits, contamination
from swarmrl.tasks.validate import TaskValidator, ValidationCache

from conftest import EXAMPLE


def test_example_task_validates(task, record):
    assert record.fail_to_pass == [
        "tests.test_core::test_mean_empty_returns_zero",
        "tests.test_core::test_median_empty_returns_zero",
    ]
    assert len(record.pass_to_pass) == 2


def test_validation_cache_by_content_hash(task, factory, tmp_path):
    cache = ValidationCache(tmp_path / "v.json")
    v = TaskValidator(factory, cache)
    first = v.validate(task)
    assert cache.get(task.content_hash()) == first
    (task.task_dir / "instruction.md").write_text("changed")
    assert cache.get(task.content_hash()) is None


def test_task_without_failing_test_is_rejected(task, factory):
    (task.task_dir / "solution" / "solve.patch").write_text("")
    assert not TaskValidator(factory).validate(task).valid


def test_broken_oracle_is_rejected(task, factory):
    p = task.task_dir / "solution" / "solve.patch"
    p.write_text(p.read_text().replace("return 0.0\n     return sum", "return 1.0\n     return sum"))
    rec = TaskValidator(factory).validate(task)
    assert not rec.valid


def test_protected_paths():
    pats = default_swarm_config().protected_paths
    assert is_protected("tests/test_core.py", pats)
    assert is_protected("pkg/conftest.py", pats)
    assert is_protected("conftest.py", pats)
    assert is_protected("pytest.ini", pats)
    assert not is_protected("stats/core.py", pats)
    diff = (
        "diff --git a/stats/core.py b/stats/core.py\n--- a/stats/core.py\n+++ b/stats/core.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/tests/test_core.py b/tests/test_core.py\n--- a/tests/test_core.py\n+++ b/tests/test_core.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    kept, dropped = filter_diff(diff, pats)
    assert "stats/core.py" in kept and "tests/test_core.py" not in kept
    assert dropped == ["tests/test_core.py"]


def test_contamination_and_repo_splits(task):
    assert contamination([task]) == []

    class T:
        def __init__(self, split, repo):
            self.split, self.repo = split, repo

    assert contamination([T("heldout", "a"), T("train", "a")]) == ["repo 'a' is in heldout and in train/dev"]
    s = assign_repo_splits(["a", "b", "c", "d", "e"], 0.2, seed=1)
    assert list(s.values()).count("heldout") == 1


def test_mutation_roundtrip():
    src = "def f(a, b):\n    return a < b and a + 1 > 0\n"
    muts = find_mutations("m.py", src)
    kinds = {(m.old, m.new) for m in muts}
    assert ("<", "<=") in kinds and ("and", "or") in kinds and ("+", "-") in kinds
    for m in muts:
        out = apply_mutation(src, m)
        assert out != src
        compile(out, "m.py", "exec")


def test_generate_mutant_tasks_are_valid(tmp_path, factory):
    out = tmp_path / "mutants"
    paths = generate_mutant_tasks(EXAMPLE / "environment" / "repo", "tests", ["tests/test_core.py"], out, "stats",
                                  default_swarm_config(), max_tasks=2)
    assert paths
    v = TaskValidator(factory)
    for t in load_tasks(out):
        rec = v.validate(t)
        assert rec.valid, rec.reason
        assert t.source == "procedural-mutant" and t.split == "train"
        assert "Do not modify the tests" in t.instruction
