import asyncio

from conftest import call

from swarmrl.sft.filters import FilterConfig, filter_episodes
from swarmrl.sft.rejection import AdaptiveSampler, cap_teacher
from swarmrl.sft.samples import IGNORE, build_sft_samples, preference_pairs
from swarmrl.swarm.config import default_swarm_config
from swarmrl.tasks.oracle_agent import oracle_file_states, oracle_script
from swarmrl.traces.schema import Episode


def oracle_episode(make_runner, task, factory):
    cfg = default_swarm_config()
    return asyncio.run(make_runner(oracle_script(oracle_file_states(task, factory), cfg), cfg).run(task, seed=1))


def test_filters_keep_clean_wins_only(make_runner, task, factory):
    good = oracle_episode(make_runner, task, factory)
    fp = default_swarm_config().fingerprint()
    rep = filter_episodes([good], FilterConfig(fp))
    assert rep.kept == [good]
    rep = filter_episodes([good], FilterConfig("other"))
    assert rep.dropped == {"fingerprint_matches": 1}
    dup = Episode.from_dict(good.to_dict())
    dup.episode_id = "dup"
    assert filter_episodes([good, dup], FilterConfig(fp)).deduped == 1
    held = Episode.from_dict(good.to_dict())
    held.split = "heldout"
    assert filter_episodes([held], FilterConfig(fp)).dropped == {"not_heldout": 1}


def test_lucky_guess_without_tests_is_dropped(make_runner, task, factory):
    files = oracle_file_states(task, factory)
    ((path, content),) = files.items()

    def script(role, turn, ctx):
        if role == "planner":
            return call("delegate", to="coder", plan="fix")
        if role == "coder":
            return [
                call("create_file", path=path, content=content, overwrite=True),
                call("handoff", to="tester", note="x"),
            ][turn]
        if role == "tester":
            return call("handoff", to="reviewer", note="looks fine")  # never ran tests
        return call("accept")

    ep = asyncio.run(make_runner(script).run(task))
    assert ep.resolved
    rep = filter_episodes([ep], FilterConfig(default_swarm_config().fingerprint()))
    assert rep.dropped == {"tester_ran_tests_after_last_edit": 1}


def test_sft_samples_mask_and_drop(make_runner, task, factory):
    ep = oracle_episode(make_runner, task, factory)
    samples, dropped = build_sft_samples([ep], max_len=10**6)
    assert len(samples) == len(ep.transitions) and dropped == 0
    for s, t in zip(samples, ep.transitions, strict=True):
        assert s.labels[: len(t.input_ids)] == [IGNORE] * len(t.input_ids)
        assert s.labels[len(t.input_ids) :] == t.output_ids + t.closing_ids
        assert s.labels[-1] == 258  # end-of-turn carries loss
    longest = max(len(t.input_ids) + len(t.output_ids) for t in ep.transitions)
    _, dropped = build_sft_samples([ep], max_len=longest - 1)
    assert dropped >= 1  # dropped, never truncated


def test_adaptive_sampler():
    class T:
        def __init__(self, i):
            self.task_id, self.split = f"t{i}", "train"

    def fake(tid, resolved, n=1):
        out = []
        for _ in range(n):
            e = Episode("x", tid, "train", "fp", "base", 0, team_reward=float(resolved))
            from swarmrl.traces.schema import GradeSummary, Termination

            e.termination, e.grade = Termination.SUBMIT, GradeSummary(resolved=resolved)
            out.append(e)
        return out

    s = AdaptiveSampler([T(0), T(1)], target_successes=4, max_attempts=16, per_round=4)
    assert len(s.next_requests(0)) == 8
    s.record(fake("t0", True, 4) + fake("t1", False, 4))
    assert s.done("t0") and not s.done("t1")
    for r in range(3):
        s.record(fake("t1", False, len(s.next_requests(r + 1))))
    assert s.done("t1") and s.never_solved() == ["t1"]
    assert s.solve_rates() == {"t0": 1.0, "t1": 0.0}
    student = fake("t0", True, 4)
    teacher = fake("t1", True, 10)
    for e in teacher:
        e.teacher = True
    assert len(cap_teacher(student + teacher, 0.5)) == 8
    assert preference_pairs(fake("t0", True, 2) + fake("t0", False, 3)) == [("t0", "x", "x")] * 2
