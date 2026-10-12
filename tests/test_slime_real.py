"""Against slime's real code: our samples through slime's own Sample type and BatchBuilder.convert.

Skipped unless slime is importable, e.g. `SLIME_SRC=/path/to/slime pytest tests/test_slime_real.py`
(checked against THUDM/slime v0.4.0). `ray` is stubbed when absent: convert() never calls it.
"""

import importlib.util
import os
import sys
import types

import pytest

if os.environ.get("SLIME_SRC"):
    sys.path.insert(0, os.environ["SLIME_SRC"])
pytest.importorskip("torch")
if importlib.util.find_spec("slime") is None:
    pytest.skip("slime is not importable (set SLIME_SRC)", allow_module_level=True)
if "ray" not in sys.modules:
    try:
        import ray  # noqa: F401
    except ImportError:
        sys.modules["ray"] = types.ModuleType("ray")
slime_types = pytest.importorskip("slime.utils.types")
batch_builder = pytest.importorskip("slime.data.batch_builder")

from test_slime import _run_group

import swarmrl.integrations.slime as integration


class Args(types.SimpleNamespace):
    def __getattr__(self, name):  # every flag slime does not need here is "unset"
        return None


def _args(g, custom: bool):
    return Args(
        advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=True, n_samples_per_prompt=g,
        rollout_batch_size=1, rollout_top_p=1.0, use_rollout_routing_replay=False,
        custom_reward_post_process_path="swarmrl.integrations.slime.reward_post_process" if custom else None,
    )  # fmt: skip


@pytest.fixture
def real_groups(make_runner, task, mixed_script, monkeypatch):
    import test_slime

    monkeypatch.setattr(test_slime, "Sample", slime_types.Sample)
    groups, episodes = _run_group(make_runner, task, mixed_script, g=4)
    assert all(isinstance(s, slime_types.Sample) for out in groups for s in out)
    return [s for out in groups for s in out], episodes


@pytest.mark.parametrize("custom", [True, False])
def test_slime_batch_builder_accepts_our_samples(real_groups, custom):
    flat, episodes = real_groups
    data = batch_builder.BatchBuilder(_args(4, custom)).convert(flat)
    n = len(flat)
    assert len(data["tokens"]) == len(data["loss_masks"]) == len(data["rollout_log_probs"]) == n
    for toks, rl, mask, lps in zip(data["tokens"], data["response_lengths"], data["loss_masks"],
                                   data["rollout_log_probs"], strict=True):  # fmt: skip
        assert len(mask) == rl == len(lps) and rl < len(toks)
    assert sorted(set(data["rollout_ids"])) == sorted({s.rollout_id for s in flat})
    if custom:  # ours: GRPO per (prompt group, role), which slime applies verbatim
        _, ours = integration.reward_post_process(None, flat, per_role_norm=True, norm_by_std=True)
        assert data["rewards"] == pytest.approx(ours)
    else:
        # slime's default also accepts them, but pools every role's rollout of a prompt into one group and divides
        # by the sample std (n-1); that is why `swarmrl slime-args` always sets our per-role post-process.
        import statistics

        pooled = [float(s.reward) for s in {s.rollout_id: s for s in flat}.values()]
        x = (1 - statistics.mean(pooled)) / statistics.stdev(pooled)
        assert {round(abs(r), 4) for r in data["rewards"]} == {round(x, 4)} and x != pytest.approx(1.0)
