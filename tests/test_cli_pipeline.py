"""The lifecycle through the CLI, over real HTTP: validate -> eval -> rejection-sample -> sft-data -> rl (stage 0)."""

import json

import pytest
from fakes.fake_sglang import StubSGLang

from swarmrl.cli import main
from swarmrl.swarm.config import default_swarm_config
from swarmrl.tasks.oracle_agent import oracle_file_states, oracle_script


@pytest.fixture
def stub(task, factory):
    s = StubSGLang(oracle_script(oracle_file_states(task, factory), default_swarm_config()))
    yield s
    s.close()


def test_validate_and_fingerprint(task, capsys):
    assert main(["validate", str(task.task_dir.parent), "--oracle-agent"]) == 0
    assert "1/1 tasks ready" in capsys.readouterr().out
    assert main(["fingerprint"]) == 0
    assert capsys.readouterr().out.strip() == default_swarm_config().fingerprint()


def test_eval_report_and_sft_data(task, stub, tmp_path, capsys):
    tasks, traces = str(task.task_dir.parent), str(tmp_path / "traces")
    serving = ["--server", stub.url, "--tokenizer", "byte", "--traces", traces]
    assert main(["eval", tasks, "--split", "dev", "-k", "2", "--report", str(tmp_path / "r.md"), *serving]) == 0
    out = capsys.readouterr().out
    assert "Resolve rate **1.000**" in out and (tmp_path / "r.json").exists()
    assert main(["report", traces, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 2
    assert main(["sft-data", traces, "--out", str(tmp_path / "sft.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 6


def test_rejection_sample_then_rl_stage0(task, stub, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    # the example task is "dev"; rejection sampling and RL use the train split
    toml = task.task_dir / "task.toml"
    toml.write_text(toml.read_text().replace('split = "dev"', 'split = "train"'))
    tasks, traces = str(task.task_dir.parent), str(tmp_path / "traces")
    serving = ["--server", stub.url, "--tokenizer", "byte", "--traces", traces]
    rates = tmp_path / "rates.json"
    assert (
        main(["rejection-sample", tasks, "--out", str(rates), "--per-round", "2", "--target-successes", "2", *serving])
        == 0
    )
    r = json.loads(rates.read_text())
    assert r["solve_rates"] == {task.task_id: 1.0} and r["never_solved"] == []

    torch.manual_seed(0)
    model_dir = tmp_path / "tiny"
    transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=300,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=4096,
        )
    ).save_pretrained(model_dir)
    r["solve_rates"] = {task.task_id: 0.5}  # pretend the task is learnable so stage filters admit it
    rates.write_text(json.dumps(r))
    capsys.readouterr()
    code = main(
        [
            "rl",
            tasks,
            "--model",
            str(model_dir),
            "--solve-rates",
            str(rates),
            "--stage",
            "0",
            "--batch-tasks",
            "1",
            "--output-dir",
            str(tmp_path / "run"),
            *serving,
        ]
    )
    out = capsys.readouterr().out
    # The stub's log-probs are not the tiny model's, so stage 0 must FAIL on mismatch KL: the gate works.
    assert code == 1 and "STAGE 0 FAIL" in out and "mismatch KL" in out
    rows = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["stage"] == "0_correctness" and rows[0]["alignment_shift"] == 0
    paths = [p for p, _ in stub.calls]
    assert "/flush_cache" in paths and "/load_lora_adapter" in paths  # step 2 pushed step-1 weights


def test_rl_with_gigpo_and_advantages_command(task, stub, tmp_path, capsys):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    toml = task.task_dir / "task.toml"
    toml.write_text(toml.read_text().replace('split = "dev"', 'split = "train"'))
    tasks, traces = str(task.task_dir.parent), str(tmp_path / "traces")
    serving = ["--server", stub.url, "--tokenizer", "byte", "--traces", traces]
    model_dir = tmp_path / "tiny"
    transformers.LlamaForCausalLM(
        transformers.LlamaConfig(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096)
    ).save_pretrained(model_dir)  # fmt: skip
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps({"solve_rates": {task.task_id: 0.5}, "min_turns": {task.task_id: 6}}))
    rl_args = ["--stage", "1", "--steps", "1", "--batch-tasks", "1", "--advantage", "gigpo"]
    run_dir = ["--output-dir", str(tmp_path / "run")]
    code = main(["rl", tasks, "--model", str(model_dir), "--solve-rates", str(rates), *rl_args, *run_dir, *serving])
    assert code == 0, capsys.readouterr().out
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[0])
    assert row["advantage_estimator"] == "gigpo" and row["gigpo/anchored_fraction"] == 1.0
    capsys.readouterr()
    assert main(["advantages", traces]) == 0
    out = capsys.readouterr().out
    assert "grpo | step | gigpo" in out and "turns (100%) got a step-level advantage" in out
