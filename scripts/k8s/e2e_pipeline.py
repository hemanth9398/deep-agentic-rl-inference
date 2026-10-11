"""Drive the swarmrl lifecycle against a real Kubernetes cluster.

Model calls go to a stub SGLang server that replays the oracle swarm over HTTP (no GPU here);
every sandbox (episodes, grading, validation) is a real pod behind the deny-all NetworkPolicy.
Usage (cluster up and the task image loaded, see scripts/k8s/local_cluster.sh):
    python scripts/k8s/e2e_pipeline.py WORK_DIR --image swarmrl-task-stats:test

Copies examples/tasks/stats-empty-input into WORK_DIR, points it at IMAGE, then runs isolation checks,
eval, rejection sampling, SFT data, and one RL step each with GRPO and GiGPO. Exits non-zero unless
every step passes with zero infra errors, a real optimizer step, and no pods left behind.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("work_dir")
ap.add_argument("--image", required=True, help="task image already loaded into the cluster")
ap.add_argument("--namespace", default="swarmrl")
args = ap.parse_args()
repo, work = Path(__file__).resolve().parents[2], Path(args.work_dir).resolve()
sys.path[:0] = [str(repo), str(repo / "tests")]
tasks_dir = work / "tasks"
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(repo / "examples" / "tasks" / "stats-empty-input", tasks_dir / "stats-empty-input")
toml = tasks_dir / "stats-empty-input" / "task.toml"
toml.write_text(toml.read_text().replace("[environment]\n", f'[environment]\ndocker_image = "{args.image}"\n', 1))

from fakes.fake_sglang import StubSGLang  # noqa: E402

from swarmrl.cli import main  # noqa: E402
from swarmrl.sandbox.kubernetes import KubernetesSandboxFactory  # noqa: E402
from swarmrl.sandbox.local import LocalSandboxFactory  # noqa: E402
from swarmrl.swarm.config import default_swarm_config  # noqa: E402
from swarmrl.tasks.harbor import HarborTask  # noqa: E402
from swarmrl.tasks.oracle_agent import oracle_file_states, oracle_script  # noqa: E402

NS = args.namespace
task = HarborTask.load(tasks_dir / "stats-empty-input")
results: dict[str, str] = {}


def step(name, fn):
    t0 = time.monotonic()
    try:
        out = fn()
        results[name] = f"PASS ({time.monotonic() - t0:.0f}s){': ' + out if out else ''}"
    except Exception as e:
        results[name] = f"FAIL ({time.monotonic() - t0:.0f}s): {type(e).__name__}: {e}"
    print(f"[{name}] {results[name]}", flush=True)


def pods_left() -> int:
    out = subprocess.run(["kubectl", "get", "pods", "-n", NS, "-o", "json"], capture_output=True, text=True).stdout
    live = [p for p in json.loads(out)["items"] if not p["metadata"].get("deletionTimestamp")]
    return len(live)


# 1. Isolation from inside a sandbox pod our own code created.
def isolation():
    f = KubernetesSandboxFactory(namespace=NS)
    f.ensure_network_policy()
    sb = f.create(task)
    try:
        probe = "import socket; s=socket.socket(); s.settimeout(3); s.connect(('{}', {})); print('CONNECTED')"
        api = sb.exec(f'python -c "{probe.format("10.43.0.1", 443)}"', timeout=20)
        dns = sb.exec("python -c \"import socket; print(socket.gethostbyname('pypi.org'))\"", timeout=20)
        spec = json.loads(subprocess.run(["kubectl", "get", "pod", sb.name, "-n", NS, "-o", "json"],
                                         capture_output=True, text=True).stdout)["spec"]  # fmt: skip
        c = spec["containers"][0]
        assert "CONNECTED" not in api.stdout, f"sandbox reached the API server: {api.output}"
        assert dns.returncode != 0, f"sandbox resolved DNS: {dns.output}"
        assert c["resources"]["limits"] == {"cpu": "2", "memory": "4Gi", "ephemeral-storage": "10Gi"}
        assert spec["automountServiceAccountToken"] is False and c["securityContext"]["capabilities"]["drop"] == ["ALL"]
        big = "x = 1\n" * 60_000
        sb.write_file("big.py", big)
        assert sb.read_file("big.py") == big
        return "API server and DNS unreachable; limits, no SA token, caps dropped; 360 KB file round-trip"
    finally:
        sb.close()


step("isolation", isolation)

# 2. The CLI lifecycle, every sandbox a pod.
oracle = oracle_script(oracle_file_states(task, LocalSandboxFactory()), default_swarm_config())
coder_starts = [0]


def mixed(role, turn, ctx):
    """Oracle play, except every other episode's coder writes a wrong fix: groups get mixed rewards,
    so advantages are non-zero and the RL step must take a real optimizer step."""
    reply = oracle(role, turn, ctx)
    if role == "coder" and turn == 0:
        coder_starts[0] += 1
        if coder_starts[0] % 2 == 0:
            reply = reply.replace("return 0.0", "return None")
    return reply


stub = StubSGLang(mixed)
traces = str(work / "traces")
serving = ["--server", stub.url, "--tokenizer", "byte", "--traces", traces, "--sandbox", "k8s", "--namespace", NS]


def cli(*args):
    rc = main([*args])
    if rc != 0:
        raise RuntimeError(f"exit code {rc}")


def eval_():
    cli("eval", str(tasks_dir), "--split", "dev", "-k", "4", "--in-flight", "4", "--workers", "4",
        "--report", str(work / "eval.md"), *serving)  # fmt: skip
    m = json.loads((work / "eval.json").read_text())
    assert m["infra_errors"] == 0 and m["episodes"] == 4, m
    return f"episodes={m['episodes']}, resolve_rate={m['resolve_rate']}, infra_errors={m['infra_errors']}"


step("eval", eval_)
toml = task.task_dir / "task.toml"
toml.write_text(toml.read_text().replace('split = "dev"', 'split = "train"'))
rates = work / "rates.json"
rs_args = ["--out", str(rates), "--per-round", "2", "--target-successes", "2", "--in-flight", "4", "--workers", "4"]
step("rejection-sample", lambda: cli("rejection-sample", str(tasks_dir), *rs_args, *serving))
step("sft-data", lambda: cli("sft-data", traces, "--out", str(work / "sft.jsonl")))


def rl(estimator):
    import torch
    import transformers

    torch.manual_seed(0)
    model_dir = work / "tiny-model"
    if not model_dir.exists():
        transformers.LlamaForCausalLM(transformers.LlamaConfig(
            vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=4096)).save_pretrained(model_dir)  # fmt: skip
    r = json.loads(rates.read_text())
    assert 0 < r["solve_rates"][task.task_id] < 1, r  # mixed play: the task is learnable
    rates.write_text(json.dumps(r))
    out = work / f"rl-{estimator}"
    cli("rl", str(tasks_dir), "--model", str(model_dir), "--solve-rates", str(rates), "--stage", "1", "--steps", "1",
        "--batch-tasks", "1", "--advantage", estimator, "--output-dir", str(out), "--in-flight", "8", "--workers", "4",
        *serving)  # fmt: skip
    m = json.loads((out / "metrics.jsonl").read_text().splitlines()[0])
    keys = ["episodes", "infra_errors", "resolve_rate", "informative_groups", "train_samples", "train/stepped",
            "train/grad_norm", "gigpo/anchored_fraction"]  # fmt: skip
    summary = ", ".join(f"{k}={m[k]}" for k in keys if k in m)
    assert m["infra_errors"] == 0, f"episodes lost to infra errors: {summary}"
    assert 0 < m["resolve_rate"] < 1 and m["train/stepped"] == 1.0, f"no real optimizer step: {summary}"
    return summary


step("rl-grpo", lambda: rl("grpo"))
step("rl-gigpo", lambda: rl("gigpo"))
stub.close()

time.sleep(5)
step("cleanup", lambda: (_ for _ in ()).throw(RuntimeError(f"{n} pods left")) if (n := pods_left()) else "0 pods left")
print("\n" + json.dumps(results, indent=1))
sys.exit(0 if all(v.startswith("PASS") for v in results.values()) else 1)
