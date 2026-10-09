"""Eval path B: Verifiers on Environments Hub benchmarks, against the same SGLang endpoint.

Answers "is general ability holding up?". Verifiers' Python API is mid-migration (legacy v0 vs
v1, and SkyRL's integration has broken on its API changes before), so this drives the `vf-eval`
CLI and reads the files it saves. Checked against verifiers 0.3.1 (legacy `vf-eval`):

    <output_dir>/evals/<env_id>--<model with / -> -->/<run uuid>/metadata.json  ("avg_reward", ...)
    <output_dir>/evals/<env_id>--<model with / -> -->/<run uuid>/results.jsonl  (one row per rollout, "reward")
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ExternalEval:
    env_id: str
    num_examples: int = 100
    rollouts_per_example: int = 1
    env_args: dict | None = None


@dataclass
class ExternalResult:
    env_id: str
    mean_reward: float
    num_rollouts: int
    results_dir: str


def vf_eval_command(ev: ExternalEval, model: str, base_url: str, output_dir: str | Path,
                    api_key_var: str = "SGLANG_API_KEY") -> list[str]:  # fmt: skip
    cmd = [
        "vf-eval", ev.env_id,
        "-m", model,
        "-b", base_url.rstrip("/") + "/v1",  # SGLang's OpenAI-compatible endpoint
        "-k", api_key_var,
        "-n", str(ev.num_examples),
        "-r", str(ev.rollouts_per_example),
        "-o", str(output_dir),
        "-s",
    ]  # fmt: skip
    if ev.env_args:
        cmd += ["-a", json.dumps(ev.env_args)]
    return cmd


def runs_dir(output_dir: str | Path, env_id: str, model: str) -> Path:
    return Path(output_dir) / "evals" / f"{env_id}--{model.replace('/', '--')}"


def latest_run(output_dir: str | Path, env_id: str, model: str) -> Path | None:
    d = runs_dir(output_dir, env_id, model)
    if not d.is_dir():
        return None
    runs = [p for p in d.iterdir() if p.is_dir() and ((p / "metadata.json").exists() or (p / "results.jsonl").exists())]
    return max(runs, key=lambda p: p.stat().st_mtime, default=None)


def read_run(run: str | Path, env_id: str = "") -> ExternalResult:
    run = Path(run)
    rewards: list[float] = []
    res = run / "results.jsonl"
    if res.exists():
        with res.open() as f:
            rewards = [float(json.loads(line).get("reward", 0.0)) for line in f if line.strip()]
    meta = run / "metadata.json"
    mean = None
    if meta.exists():
        m = json.loads(meta.read_text())
        mean = m.get("avg_reward")
        env_id = env_id or m.get("env_id", "")
    if mean is None:
        if not rewards:
            raise ValueError(f"{run}: no avg_reward in metadata.json and no rewards in results.jsonl")
        mean = sum(rewards) / len(rewards)
    return ExternalResult(env_id, float(mean), len(rewards), str(run))


def run_external(
    ev: ExternalEval, model: str, base_url: str, output_dir: str | Path, timeout_s: float = 7200
) -> ExternalResult:
    env = dict(os.environ)
    env.setdefault("SGLANG_API_KEY", "EMPTY")
    before = latest_run(output_dir, ev.env_id, model)
    cmd = vf_eval_command(ev, model, base_url, output_dir)
    log.info("running %s", " ".join(cmd))
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError as e:
        raise RuntimeError("vf-eval not found: pip install 'swarmrl[verifiers]'") from e
    if p.returncode != 0:
        raise RuntimeError(f"vf-eval failed for {ev.env_id}: {p.stderr[-2000:]}")
    run = latest_run(output_dir, ev.env_id, model)
    if run is None or run == before:
        raise RuntimeError(f"vf-eval finished but saved no new run under {runs_dir(output_dir, ev.env_id, model)}")
    return read_run(run, ev.env_id)


def run_suite(evals: list[ExternalEval], model: str, base_url: str, output_dir: str | Path) -> dict[str, float]:
    """Mean reward per benchmark; the input to the RL stage-3 gate's external comparison."""
    return {ev.env_id: run_external(ev, model, base_url, output_dir).mean_reward for ev in evals}
