"""Eval path B: Verifiers on Environments Hub benchmarks, against the same SGLang endpoint.

Answers "is general ability holding up?". Verifiers' Python API is mid-migration (legacy v0
vs v1, and SkyRL's integration has broken on its API changes before), so this calls the
pinned `vf-eval` CLI and reads its saved results instead of importing internals.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExternalEval:
    env_id: str
    num_examples: int = 100
    rollouts_per_example: int = 1
    env_args: dict | None = None


def vf_eval_command(ev: ExternalEval, model: str, base_url: str, api_key_var: str = "SGLANG_API_KEY") -> list[str]:
    cmd = [
        "vf-eval", ev.env_id,
        "-m", model,
        "-b", base_url.rstrip("/") + "/v1",
        "-k", api_key_var,
        "-n", str(ev.num_examples),
        "-r", str(ev.rollouts_per_example),
        "-s",  # save results to disk
    ]  # fmt: skip
    if ev.env_args:
        cmd += ["-a", json.dumps(ev.env_args)]
    return cmd


def run_external(ev: ExternalEval, model: str, base_url: str, cwd: str | Path, timeout_s: float = 7200) -> dict:
    env = dict(os.environ)
    env.setdefault("SGLANG_API_KEY", "EMPTY")
    p = subprocess.run(
        vf_eval_command(ev, model, base_url), cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_s
    )
    if p.returncode != 0:
        raise RuntimeError(f"vf-eval failed for {ev.env_id}: {p.stderr[-2000:]}")
    return {"env_id": ev.env_id, "stdout_tail": p.stdout[-4000:], "results_dir": str(latest_results_dir(cwd))}


def latest_results_dir(cwd: str | Path) -> Path | None:
    root = Path(cwd) / "outputs" / "evals"
    if not root.exists():
        return None
    dirs = [p for p in root.rglob("*") if p.is_dir() and (p / "metadata.json").exists()]
    return max(dirs, key=lambda p: p.stat().st_mtime, default=None)


def read_mean_reward(results_dir: str | Path) -> float | None:
    meta = Path(results_dir) / "metadata.json"
    if not meta.exists():
        return None
    d = json.loads(meta.read_text())
    for key in ("avg_reward", "reward_avg", "mean_reward"):
        if key in d:
            return float(d[key])
    return None
