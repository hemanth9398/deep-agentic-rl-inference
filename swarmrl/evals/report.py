"""Eval path A (your tasks, through the real swarm) and report rendering."""

from __future__ import annotations

import json
from collections.abc import Sequence

from swarmrl.evals.metrics import compute_metrics
from swarmrl.swarm.dispatcher import Dispatcher, EpisodeRequest
from swarmrl.traces.schema import Episode


def eval_requests(tasks: Sequence, k: int = 8, seeds: Sequence[int] | None = None) -> list[EpisodeRequest]:
    """k samples per task with fixed seeds, so a re-run after a change is paired task by task."""
    seeds = list(seeds) if seeds is not None else list(range(k))
    return [EpisodeRequest(t, s, t.split, group_id=f"eval:{t.task_id}") for t in tasks for s in seeds]


def run_eval_path_a(dispatcher: Dispatcher, tasks: Sequence, k: int = 8) -> tuple[list[Episode], dict]:
    eps = dispatcher.run_sync(eval_requests(tasks, k))
    return eps, compute_metrics(eps, k_values=tuple(x for x in (1, 4, 8) if x <= k))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, tuple):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) or "-"
    return str(v)


def render_markdown(metrics: dict, title: str = "Eval report") -> str:
    lo, hi = metrics["resolve_ci95"]
    lines = [
        f"# {title}",
        "",
        f"Resolve rate **{metrics['resolve_rate']:.3f}** (95% CI {lo:.3f}-{hi:.3f}) over "
        f"{metrics['episodes']} scored episodes ({metrics['infra_errors']} infra errors excluded).",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in sorted(metrics):
        if key in ("per_task", "resolve_rate", "resolve_ci95", "episodes", "infra_errors"):
            continue
        lines.append(f"| {key} | {_fmt(metrics[key])} |")
    lines += ["", "| Task | Resolve rate |", "| --- | --- |"]
    lines += [f"| {t} | {r:.3f} |" for t, r in metrics["per_task"].items()]
    return "\n".join(lines) + "\n"


def write_report(metrics: dict, path_md: str, path_json: str | None = None, title: str = "Eval report") -> None:
    with open(path_md, "w") as f:
        f.write(render_markdown(metrics, title))
    if path_json:
        with open(path_json, "w") as f:
            json.dump(metrics, f, indent=1, default=list)
