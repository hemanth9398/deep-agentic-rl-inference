"""swarmrl command line.

  swarmrl validate TASKS_DIR [--docker] [--oracle-agent]   validate tasks, cache F2P/P2P, run the oracle agent
  swarmrl mutants REPO --tests DIR --targets ... --out OUT generate procedural-mutant tasks
  swarmrl eval TASKS_DIR --split dev --server URL ...      eval path A through SGLang
  swarmrl report TRACES [--split heldout] [--checkpoint C] metrics report from the trace store
  swarmrl sft-data TRACES --out samples.jsonl ...          filter episodes -> transition-level SFT samples
  swarmrl fingerprint                                       print the default swarm fingerprint
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _factory(docker: bool):
    if docker:
        from swarmrl.sandbox.docker import DockerSandboxFactory

        return DockerSandboxFactory()
    from swarmrl.sandbox.local import LocalSandboxFactory

    return LocalSandboxFactory()


def _swarm(single: bool):
    from swarmrl.swarm.config import default_swarm_config, single_agent_config

    return single_agent_config() if single else default_swarm_config()


def cmd_validate(a) -> int:
    from swarmrl.grading.grader import CleanRoomGrader
    from swarmrl.inference.chat import ByteTokenizer, ChatFormat
    from swarmrl.swarm.runtime import EpisodeRunner
    from swarmrl.tasks.harbor import load_tasks
    from swarmrl.tasks.oracle_agent import run_oracle_agent
    from swarmrl.tasks.splits import contamination
    from swarmrl.tasks.validate import TaskValidator, ValidationCache

    tasks = load_tasks(a.tasks)
    f = _factory(a.docker)
    cfg = _swarm(False)
    cache = ValidationCache(Path(a.tasks) / ".validation.json")
    v = TaskValidator(f, cache)
    bad = 0
    for t in tasks:
        rec = v.validate(t)
        if rec.valid and a.oracle_agent and rec.oracle_agent_resolved is None:
            chat = ChatFormat(ByteTokenizer())
            mk = lambda pol, rec=rec: EpisodeRunner(cfg, pol, chat, f, CleanRoomGrader(f, cfg.protected_paths), lambda _t: rec)  # noqa: E731
            ep = asyncio.run(run_oracle_agent(t, mk, f))
            rec.oracle_agent_resolved = ep.resolved
            cache.put(rec)
        ok = rec.valid and rec.oracle_agent_resolved is not False
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {t.task_id:40s} f2p={len(rec.fail_to_pass)} p2p={len(rec.pass_to_pass)} "
              f"oracle_agent={rec.oracle_agent_resolved} {rec.reason}")  # fmt: skip
    for c in contamination(tasks):
        print("CONTAMINATION", c)
        bad += 1
    return 1 if bad else 0


def cmd_mutants(a) -> int:
    from swarmrl.tasks.mutants import generate_mutant_tasks

    paths = generate_mutant_tasks(a.repo, a.tests, a.targets, a.out, a.name or Path(a.repo).name, _swarm(False),
                                  max_tasks=a.max, seed=a.seed, split=a.split)  # fmt: skip
    print(f"wrote {len(paths)} tasks to {a.out}")
    return 0


def cmd_eval(a) -> int:
    from swarmrl.evals.report import eval_requests, render_markdown
    from swarmrl.evals.metrics import compute_metrics
    from swarmrl.grading.grader import CleanRoomGrader
    from swarmrl.inference.chat import ChatFormat, HFTokenizer
    from swarmrl.inference.client import SGLangClient
    from swarmrl.swarm.dispatcher import DispatchLimits, Dispatcher
    from swarmrl.swarm.runtime import EpisodeRunner
    from swarmrl.tasks.harbor import load_tasks
    from swarmrl.tasks.validate import TaskValidator, ValidationCache
    from swarmrl.traces.store import TraceStore

    tasks = load_tasks(a.tasks, a.split)
    f = _factory(a.docker)
    cfg = _swarm(a.single_agent).with_sampling(temperature=a.temperature, top_p=1.0)
    v = TaskValidator(f, ValidationCache(Path(a.tasks) / ".validation.json"))
    chat = ChatFormat(HFTokenizer(a.tokenizer))
    runner = EpisodeRunner(cfg, SGLangClient(a.server), chat, f, CleanRoomGrader(f, cfg.protected_paths),
                           v.validate, model_checkpoint=a.checkpoint)  # fmt: skip
    store = TraceStore(a.traces)
    d = Dispatcher(runner, DispatchLimits(a.workers, a.in_flight, a.workers), on_episode=store.put)
    eps = d.run_sync(eval_requests(tasks, a.k))
    m = compute_metrics(eps, k_values=tuple(x for x in (1, 4, 8) if x <= a.k))
    print(render_markdown(m, f"Eval {a.split} @ {a.checkpoint}"))
    return 0


def cmd_report(a) -> int:
    from swarmrl.evals.metrics import compute_metrics
    from swarmrl.evals.report import render_markdown
    from swarmrl.traces.store import TraceStore

    filt = {k: v for k, v in (("split", a.split), ("model_checkpoint", a.checkpoint)) if v}
    eps = list(TraceStore(a.traces).episodes(**filt))
    m = compute_metrics(eps)
    print(json.dumps(m, indent=1, default=list) if a.json else render_markdown(m))
    return 0


def cmd_sft_data(a) -> int:
    from swarmrl.sft.filters import FilterConfig, filter_episodes
    from swarmrl.sft.rejection import cap_teacher
    from swarmrl.sft.samples import build_sft_samples, write_jsonl
    from swarmrl.traces.store import TraceStore

    eps = list(TraceStore(a.traces).episodes())
    fp = a.fingerprint or _swarm(a.single_agent).fingerprint()
    rep = filter_episodes(eps, FilterConfig(fp, turn_cap=a.turn_cap, max_per_task=a.max_per_task))
    kept = cap_teacher(rep.kept, a.teacher_fraction)
    samples, dropped = build_sft_samples(kept, a.max_len)
    write_jsonl(samples, a.out)
    print(json.dumps({"episodes": len(eps), "kept": len(kept), "dropped_by_rule": dict(rep.dropped),
                      "deduped": rep.deduped, "capped": rep.capped, "samples": len(samples),
                      "overlength_dropped": dropped}, indent=1))  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swarmrl")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate")
    s.add_argument("tasks")
    s.add_argument("--docker", action="store_true")
    s.add_argument("--oracle-agent", action="store_true")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("mutants")
    s.add_argument("repo")
    s.add_argument("--tests", required=True)
    s.add_argument("--targets", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--name")
    s.add_argument("--max", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--split", default="train")
    s.set_defaults(fn=cmd_mutants)

    s = sub.add_parser("eval")
    s.add_argument("tasks")
    s.add_argument("--split", default="dev")
    s.add_argument("--server", nargs="+", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--checkpoint", default="base")
    s.add_argument("--traces", default="traces")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--in-flight", type=int, default=128)
    s.add_argument("--docker", action="store_true")
    s.add_argument("--single-agent", action="store_true")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("report")
    s.add_argument("traces")
    s.add_argument("--split")
    s.add_argument("--checkpoint")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("sft-data")
    s.add_argument("traces")
    s.add_argument("--out", required=True)
    s.add_argument("--fingerprint")
    s.add_argument("--single-agent", action="store_true")
    s.add_argument("--turn-cap", type=int, default=30)
    s.add_argument("--max-per-task", type=int, default=4)
    s.add_argument("--max-len", type=int, default=32_768)
    s.add_argument("--teacher-fraction", type=float, default=0.5)
    s.set_defaults(fn=cmd_sft_data)

    s = sub.add_parser("fingerprint")
    s.add_argument("--single-agent", action="store_true")
    s.set_defaults(fn=lambda a: print(_swarm(a.single_agent).fingerprint()) or 0)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
