"""swarmrl command line: one command per lifecycle stage.

  swarmrl validate TASKS [--oracle-agent]           validate tasks, cache F2P/P2P, oracle agent, contamination
  swarmrl mutants REPO --tests D --targets ... --out generate procedural-mutant tasks
  swarmrl eval TASKS --split dev --server URL ...   eval path A through the swarm and SGLang
  swarmrl rejection-sample TASKS --server URL ...   adaptive rejection sampling; writes solve rates
  swarmrl sft-data TRACES --out samples.jsonl       filter episodes -> transition-level SFT samples
  swarmrl sft-train SAMPLES --base-model M --out D  QLoRA SFT                         (needs .[sft])
  swarmrl merge --base-model M --adapter A --out D  merge a LoRA adapter in bf16       (needs .[sft])
  swarmrl rl TASKS --server URL --model M ...       staged RL with the torch trainer (needs .[train]);
                                                    --advantage grpo (default) | gigpo (experimental)
  swarmrl advantages TRACES [--group G]             GRPO vs GiGPO advantages per turn, side by side
  swarmrl report TRACES [--split S]                 metrics report from the trace store
  swarmrl fingerprint [--single-agent]              print the swarm fingerprint

Sandboxes: --sandbox local (dev only, no isolation) | docker | k8s [--namespace NS].
Logging: SWARMRL_LOG_LEVEL, SWARMRL_LOG_JSON=1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("swarmrl.cli")


# --- builders shared by commands --------------------------------------------------------


def _factory(a):
    kind = getattr(a, "sandbox", "local")
    if kind == "docker":
        from swarmrl.sandbox.docker import DockerSandboxFactory

        return DockerSandboxFactory()
    if kind == "k8s":
        from swarmrl.sandbox.kubernetes import KubernetesSandboxFactory, PodLimits

        limits = PodLimits(cpu=a.pod_cpu, memory=a.pod_memory, cpu_request=a.pod_cpu_request,
                           memory_request=a.pod_memory_request)  # fmt: skip
        f = KubernetesSandboxFactory(namespace=a.namespace, limits=limits, ready_timeout_s=a.pod_ready_timeout)
        f.ensure_network_policy()
        return f
    from swarmrl.sandbox.local import LocalSandboxFactory

    return LocalSandboxFactory()


def _swarm(single: bool):
    from swarmrl.swarm.config import default_swarm_config, single_agent_config

    return single_agent_config() if single else default_swarm_config()


def _validator(tasks_dir: str, factory):
    from swarmrl.tasks.validate import TaskValidator, ValidationCache

    return TaskValidator(factory, ValidationCache(Path(tasks_dir) / ".validation.json"))


def _runner(a, cfg, factory, validator, policy=None, chat=None, checkpoint: str | None = None):
    from swarmrl.grading.grader import CleanRoomGrader
    from swarmrl.inference.chat import ByteTokenizer, ChatFormat, HFTokenizer
    from swarmrl.inference.client import SGLangClient
    from swarmrl.swarm.runtime import EpisodeRunner

    if chat is None:  # "byte" = deterministic byte tokenizer for dry runs against a stub server
        chat = ChatFormat(ByteTokenizer() if a.tokenizer == "byte" else HFTokenizer(a.tokenizer))
    policy = policy or SGLangClient(a.server)
    return EpisodeRunner(cfg, policy, chat, factory, CleanRoomGrader(factory, cfg.protected_paths), validator.validate,
                         model_checkpoint=checkpoint or a.checkpoint)  # fmt: skip


def _dispatcher(a, runner):
    from swarmrl.swarm.dispatcher import Dispatcher, DispatchLimits
    from swarmrl.traces.store import TraceStore

    store = TraceStore(a.traces)
    return Dispatcher(runner, DispatchLimits(a.workers, a.in_flight, a.workers), on_episode=store.put)


def _valid_tasks(tasks, validator) -> list:
    ok = []
    for t in tasks:
        rec = validator.validate(t)
        if rec.valid:
            ok.append(t)
        else:
            log.warning("skipping %s: %s", t.task_id, rec.reason)
    return ok


# --- commands ------------------------------------------------------------------------------


def cmd_validate(a) -> int:
    from swarmrl.inference.chat import ByteTokenizer, ChatFormat
    from swarmrl.tasks.harbor import load_tasks
    from swarmrl.tasks.oracle_agent import run_oracle_agent
    from swarmrl.tasks.splits import contamination

    tasks = load_tasks(a.tasks)
    f = _factory(a)
    cfg = _swarm(False)
    v = _validator(a.tasks, f)
    bad = 0
    for t in tasks:
        rec = v.validate(t)
        if rec.valid and a.oracle_agent and rec.oracle_agent_resolved is None:
            chat = ChatFormat(ByteTokenizer())  # the oracle agent never calls a model
            mk = lambda pol, chat=chat: _runner(a, cfg, f, v, policy=pol, chat=chat, checkpoint="oracle")  # noqa: E731
            ep = asyncio.run(run_oracle_agent(t, mk, f, cfg, chat))
            rec.oracle_agent_resolved = ep.resolved
            v.cache.put(rec)
        ok = rec.valid and rec.oracle_agent_resolved is not False
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {t.task_id:40s} f2p={len(rec.fail_to_pass)} p2p={len(rec.pass_to_pass)} "
              f"oracle_agent={rec.oracle_agent_resolved} {rec.reason}")  # fmt: skip
    for c in contamination(tasks):
        print("CONTAMINATION", c)
        bad += 1
    print(f"{len(tasks) - bad}/{len(tasks)} tasks ready")
    return 1 if bad else 0


def cmd_mutants(a) -> int:
    from swarmrl.tasks.mutants import generate_mutant_tasks

    paths = generate_mutant_tasks(a.repo, a.tests, a.targets, a.out, a.name or Path(a.repo).name, _swarm(False),
                                  max_tasks=a.max, seed=a.seed, split=a.split)  # fmt: skip
    print(f"wrote {len(paths)} tasks to {a.out}")
    return 0


def cmd_eval(a) -> int:
    from swarmrl.evals.metrics import compute_metrics
    from swarmrl.evals.report import eval_requests, render_markdown, write_report
    from swarmrl.tasks.harbor import load_tasks

    f = _factory(a)
    v = _validator(a.tasks, f)
    tasks = _valid_tasks(load_tasks(a.tasks, a.split), v)
    cfg = _swarm(a.single_agent).with_sampling(temperature=a.temperature, top_p=1.0)
    d = _dispatcher(a, _runner(a, cfg, f, v))
    eps = d.run_sync(eval_requests(tasks, a.k))
    m = compute_metrics(eps, k_values=tuple(x for x in (1, 4, 8) if x <= a.k))
    title = f"Eval {a.split} @ {a.checkpoint}"
    if a.report:
        write_report(m, a.report, a.report.rsplit(".", 1)[0] + ".json", title)
    print(render_markdown(m, title))
    return 0


def cmd_rejection_sample(a) -> int:
    from swarmrl.rl.curriculum import min_turns_resolved
    from swarmrl.sft.rejection import AdaptiveSampler, run_rejection_sampling
    from swarmrl.tasks.harbor import load_tasks

    f = _factory(a)
    v = _validator(a.tasks, f)
    tasks = _valid_tasks(load_tasks(a.tasks, "train"), v)
    cfg = _swarm(a.single_agent).with_sampling(temperature=a.temperature, top_p=1.0)
    d = _dispatcher(a, _runner(a, cfg, f, v))
    sampler = AdaptiveSampler(tasks, a.target_successes, a.max_attempts, a.per_round)
    eps = run_rejection_sampling(d, sampler)
    out = {"solve_rates": sampler.solve_rates(), "min_turns": min_turns_resolved(eps),
           "never_solved": sampler.never_solved(), "fingerprint": cfg.fingerprint()}  # fmt: skip
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{len(eps)} episodes; {len(out['never_solved'])} tasks never solved; solve rates -> {a.out}")
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
    return 0 if samples else 1


def cmd_sft_train(a) -> int:
    from swarmrl.sft.train import QLoRAConfig, train_qlora

    cfg = QLoRAConfig(
        a.base_model, a.out, lora_rank=a.lora_rank, learning_rate=a.lr, epochs=a.epochs, max_len=a.max_len
    )
    print(train_qlora(cfg, a.samples))
    return 0


def cmd_merge(a) -> int:
    from swarmrl.sft.train import merge_adapter

    print(merge_adapter(a.base_model, a.adapter, a.out))
    return 0


def cmd_rl(a) -> int:
    from swarmrl.inference.client import SGLangClient
    from swarmrl.inference.weight_sync import DiskWeightSync, SGLangAdmin
    from swarmrl.observability import MetricsSink
    from swarmrl.rl.curriculum import Curriculum, stage_pool
    from swarmrl.rl.gigpo import GiGPOConfig
    from swarmrl.rl.loop import RLLoop, stage0_verdict
    from swarmrl.rl.schedule import stage
    from swarmrl.rl.trainer import TorchPolicyTrainer, TorchTrainerConfig
    from swarmrl.tasks.harbor import load_tasks

    st = stage(a.stage)
    f = _factory(a)
    v = _validator(a.tasks, f)
    tasks = {t.task_id: t for t in _valid_tasks(load_tasks(a.tasks, "train"), v)}
    rates_file = json.loads(Path(a.solve_rates).read_text())
    rates = {k: p for k, p in rates_file["solve_rates"].items() if k in tasks}
    allowed = stage_pool(rates, st.data_lo, st.data_hi, st.max_turns_to_solve, rates_file.get("min_turns"))
    if not allowed:
        print(f"no tasks fit stage {st.name}'s data filter ({st.data_lo}-{st.data_hi}); re-measure solve rates")
        return 1
    cfg = st.apply(_swarm(False))
    client = SGLangClient(a.server)
    runner = _runner(a, cfg, f, v, policy=client, checkpoint=f"{a.run_name}:{st.name}")
    tcfg = TorchTrainerConfig(
        a.model,
        str(Path(a.output_dir) / "checkpoints"),
        lora_rank=a.lora_rank,
        learning_rate=a.lr,
        save_mode=a.sync_mode,
    )
    trainer = TorchPolicyTrainer(tcfg)
    sync = DiskWeightSync(SGLangAdmin(a.server), a.sync_mode, client, colocated=a.colocated)
    loop = RLLoop(
        _dispatcher(a, runner),
        trainer,
        sync,
        Curriculum(rates, seed=a.seed),
        st,
        tasks,
        batch_tasks=a.batch_tasks,
        allowed_tasks=allowed,
        log=MetricsSink(str(Path(a.output_dir) / "metrics.jsonl")),
        advantage_estimator=a.advantage,
        gigpo=GiGPOConfig(omega=a.gigpo_omega, gamma=a.gigpo_gamma, state_key=a.gigpo_state_key),
    )
    hist = loop.run(a.steps)
    if st.lr_scale == 0.0:
        verdict = stage0_verdict(hist)
        print("STAGE 0 PASS" if verdict.passed else "STAGE 0 FAIL\n" + "\n".join(verdict.reasons))
        return 0 if verdict.passed else 1
    print(f"done: {len(hist)} steps, last checkpoint {loop.checkpoint}")
    return 0


def cmd_advantages(a) -> int:
    from swarmrl.rl.gigpo import GiGPOConfig, explain, step_advantages
    from swarmrl.traces.store import TraceStore

    eps = [e for e in TraceStore(a.traces).episodes() if a.group is None or (e.group_id or e.task_id) == a.group]
    cfg = GiGPOConfig(omega=a.gigpo_omega, gamma=a.gigpo_gamma, state_key=a.gigpo_state_key)
    rows = explain(eps, cfg)
    _, stats = step_advantages(eps, cfg)
    if a.json:
        print(json.dumps({"stats": vars(stats), "rows": rows}, indent=1))
        return 0
    cols = ["group", "episode", "reward", "turn", "role", "tool", "state", "grpo", "step", "gigpo"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join("-" if r[c] is None else str(r[c]) for c in cols))
    print(
        f"\n{stats.anchor_groups} anchor groups; {stats.anchored_turns}/{stats.total_turns} turns "
        f"({stats.anchored_fraction:.0%}) got a step-level advantage; largest anchor {stats.max_anchor_size}"
    )
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


# --- argument parsing ----------------------------------------------------------------------


def _add_sandbox(s) -> None:
    s.add_argument("--sandbox", choices=["local", "docker", "k8s"], default="local",
                   help="local is for development only: no network or resource isolation")  # fmt: skip
    s.add_argument("--namespace", default="swarmrl", help="k8s namespace for sandbox pods")
    s.add_argument("--pod-cpu", default="2", help="k8s sandbox CPU limit")
    s.add_argument("--pod-memory", default="4Gi", help="k8s sandbox memory limit")
    s.add_argument("--pod-cpu-request", default="500m", help="k8s sandbox CPU request (what the scheduler reserves)")
    s.add_argument("--pod-memory-request", default="1Gi", help="k8s sandbox memory request")
    s.add_argument("--pod-ready-timeout", type=int, default=900, help="seconds to wait for a sandbox pod to start")


def _add_serving(s) -> None:
    s.add_argument(
        "--server", nargs="+", required=True, help="SGLang base URL(s); turns are routed sticky per (episode, role)"
    )
    s.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer name or path (must match the served model); 'byte' for dry runs",
    )
    s.add_argument("--checkpoint", default="base", help="label recorded in every trace")
    s.add_argument("--traces", default="traces")
    s.add_argument("--workers", type=int, default=16, help="CPU pool for container start-up and grading")
    s.add_argument("--in-flight", type=int, default=128, help="max episodes in the run stage at once")
    s.add_argument("--single-agent", action="store_true")
    _add_sandbox(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swarmrl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate", help="validate Harbor tasks")
    s.add_argument("tasks")
    s.add_argument("--oracle-agent", action="store_true")
    _add_sandbox(s)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("mutants", help="generate procedural-mutant tasks from a repo")
    s.add_argument("repo")
    s.add_argument("--tests", required=True)
    s.add_argument("--targets", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--name")
    s.add_argument("--max", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--split", default="train")
    s.set_defaults(fn=cmd_mutants)

    s = sub.add_parser("eval", help="eval path A")
    s.add_argument("tasks")
    s.add_argument("--split", default="dev")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--report", help="write markdown (+ .json) report here")
    _add_serving(s)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("rejection-sample", help="adaptive rejection sampling on the train split")
    s.add_argument("tasks")
    s.add_argument("--out", default="solve_rates.json")
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--target-successes", type=int, default=4)
    s.add_argument("--max-attempts", type=int, default=16)
    s.add_argument("--per-round", type=int, default=4)
    _add_serving(s)
    s.set_defaults(fn=cmd_rejection_sample)

    s = sub.add_parser("sft-data", help="build SFT samples from traces")
    s.add_argument("traces")
    s.add_argument("--out", required=True)
    s.add_argument("--fingerprint")
    s.add_argument("--single-agent", action="store_true")
    s.add_argument("--turn-cap", type=int, default=30)
    s.add_argument("--max-per-task", type=int, default=4)
    s.add_argument("--max-len", type=int, default=32_768)
    s.add_argument("--teacher-fraction", type=float, default=0.5)
    s.set_defaults(fn=cmd_sft_data)

    s = sub.add_parser("sft-train", help="QLoRA SFT")
    s.add_argument("samples")
    s.add_argument("--base-model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--lora-rank", type=int, default=32)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--epochs", type=float, default=2.0)
    s.add_argument("--max-len", type=int, default=32_768)
    s.set_defaults(fn=cmd_sft_train)

    s = sub.add_parser("merge", help="merge a LoRA adapter into bf16 base weights")
    s.add_argument("--base-model", required=True)
    s.add_argument("--adapter", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("rl", help="staged GRPO (single-process torch trainer)")
    s.add_argument("tasks")
    s.add_argument("--model", required=True, help="trainer weights: the merged SFT checkpoint SGLang is serving")
    s.add_argument("--solve-rates", required=True, help="JSON from rejection-sample (re-measure per stage)")
    s.add_argument("--stage", default="1", help="0 (correctness), 1, 2 or 3")
    s.add_argument("--steps", type=int)
    s.add_argument("--batch-tasks", type=int, default=16)
    s.add_argument("--lr", type=float, default=1e-6)
    s.add_argument("--lora-rank", type=int, default=32)
    s.add_argument("--sync-mode", choices=["merged", "adapter"], default="adapter")
    s.add_argument("--colocated", action="store_true", help="trainer and SGLang share GPUs")
    s.add_argument("--output-dir", default="runs/rl")
    s.add_argument("--run-name", default="rl")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--advantage", choices=["grpo", "gigpo"], default="grpo",
                   help="grpo: per-role GRPO (default). gigpo: experimental two-level GiGPO")  # fmt: skip
    s.add_argument("--gigpo-omega", type=float, default=1.0, help="weight of GiGPO's step-level advantage")
    s.add_argument("--gigpo-gamma", type=float, default=0.95, help="discount for GiGPO step returns")
    s.add_argument("--gigpo-state-key", choices=["env", "context"], default="env", help="what counts as one state")
    _add_serving(s)
    s.set_defaults(fn=cmd_rl)

    s = sub.add_parser("advantages", help="compare GRPO and GiGPO advantages on stored traces, turn by turn")
    s.add_argument("traces")
    s.add_argument("--group", help="only this group id (e.g. step3:task-id)")
    s.add_argument("--gigpo-omega", type=float, default=1.0)
    s.add_argument("--gigpo-gamma", type=float, default=0.95)
    s.add_argument("--gigpo-state-key", choices=["env", "context"], default="env")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_advantages)

    s = sub.add_parser("report", help="metrics report from the trace store")
    s.add_argument("traces")
    s.add_argument("--split")
    s.add_argument("--checkpoint")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("fingerprint", help="print the swarm fingerprint")
    s.add_argument("--single-agent", action="store_true")
    s.set_defaults(fn=lambda a: print(_swarm(a.single_agent).fingerprint()) or 0)
    return p


def main(argv: list[str] | None = None) -> int:
    from swarmrl.observability import setup_logging

    setup_logging()
    a = build_parser().parse_args(argv)
    if getattr(a, "stage", None) is not None and str(a.stage).isdigit():
        a.stage = int(a.stage)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
