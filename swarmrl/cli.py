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
  swarmrl plan --model M --nodes N [--json]         5D parallelism plan: Megatron layout + SGLang engines
  swarmrl slime-data TASKS --out tasks.jsonl        slime --prompt-data rows for the valid train tasks
  swarmrl slime-args --model M --stage S ...        slime train.py command (Megatron + SGLang, swarm rollouts)
  swarmrl deploy --profile P --out DIR              render the k8s deployment (RDMA, Kueue/Volcano, slime, llm-d)
  swarmrl v2-rl TASKS --planner-server ... ...      v2: planner / router / coder, one model + LoRA per role
  swarmrl v2-sim                                    v2 learning dynamics: shadows + IPS vs a hard router gate
  swarmrl pool-endpoints --namespace NS --selector  host:port of the Ready rollout-pool engines

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


def cmd_plan(a) -> int:
    from swarmrl.parallel.planner import ClusterSpec, ModelSpec, TrainSpec, plan

    m = ModelSpec.load(a.model)
    c = ClusterSpec(nodes=a.nodes, gpus_per_node=a.gpus_per_node, gpu_mem_gib=a.gpu_mem_gib, inter_gbs=a.inter_gbs)
    t = TrainSpec(seq_len=a.seq_len, tokens_per_microbatch=a.tokens_per_microbatch, global_tokens=a.global_tokens,
                  precision=a.precision, recompute=a.recompute, lora_rank=a.lora_rank,
                  optimizer_offload=a.optimizer_offload)  # fmt: skip
    try:
        p = plan(m, c, t, rollout_gpus=a.rollout_gpus, context=a.rollout_context, min_concurrent=a.min_concurrent,
                 auto_fit=not a.no_auto_fit)  # fmt: skip
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(p.to_dict(), indent=1) if a.json else p.report(a.top))
    return 0


def cmd_slime_data(a) -> int:
    from swarmrl.integrations.slime import prompt_rows
    from swarmrl.rl.schedule import stage
    from swarmrl.tasks.harbor import load_tasks

    st = stage(a.stage)
    rates = json.loads(Path(a.solve_rates).read_text())["solve_rates"] if a.solve_rates else None
    tasks = _valid_tasks(load_tasks(a.tasks, split="train"), _validator(a.tasks, _factory(a)))
    rows = prompt_rows(tasks, rates, st.data_lo, st.data_hi)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"tasks": len(rows), "out": a.out}))
    return 0 if rows else 1


def cmd_slime_args(a) -> int:
    from swarmrl.integrations.slime import launch_command, slime_args
    from swarmrl.parallel.planner import ClusterSpec, ModelSpec, TrainSpec, plan
    from swarmrl.rl.schedule import stage

    st = stage(a.stage)
    p = plan(ModelSpec.load(a.model), ClusterSpec(nodes=a.nodes, gpus_per_node=a.gpus_per_node),
             TrainSpec(seq_len=a.seq_len, global_tokens=a.global_tokens), rollout_gpus=a.rollout_gpus,
             context=a.seq_len)  # fmt: skip
    args = slime_args(p, st, hf_checkpoint=a.hf_checkpoint, load=a.load, save=a.save, prompt_data=a.prompt_data,
                      rollout_batch_size=a.rollout_batch_size, num_rollout=a.num_rollout,
                      max_response_len=a.max_response_len, lr=a.lr,
                      external_engines=a.external_engines or ())  # fmt: skip
    if a.json:
        print(json.dumps({"args": args, "env": p.env(), "plan": p.to_dict()}, indent=1))
    else:
        print(launch_command(args, p.env()))
    return 0


def cmd_deploy(a) -> int:
    from swarmrl.deploy.profile import DeployProfile
    from swarmrl.deploy.render import write

    for path in write(DeployProfile.load(a.profile), a.out):
        print(path)
    return 0


def cmd_pool_endpoints(a) -> int:
    import subprocess

    r = subprocess.run(["kubectl", "get", "pods", "-n", a.namespace, "-l", a.selector, "-o", "json"],
                       capture_output=True, text=True, timeout=60)  # fmt: skip
    if r.returncode != 0:
        print(r.stderr.strip(), file=sys.stderr)
        return 1
    addrs = []
    for pod in json.loads(r.stdout)["items"]:
        conds = {c["type"]: c["status"] for c in pod.get("status", {}).get("conditions", [])}
        ip = pod.get("status", {}).get("podIP")
        if ip and conds.get("Ready") == "True" and not pod["metadata"].get("deletionTimestamp"):
            addrs.append(f"{ip}:{a.port}")
    if len(addrs) < a.min_ready:
        print(f"only {len(addrs)} ready engine(s), need {a.min_ready}", file=sys.stderr)
        return 1
    print(" ".join(sorted(addrs)))
    return 0


ROLES = ("planner", "coder", "router")


def cmd_v2_rl(a) -> int:
    from swarmrl.grading.grader import CleanRoomGrader
    from swarmrl.inference.chat import ByteTokenizer, ChatFormat, HFTokenizer
    from swarmrl.inference.client import SGLangClient
    from swarmrl.inference.weight_sync import DiskWeightSync, SGLangAdmin
    from swarmrl.observability import MetricsSink
    from swarmrl.rl.schedule import stage
    from swarmrl.rl.trainer import TorchPolicyTrainer, TorchTrainerConfig
    from swarmrl.tasks.harbor import load_tasks
    from swarmrl.v2.loop import TriLoop
    from swarmrl.v2.router import Gate, GateConfig
    from swarmrl.v2.runtime import RoleModel, TriConfig, TriRunner
    from swarmrl.v2.train import VerifierTrainer

    st = stage(a.stage)
    f = _factory(a)
    v = _validator(a.tasks, f)
    tasks = _valid_tasks(load_tasks(a.tasks, "train"), v)
    if not tasks:
        print("no valid train tasks")
        return 1
    roles, trainers, syncs = {}, {}, {}
    for role in ROLES:
        servers, tok = getattr(a, f"{role}_server"), getattr(a, f"{role}_tokenizer")
        client = SGLangClient(servers)
        chat = ChatFormat(ByteTokenizer() if tok == "byte" else HFTokenizer(tok))
        roles[role] = RoleModel(client, chat, getattr(a, f"{role}_max_tokens"))
        cls = VerifierTrainer if role == "router" else TorchPolicyTrainer
        trainers[role] = cls(
            TorchTrainerConfig(
                getattr(a, f"{role}_model"),
                str(Path(a.output_dir) / role),
                lora_rank=a.lora_rank,
                learning_rate=a.lr,
                save_mode=a.sync_mode,
            )
        )
        syncs[role] = DiskWeightSync(SGLangAdmin(servers), a.sync_mode, client, colocated=a.colocated)
    gate_cfg = GateConfig(cost_false_accept=a.cost_false_accept, cost_false_reject=a.cost_false_reject,
                          epsilon=a.epsilon)  # fmt: skip
    cfg = TriConfig(plans_per_round=a.plans, max_rounds=a.rounds, patches_per_plan=a.patches, gate=gate_cfg)
    grader = CleanRoomGrader(f, _swarm(False).protected_paths)
    runner = TriRunner(roles["planner"], roles["coder"], roles["router"], f, grader, v.validate, cfg, Gate(gate_cfg))
    sink = MetricsSink(str(Path(a.output_dir) / "metrics.jsonl"))
    loop = TriLoop(
        runner,
        tasks[: a.batch_tasks],
        trainers["planner"],
        trainers["coder"],
        trainers["router"],
        st,
        group_size=a.group_size,
        router_lr_scale=a.router_lr_scale,
        syncs=syncs,
        on_metrics=sink.log,
    )
    for i in range(a.steps):
        m = loop.step(i)
        print(json.dumps({k: m[k] for k in ("step", "resolve_rate", "router/false_reject_rate", "shadows")}))
    return 0


def cmd_v2_sim(a) -> int:
    from swarmrl.v2.sim import SimConfig, run, summary

    out = {}
    for explore in (True, False):
        rows = [summary(run(SimConfig(explore=explore, steps=a.steps, seed=s))) for s in range(a.seeds)]
        out["v2 (shadows + IPS)" if explore else "hard router gate"] = {
            k: round(sum(r[k] for r in rows) / len(rows), 3) for k in rows[0]
        }
    print(json.dumps(out, indent=1))
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

    s = sub.add_parser("plan", help="5D parallelism plan (TP/PP/CP/EP/DP) for Megatron + SGLang rollout engines")
    s.add_argument("--model", required=True, help="preset (e.g. gemma3-27b, qwen3-30b-a3b), model dir or config.json")
    s.add_argument("--nodes", type=int, default=1, help="training nodes (all GPUs when colocated)")
    s.add_argument("--gpus-per-node", type=int, default=8)
    s.add_argument("--gpu-mem-gib", type=float, default=80.0)
    s.add_argument("--inter-gbs", type=float, default=50.0, help="RDMA bandwidth per GPU, GB/s (400 Gb/s NIC = 50)")
    s.add_argument("--seq-len", type=int, default=32768, help="longest training sample in tokens")
    s.add_argument("--tokens-per-microbatch", type=int, default=0, help="dynamic-batching budget (default seq-len)")
    s.add_argument("--global-tokens", type=int, default=1 << 21, help="tokens per optimizer step")
    s.add_argument("--precision", choices=["bf16", "fp8"], default="bf16")
    s.add_argument("--recompute", choices=["selective", "full"], default="selective")
    s.add_argument("--lora-rank", type=int, default=0, help="0 = full-parameter training")
    s.add_argument("--optimizer-offload", action="store_true")
    s.add_argument("--no-auto-fit", action="store_true", help="fail instead of applying memory remedies")
    s.add_argument("--rollout-gpus", type=int, help="separate rollout GPUs (default: colocated on the training GPUs)")
    s.add_argument("--rollout-context", type=int, default=16384, help="typical rollout context length")
    s.add_argument("--min-concurrent", type=int, default=32, help="sequences each engine must hold in KV cache")
    s.add_argument("--top", type=int, default=8)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("slime-data", help="slime --prompt-data rows (one per valid train task)")
    s.add_argument("tasks")
    s.add_argument("--out", required=True)
    s.add_argument("--stage", default="1", help="stage whose solve-rate band selects tasks")
    s.add_argument("--solve-rates", help="rates.json from rejection-sample")
    _add_sandbox(s)
    s.set_defaults(fn=cmd_slime_data)

    s = sub.add_parser("slime-args", help="slime train.py command for a stage (Megatron + SGLang + swarm)")
    s.add_argument("--model", required=True, help="planner preset, model dir or config.json")
    s.add_argument("--stage", default="1")
    s.add_argument("--nodes", type=int, default=1)
    s.add_argument("--gpus-per-node", type=int, default=8)
    s.add_argument("--seq-len", type=int, default=32768)
    s.add_argument("--global-tokens", type=int, default=1 << 21)
    s.add_argument("--rollout-gpus", type=int, help="separate rollout GPUs (default: colocated)")
    s.add_argument("--external-engines", nargs="+", help="host:port of pre-launched SGLang servers (llm-d pool)")
    s.add_argument("--hf-checkpoint", required=True)
    s.add_argument("--load", required=True, help="Megatron torch_dist checkpoint")
    s.add_argument("--save", required=True)
    s.add_argument("--prompt-data", required=True, help="from `swarmrl slime-data`")
    s.add_argument("--rollout-batch-size", type=int, default=32, help="tasks per RL step")
    s.add_argument("--num-rollout", type=int, default=200, help="RL steps")
    s.add_argument("--max-response-len", type=int, default=8192, help="tokens per model turn")
    s.add_argument("--lr", type=float, default=1e-6)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_slime_args)

    s = sub.add_parser("deploy", help="render the Kubernetes deployment for a profile")
    s.add_argument("--profile", required=True, help="profile YAML/JSON (see deploy/profiles/)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_deploy)

    s = sub.add_parser("pool-endpoints", help="host:port of the Ready engines behind a rollout pool")
    s.add_argument("--namespace", required=True)
    s.add_argument("--selector", required=True)
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--min-ready", type=int, default=1)
    s.set_defaults(fn=cmd_pool_endpoints)

    s = sub.add_parser("v2-rl", help="v2 RL: large planner, small coder, small router; one model and LoRA per role")
    s.add_argument("tasks")
    for role, tokens in (("planner", 1024), ("coder", 8192), ("router", 128)):
        s.add_argument(f"--{role}-server", nargs="+", required=True, help=f"SGLang URL(s) serving the {role}")
        s.add_argument(f"--{role}-tokenizer", required=True, help="HF tokenizer of that model ('byte' for dry runs)")
        s.add_argument(f"--{role}-model", required=True, help="base model the trainer loads (LoRA on top)")
        s.add_argument(f"--{role}-max-tokens", type=int, default=tokens)
    s.add_argument("--stage", default="1")
    s.add_argument("--steps", type=int, default=100)
    s.add_argument("--batch-tasks", type=int, default=8)
    s.add_argument("--group-size", type=int, default=4, help="episodes per task per step")
    s.add_argument("--plans", type=int, default=4, help="plans per round (K)")
    s.add_argument("--patches", type=int, default=4, help="patches per executed plan (M)")
    s.add_argument("--rounds", type=int, default=2, help="planning rounds before the best plan is forced")
    s.add_argument("--epsilon", type=float, default=0.1, help="floor on any plan's execution probability")
    s.add_argument("--cost-false-accept", type=float, default=1.0)
    s.add_argument("--cost-false-reject", type=float, default=3.0)
    s.add_argument("--lr", type=float, default=1e-6)
    s.add_argument("--router-lr-scale", type=float, default=3.0, help="the value model learns faster (two timescales)")
    s.add_argument("--lora-rank", type=int, default=32)
    s.add_argument("--sync-mode", choices=["merged", "adapter"], default="adapter")
    s.add_argument("--colocated", action="store_true")
    s.add_argument("--output-dir", default="runs/v2")
    _add_sandbox(s)
    s.set_defaults(fn=cmd_v2_rl)

    s = sub.add_parser("v2-sim", help="simulate v2 learning dynamics against a hard router gate")
    s.add_argument("--steps", type=int, default=150)
    s.add_argument("--seeds", type=int, default=3)
    s.set_defaults(fn=cmd_v2_sim)

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
