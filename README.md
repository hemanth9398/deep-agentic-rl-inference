# deep-agentic-rl-inference

Post-training for a shared-policy agent swarm. One policy plays Planner, Coder, Tester and Reviewer
on Harbor-format coding tasks. Rollouts run through one runtime, inference is token-in/token-out
(SGLang), and training happens in exactly two places: LoRA/QLoRA SFT, then staged online GRPO.

Design doc: *Swarm Agent Post-Training Architecture (SkyRL-Agent + SGLang)*. This repo implements it
as the `swarmrl` package.

## Layers -> code

| Layer | Owns | Code |
| --- | --- | --- |
| Tasks | Harbor dirs, validation (F2P/P2P, oracle, oracle agent), splits, procedural mutants | `swarmrl/tasks/` |
| Sandboxes | `Sandbox` base class; Local (dev), Docker and Kubernetes backends: network off, resource limits | `swarmrl/sandbox/` |
| Swarm runtime | roles, tools, hand-offs, harness controls, failure-isolated staged dispatcher | `swarmrl/swarm/` |
| Inference | token-level chat format, SGLang client, sticky routing, weight sync + KV flush | `swarmrl/inference/` |
| Rewards | clean-room grader, protected paths, team reward, bounded rubric terms | `swarmrl/grading/` |
| Traces | Episode -> AgentSpan -> Transition; one schema for evals and training | `swarmrl/traces/` |
| Evals | path A (swarm on dev/heldout), path B (`vf-eval`), metrics, keep/freeze rule | `swarmrl/evals/` |
| SFT | adaptive rejection sampling, episode filters, transition samples, LoRA/QLoRA, bf16 merge | `swarmrl/sft/` |
| RL | packing, per-role GRPO, clipped loss + IS correction, curriculum, stages, step loop, torch trainer | `swarmrl/rl/` |
| SkyRL | swarm as a step-wise SkyRL generator; stage -> SkyRL config overrides | `swarmrl/integrations/skyrl.py` |
| Megatron + SGLang | swarm as slime's generate hook, per-role GRPO post-process, stage -> slime flags | `swarmrl/integrations/slime.py` |
| Parallelism | 5D planner (TP, PP, CP, EP/ETP, DP): Megatron layout, memory/time model, SGLang engine sizing | `swarmrl/parallel/planner.py` |
| Deployment | RDMA rails (SR-IOV + Multus), Kueue or Volcano, slime RayJob, llm-d pool + EPP, Envoy, KServe | `swarmrl/deploy/`, `deploy/` |
| Lifecycle | 8 stages, 7 gates, deployment choice | `swarmrl/lifecycle/gates.py` |

Every extension point is an abstract base class with working implementations, enforced by
`tests/test_interfaces.py`: `Sandbox`/`SandboxFactory`, `Tokenizer`, `PolicyClient`, `WeightSync`
and `Trainer`.

## Install

```bash
pip install -e '.[dev]'            # core: numpy only, runs every non-GPU stage
pip install -e '.[dev,train]'      # + torch/transformers/peft: RL trainer, LoRA SFT, merge
pip install -e '.[qlora]'          # + bitsandbytes: 4-bit QLoRA SFT (CUDA)
pip install -e '.[deploy]'         # + pyyaml: render the Kubernetes deployment
```

## Running the lifecycle

```bash
swarmrl validate tasks/ --oracle-agent --sandbox docker          # gate 1: oracle agent resolves 100%
swarmrl eval tasks/ --split dev -k 8 --server $SGLANG --tokenizer $MODEL --sandbox docker --report dev.md
swarmrl rejection-sample tasks/ --server $SGLANG --tokenizer $MODEL --sandbox docker --out rates.json
swarmrl sft-data traces/ --out sft.jsonl
swarmrl sft-train sft.jsonl --base-model $MODEL --out runs/sft
swarmrl merge --base-model $MODEL --adapter runs/sft/adapter --out runs/sft-merged
swarmrl rl tasks/ --stage 0 --model runs/sft-merged --solve-rates rates.json \
    --server $SGLANG --tokenizer runs/sft-merged --sandbox docker  # gate 4: exits 1 unless log-probs align
swarmrl rl tasks/ --stage 1 --steps 200 ...                       # then 2 and 3, re-measuring rates.json
swarmrl report traces/ --split heldout
```

`--sandbox local` has no isolation and is for development only. Use `docker`, or `k8s --namespace NS`
(it applies a deny-all NetworkPolicy; your CNI must enforce NetworkPolicy). Logs go to stderr
(`SWARMRL_LOG_LEVEL`, `SWARMRL_LOG_JSON=1`), and RL metrics go to `<output-dir>/metrics.jsonl`.

## Kubernetes sandboxes

`--sandbox k8s` runs every episode, grading and validation sandbox as its own pod. The namespace is
created if missing, a deny-all NetworkPolicy is applied (your CNI must enforce NetworkPolicy), and pods
get no service-account token and drop all capabilities. Pods request 0.5 CPU / 1 GiB and may burst to
2 CPU / 4 GiB (`--pod-cpu-request`, `--pod-cpu`, ...), so idle sandboxes don't block scheduling; a pod
that can't start within `--pod-ready-timeout` (900 s) becomes an infra error with the scheduler's reason.
Task images must be pullable by the cluster: set `docker_image` in `task.toml`.

End-to-end check on a local single-node cluster (k3s in Docker):

```bash
scripts/k8s/local_cluster.sh up
docker build -t swarmrl-task-stats:test examples/tasks/stats-empty-input/environment
scripts/k8s/local_cluster.sh load swarmrl-task-stats:test
eval "$(scripts/k8s/local_cluster.sh env)"
python scripts/k8s/e2e_pipeline.py /tmp/swarmrl-e2e --image swarmrl-task-stats:test
```

It checks network isolation from inside a sandbox pod, then runs eval, rejection sampling, SFT data and
one RL step each with GRPO and GiGPO through the CLI (a stub server plays the model), and fails unless
there are zero infra errors, a real optimizer step, and no pods left behind.

## Advantage estimators: GRPO (default) and GiGPO (experimental)

Both use the same rollouts, rewards and loss; only the advantage on each token differs.

| | GRPO (default) | GiGPO (`--advantage gigpo`) |
| --- | --- | --- |
| Unit | one number per (episode, role) | one number per turn |
| Formula | team reward minus the group mean, per role | GRPO's number + omega x step advantage |
| Step advantage | none | turns taken from the same state in different episodes are compared by discounted return gamma^(T-1-t) x reward |
| All-success group | no signal | still prefers the faster path when episodes share states |
| Runs on | `swarmrl rl` and SkyRL | `swarmrl rl` only (SkyRL computes its own GRPO) |

"Same state" (`--gigpo-state-key`): `env` (default) matches the role plus the history of tool calls and
observations, ignoring free-text reasoning and run-to-run noise such as timings; `context` matches exact
input tokens. A/B it with the same tasks and seeds:

```bash
swarmrl rl tasks/ --stage 1 --advantage grpo  --run-name grpo  --output-dir runs/grpo  ...
swarmrl rl tasks/ --stage 1 --advantage gigpo --run-name gigpo --output-dir runs/gigpo ...
swarmrl advantages traces/ --group step3:<task-id>    # both estimators per turn, side by side
```

Watch `gigpo/anchored_fraction` in `metrics.jsonl`: if few turns share a state, GiGPO falls back to GRPO.

## Scale-out: Megatron + SGLang on H100 clusters

The single-process trainer above is for correctness work. At scale the same swarm, rewards and per-role GRPO
run inside [slime](https://github.com/THUDM/slime) v0.4 (Megatron-Core 0.19 for training, SGLang for
rollouts), launched on Kubernetes by a KubeRay RayJob.

```text
            +--------------------------- one RL step ----------------------------+
 tasks ---> | slime rollout: swarmrl.integrations.slime.generate (one episode)   |
            |   4 roles, token-in/out --> SGLang engines (colocated, CUDA IPC)   |
            |                         or --> Envoy -> llm-d EPP -> SGLang pool   |
            |   tools run in k8s sandbox pods (deny-all netpol)                  |
            | reward_post_process: per-(task, role) GRPO                         |
            | Megatron step: TP x PP x CP x DP, MoE EP/ETP, TIS / masked IS      |
            | weight sync: CUDA IPC (colocated) | NCCL over RDMA rails (pool)    |
            +--------------------------------------------------------------------+
```

**1. Plan the parallelism** (`swarmrl plan`). The planner enumerates every layout Megatron accepts
(TP within NVLink, heads/KV heads divisible by TP, PP x VPP dividing the layers, CP needing
`seq_len % 2CP == 0`, EP dividing the experts, parallel folding for experts). For each layout it estimates
per-GPU memory and step time, and returns the fastest one that fits along with its flags. When nothing
fits, it applies full recompute, then bf16 Adam moments, then optimizer offload, in that order:

```text
$ swarmrl plan --model qwen3-30b-a3b                    # one 8xH100 node, 32k-token samples
chosen: TP4 PP1 CP2 DP1 EP8 ETP1 EDP1  peak 75.5 GiB/GPU, ~5,931 tok/s/GPU (~34% MFU)
rollout: colocated on the same GPUs; 4 SGLang engine(s) x TP2
$ swarmrl plan --model qwen3-235b-a22b --nodes 8        # EP crosses nodes -> DeepEP flags
chosen: TP4 PP2 CP8 DP1 EP32 ETP1 EDP1  peak 73.8 GiB/GPU
```

For Qwen3-30B-A3B on 8 GPUs this matches slime's own tuned script (TP4 + SP, EP8, ETP1, full recompute).
Treat the numbers as estimates for ranking layouts. Confirm memory with a short stage-0 run before a long job.

**2. Run on slime.** `swarmrl slime-data` writes `--prompt-data` (one row per valid train task) and
`swarmrl slime-args --model M --stage S ...` prints the `ray job submit` command. The swarm runs as
`--custom-generate-function-path`, which turns each episode into one packed, loss-masked sample per role
with rollout log-probs. Per-role GRPO runs as `--custom-reward-post-process-path`; slime's default would pool
all four roles into one group. Infra errors become inert `remove_sample` rows. Algorithm flags by stage:
- stage 0: lr 0;
- stage 2: `--use-tis`;
- stage 3: fully-async rollout plus masked IS (`icepop`);
- every stage: token-mean loss, no KL.

`tests/test_slime_real.py` runs real swarm episodes through slime's own `Sample` type and
`BatchBuilder.convert` (set `SLIME_SRC`).

Gemma 3 has no slime model provider or converter yet; use a supported family (Qwen3, Qwen3-MoE, GLM, Llama),
or add one through `--custom-model-provider-path` and Megatron-Bridge.

**3. Deploy** (`swarmrl deploy --profile deploy/profiles/h100.yaml --out DIR`). Three profiles are rendered
under `deploy/rendered/`, each with an install guide:

| Profile | Training | Rollouts | Scheduler |
| --- | --- | --- | --- |
| `h100` | 1 node, colocated | same GPUs, CUDA IPC | Kueue |
| `h100-pool` | 1 node | llm-d pool on a 2nd node, NCCL over RDMA, EPP routing | Kueue |
| `h100-volcano` | 2 nodes, colocated | same GPUs | Volcano |

What the manifests set up:
- **RDMA fabric.** Each GPU's NIC ("rail") is split into SR-IOV VFs with RDMA (exclusive RDMA namespaces).
  Pods get one VF per rail through Multus (`net1..net8`), next to Cilium on `eth0` (`cni.exclusive=false`).
  They also get IPC_LOCK, `/dev/shm` and NCCL settings: bootstrap on eth0, `NCCL_CROSS_NIC=0`, RoCEv2 GID,
  NVLS. GPUDirect RDMA comes from the GPU Operator (DMA-BUF).
- **Kueue v0.20.** One GPU cohort with three ClusterQueues:
  - `serving` never borrows; it lends idle GPUs and takes them back by preemption.
  - `rl-train` borrows serving's idle GPUs.
  - `eval` has no quota and only borrows.

  It also sets priorities, topology-aware placement (multi-node jobs stay in one rack) and `waitForPodsReady`,
  and excludes VF resources from quota.
- **Volcano** is the alternative scheduler: queues, rack HyperNodes and the network-topology plugin. Kueue and
  Volcano have no upstream integration, so a workload uses one or the other, never both.
- **llm-d pool.** SGLang engines (the same SGLang build as the trainer) publish KV-cache events. llm-d's router
  chart deploys the EPP with the precise prefix-cache producer (`engineType: sglang`). It reads our `/generate`
  `input_ids` directly (`sglanghttp-parser`), so no tokenizer runs in the routing path. Traffic goes through
  Envoy Gateway with the Envoy AI Gateway InferencePool extension. Each turn of a role goes to the engine
  that already caches its prefix.
- **Serving.** A KServe `LLMInferenceService` (v1alpha2, vLLM) serves a trained checkpoint with the same EPP.

### What has been verified, and where

| Check | Where | Result |
| --- | --- | --- |
| Planner rules, memory/time model, flags | unit tests | 21 tests |
| Swarm -> slime samples, per-role GRPO, flags per stage | unit tests + real slime `BatchBuilder` | pass |
| Every rendered manifest vs real CRD schemas (Kueue, KubeRay, SR-IOV, Volcano, Gateway API, InferencePool, KServe) | `kubectl apply --dry-run=server --validate=strict` on k3s | 26/26 files |
| Cross-references (NADs, VF resources, queues, selectors, EPP refs, ports) | `tests/test_deploy.py` | pass |
| llm-d routing: Envoy -> EPP v0.11 -> engines with KV events | `scripts/k8s/llmd_routing_e2e.py` on k3s | 32/32 turns kept their prefix's engine; 224 blocks indexed from events; SGLang config loads |
| Kueue policy: quota, borrowing, reclaim, ordering | `scripts/k8s/kueue_policy_e2e.py`, real Kueue v0.20.1 | pass |

Not verified here (no GPUs or RDMA NICs in this sandbox; ghcr, quay and registry.k8s.io images were built
from source or unavailable): Megatron/SGLang on GPUs, NCCL bandwidth on the rails, the SR-IOV operator
creating VFs, KubeRay running the job, Envoy AI Gateway and the KServe controller, and Volcano at runtime.
Run `scripts/verify/*` and a stage-0 job on the target cluster first.

## Failure handling

- An episode that hits an infrastructure problem, a wall-clock timeout (`[agent] timeout_sec`) or an
  unexpected exception is recorded as `infra_error`. It is never scored against the model and never
  aborts the batch. Our own bugs are additionally flagged `internal_error`.
- Tasks that fail validation are skipped by every command and refused at grading.
- Every sandbox is torn down on every path, including startup failures and timeouts.

## Tests

`pytest` runs 129 tests (one more with `SLIME_SRC` set); the torch tests skip when torch is absent. CI runs lint, the core suite on
Python 3.11-3.13, and the torch suite on CPU. The tests cover:

| Area | Exercised for real |
| --- | --- |
| Docker and Kubernetes sandboxes | full validation and oracle episodes through the real backend code against fake `docker`/`kubectl` CLIs; isolation flags asserted |
| CLI lifecycle | validate -> eval -> rejection-sample -> sft-data -> rl stage 0 over real HTTP against a stub SGLang server |
| RL trainer | exact log-probs (chunked), update direction, lr 0 = no change, ratio = 1 with dropout configured, adapter and merged checkpoints reload |
| SFT | LoRA training and bf16 merge on a tiny model |

## Verify before production

These need a GPU, a real server or a real cluster, so CI cannot run them:

| Check | Script |
| --- | --- |
| Token-in/token-out, log-probs, end-of-turn kept, re-scoring gap | `scripts/verify/sglang_tito.py` |
| Weight update + KV flush actually change and restore outputs | `scripts/verify/weight_sync.py` |
| Gemma chat template matches `ChatFormat` token for token | `scripts/verify/gemma_template.py` |
| Harbor entry points give reward 0 then 1 in your sandbox | `scripts/verify/harbor_runtime.py` |
| SkyRL inference backend, Verifiers and `vf-eval` versions | `scripts/verify/stack_versions.py` |

Competition rules on teacher models and external environments must be checked by hand.

**Training engine decision:** SkyRL-train (at `b896332`) can only weight-sync vLLM, so the multi-GPU path is
slime: Megatron training with SGLang rollouts and native CUDA-IPC/NCCL weight sync (see "Scale-out" above).
`swarmrl/integrations/skyrl.py` remains for teams that standardize on SkyRL with vLLM.
