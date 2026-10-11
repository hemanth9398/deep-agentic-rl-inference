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
| Lifecycle | 8 stages, 7 gates, deployment choice | `swarmrl/lifecycle/gates.py` |

Every extension point is an abstract base class with working implementations, enforced by
`tests/test_interfaces.py`: `Sandbox`/`SandboxFactory`, `Tokenizer`, `PolicyClient`, `WeightSync`
and `Trainer`.

## Install

```bash
pip install -e '.[dev]'            # core: numpy only, runs every non-GPU stage
pip install -e '.[dev,train]'      # + torch/transformers/peft: RL trainer, LoRA SFT, merge
pip install -e '.[qlora]'          # + bitsandbytes: 4-bit QLoRA SFT (CUDA)
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

## Failure handling

- An episode that hits an infrastructure problem, a wall-clock timeout (`[agent] timeout_sec`) or an
  unexpected exception is recorded as `infra_error`. It is never scored against the model and never
  aborts the batch. Our own bugs are additionally flagged `internal_error`.
- Tasks that fail validation are skipped by every command and refused at grading.
- Every sandbox is torn down on every path, including startup failures and timeouts.

## Tests

`pytest` runs 75 tests; the torch tests skip when torch is absent. CI runs lint, the core suite on
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

**Open decision:** at SkyRL main `b896332`, SkyRL-train's inference backend is vLLM only and cannot
weight-sync SGLang. Either let SkyRL drive vLLM (`SkyRLEnginePolicy`), or keep SGLang and train with
`swarmrl rl` (single process) or SkyRL as the optimizer only. See `swarmrl/integrations/skyrl.py`.
