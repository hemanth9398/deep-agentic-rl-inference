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

## Failure handling

- An episode that hits an infrastructure problem, a wall-clock timeout (`[agent] timeout_sec`) or an
  unexpected exception is recorded as `infra_error`. It is never scored against the model and never
  aborts the batch. Our own bugs are additionally flagged `internal_error`.
- Tasks that fail validation are skipped by every command and refused at grading.
- Every sandbox is torn down on every path, including startup failures and timeouts.

## Tests

`pytest` runs 65 tests; the torch tests skip when torch is absent. CI runs lint, the core suite on
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
