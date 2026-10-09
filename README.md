# deep-agentic-rl-inference

Post-training for a shared-policy agent swarm: one Gemma policy plays Planner, Coder, Tester and
Reviewer on Harbor-format coding tasks. Rollouts run through one runtime, inference is
token-in/token-out (SGLang), and training happens in exactly two places: offline QLoRA SFT, then
online GRPO (SkyRL-train).

Design doc: *Swarm Agent Post-Training Architecture (SkyRL-Agent + SGLang)*. This repo implements it
as the `swarmrl` package.

## Layers -> code

| Layer | Owns | Code |
| --- | --- | --- |
| Tasks | Harbor dirs, validation (F2P/P2P, oracle, oracle agent), splits, procedural mutants | `swarmrl/tasks/` |
| Sandboxes | fresh repo per episode; Docker: network off, memory/CPU/pid limits | `swarmrl/sandbox/` |
| Swarm runtime | roles, tools, hand-offs, harness controls, staged dispatcher | `swarmrl/swarm/` |
| Inference | token-level chat format, SGLang client, sticky routing, weight sync + KV flush | `swarmrl/inference/` |
| Rewards | clean-room grader, protected paths, team reward, bounded rubric terms | `swarmrl/grading/` |
| Traces | Episode -> AgentSpan -> Transition; one schema for evals and training | `swarmrl/traces/` |
| Evals | path A (swarm on dev/heldout), path B (Verifiers `vf-eval`), metrics, keep/freeze rule | `swarmrl/evals/` |
| SFT | adaptive rejection sampling, episode filters, transition samples, QLoRA, bf16 merge | `swarmrl/sft/` |
| RL | packing, per-role GRPO advantages, clipped loss + IS correction, curriculum, stages, step loop | `swarmrl/rl/` |
| SkyRL | swarm as a step-wise SkyRL generator; stage -> SkyRL config overrides | `swarmrl/integrations/skyrl.py` |
| Lifecycle | 8 stages, 7 gates, deployment choice | `swarmrl/lifecycle/gates.py` |

## Quick start

```bash
pip install -e '.[dev]'
pytest                                    # 42 tests; no GPU, Docker or model needed
swarmrl validate examples/tasks --oracle-agent
swarmrl mutants path/to/repo --tests tests --targets tests --out tasks/train --name myrepo
swarmrl eval tasks --split dev --server http://sglang:30000 --tokenizer <gemma> -k 8 --docker
swarmrl report traces --split dev
swarmrl sft-data traces --out sft.jsonl
```

Tests and the oracle agent use `LocalSandbox` (temp dir, no isolation). Use `--docker` for anything
that executes model output.

## Verify before building

Run these against your pinned versions before any training run:

| Check | Script |
| --- | --- |
| Token-in/token-out, log-probs, end-of-turn kept, re-scoring gap | `scripts/verify/sglang_tito.py` |
| Weight update + KV flush actually change and restore outputs | `scripts/verify/weight_sync.py` |
| Gemma chat template matches `ChatFormat` token for token | `scripts/verify/gemma_template.py` |
| Harbor entry points (`test.sh`, `solve.sh`) give reward 0 then 1 in your sandbox | `scripts/verify/harbor_runtime.py` |
| SkyRL inference backend, Verifiers and `vf-eval` versions | `scripts/verify/stack_versions.py` |

Competition rules on teacher models and external environments must be checked by hand.

**Known conflict:** at SkyRL main `b896332`, SkyRL-train's inference backend is vLLM only
(`generator.inference_engine.backend: "vllm"`). It cannot launch or weight-sync SGLang engines.
`swarmrl/integrations/skyrl.py` documents the two options: let SkyRL drive vLLM through
`SkyRLEnginePolicy`, or keep SGLang outside SkyRL with `SGLangClient` + `DiskWeightSync`.
Everything else (runtime, traces, rewards, packing, advantages) is the same either way.
