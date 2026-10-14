# v2: Planner / Router / Coder with one model per role

v1 runs one shared policy for four roles. v2 splits the work by cost and by kind of decision:

| Role | Model | Produces | Learns from |
| --- | --- | --- | --- |
| Planner | large (e.g. Qwen3-32B / 30B-A3B) | K short plans per round, each with a sandbox **probe** | what each executed plan achieved (`V_k`) |
| Router | small (e.g. Qwen3-1.7B / 4B) | `P(solved | candidate)` for every plan and patch, plus a written reason | real outcomes (calibration + ranking) |
| Coder | small (e.g. Qwen3-8B) | M patches per executed plan (full new file contents) | hidden tests, compared with its sibling patches |

The large model writes few tokens (plans are short). The small models write the many tokens (code) and make the
many decisions. That is where v2 saves compute.

## One episode

```text
round r:  planner --K plans--> probes run in sandboxes (test the hypothesis, write no fix)
                      |
                      v
          router: p_k = P(solved | plan_k, probe evidence)  --gate-->  executed with probability q_k
                      |                                                 (main plans + shadows of rejected ones)
          nothing executed? --> planner gets every rejection back (router reason + probe evidence), replans;
                                the last round always executes the best plan
coder:    M patches per executed plan -> apply in sandbox -> re-run probe + visible tests
router:   s_j = P(solved | patch_j, checks)  -> submit argmax s_j among the main plans' patches
grading:  EVERY executed patch on the hidden tests in a clean room (shadows and losing siblings included)
reward:   R = hidden-test outcome of the submitted patch
```

## The router's decision: calibration first, decision second

The router does not learn "accept" or "reject" directly. It learns a calibrated value
`p = P(solved | candidate)`. The decision then follows from costs (`swarmrl/v2/router.py`):

```text
accepting a bad plan costs c_FA, rejecting a good plan costs c_FR
accept iff p >= tau = c_FA / (c_FA + c_FR)         (default 1 : 3, tau = 0.25: losing ideas is worse)
q_k = eps + (1 - eps) * sigmoid((logit p_k - logit tau) / T)     execution probability, drawn independently
```

- `eps` (default 0.1): every plan keeps a chance to run, however sure the router is. A plan the router rejected
  that runs anyway is a **shadow**. Its real outcome is what an idea the router rejected was worth.
- `T` adapts each step so the gate's mean binary entropy tracks a target (default 0.5 bits). The gate never
  collapses to always or never; this is the "keep the entropy, don't collapse" requirement.
- Uncertainty is part of the decision: a plan near `tau` has `q` near 0.5, so the gate explores where the router
  is least sure.

## Losses: one per role, all tied to the final outcome (`swarmrl/v2/credit.py`, `train.py`)

Notation: `y_kj` is the hidden-test outcome of patch j of plan k. `V_k = mean_j y_kj` is the plan's Monte-Carlo
value. `w_k = min(1/q_k, cap)`. `A_team(e)` is GRPO of R over the task's G episodes.

**Planner** (actor over plans; off-policy, because the router filters what runs):

```text
b = sum_k w_k V_k / sum_k w_k ,   s^2 = sum_k w_k (V_k - b)^2 / sum_k w_k          (executed siblings)
A_P(k) = (w_k / mean w) (V_k - b) / (s + d)  +  beta A_team(e) [k is a main plan]
L_P = -(1/N) sum_tokens m min(rho A_P, clip(rho, 0.8, 1.28) A_P)
```

The planner is scored by what its plan achieved when executed, never by the router's opinion. A good idea the
router wrongly rejected still pulls the planner toward it, through its shadow and the inverse-propensity weight.
The IPS weight makes this estimate unbiased whatever the router does. An unparseable plan counts as an observed
failure.

**Coder** (actor over patches, conditioned on the plan, so plan quality cancels out):

```text
A_C(k, j) = (y_kj - mean_j y_kj) / (std_j y_kj + d)  +  beta A_team(e) [k is a main plan]
```

**Router** (verifier / value: `z = log p(YES) - log p(NO)` at its first output token):

```text
L_R = sum_i w_i BCE(sigmoid(z_i), label_i) / sum_i w_i  +  lambda mean_pairs softplus(z_worse - z_better)
labels: V_k for plans (IPS-weighted), y_kj for patches; pairs: a plan's patches with different outcomes
```

The router therefore learns "how good would this be if accepted", including for ideas it rejected, because
shadows provide those labels. The trainer accumulates the exact gradient of this batch loss one sequence at a
time; `tests/test_v2_train.py` checks it against autograd.

**Learning from rejection.** There are three channels:
1. The rejected plan, the router's reason and the probe evidence go into the next round's prompt
   (in-context).
2. Shadows give the rejected plan its real value. The planner is credited for it and the router is corrected
   (counterfactual).
3. A revised plan in the next round is its own sibling group, so a revision that fixes what the router flagged
   is reinforced.

Three models, three LoRA adapters, three optimizers; no gradient crosses between them. The dependency between
roles runs through the data: router decisions choose what runs (corrected by IPS), the planner's plans set the
coder's context, and the coder's skill sets each plan's value. The router gets a higher learning rate
(`--router-lr-scale`, default 3): it is the critic the actors depend on.

## Does it work? (`swarmrl v2-sim`, `tests/test_v2_core.py`)

This simulation runs through the real gate and credit code. Ideas have true success rates 0.8 / 0.35 / 0.2 /
0.05. The router starts out confident that the best idea is bad (p = 0.03). Averages over 3 seeds, 120 steps:

| | Planner P(best idea) | Router p for the best idea (true 0.8) | Resolve rate |
| --- | --- | --- | --- |
| v2: shadows + IPS + calibration | **0.98** | **0.79** (calibrated) | **1.00** |
| Hard router gate (only accepted plans run) | 0.19 | 0.05 (never corrected) | 0.80 |

With a hard gate the best idea is lost for good. The router never sees it run, so it never learns it was wrong,
and its measured false-reject rate looks low because rejected ideas are never measured. This is the failure
the design exists to prevent.

## Changes from the original idea, and why

- **The router is not trained to imitate the large model's decision.** It is trained on real hidden-test
  outcomes. Imitation would copy the planner's mistakes; outcomes are ground truth for both.
- **"Final state in the router's mind"** is literal: the router's output is `P(solved final state | candidate)`,
  a value function over the final state. Decisions come from that value plus costs.
- **Every executed patch is graded**, not only the submitted one. Grading costs K x M sandboxes per round; the
  per-patch labels give the coder and router signal even when the submitted answer fails.
- **Router reasons are not RL-trained yet.** Only the verdict token carries loss, which avoids rewarding
  persuasive text. A follow-up could reward a reason by whether the revision that read it succeeded.
- **P(YES) at rollout** comes from the sampled first token's log-prob (two-way). The trainer computes the exact
  two-way softmax. Serve with top-logprobs to make rollout p exact.

## Run it

```bash
swarmrl v2-sim                                        # the dynamics check above
swarmrl v2-rl tasks/ \
  --planner-server $BIG --planner-tokenizer Qwen/Qwen3-32B --planner-model Qwen/Qwen3-32B \
  --coder-server $MID  --coder-tokenizer Qwen/Qwen3-8B     --coder-model Qwen/Qwen3-8B \
  --router-server $SMALL --router-tokenizer Qwen/Qwen3-1.7B --router-model Qwen/Qwen3-1.7B \
  --sandbox k8s --plans 4 --patches 4 --rounds 2 --epsilon 0.1 --output-dir runs/v2
```

Metrics per step (`runs/v2/metrics.jsonl`):
- resolve rate and rounds per episode;
- executed fraction and number of shadows;
- `router/false_reject_rate`, `router/false_accept_rate`, `router/brier`;
- gate temperature;
- tokens per role;
- each role's loss and step.

## Verified here, and not

Verified:
- Gate, credit and IPS math, with worked numbers in tests.
- The dynamics simulation.
- Full episodes on the real example task: real sandboxes, probes, visible checks, clean-room hidden-test grading,
  shadows catching a wrong rejection, rejections fed back to the planner.
- The router's exact gradient.
- Router calibration on a tiny model.
- One full step updating three separate, differently sized models.

Not verified: real large/small models on GPUs and learning curves on a real task set. Run `v2-rl` against your
servers and compare its resolve rate per token with v1 at the same compute.
