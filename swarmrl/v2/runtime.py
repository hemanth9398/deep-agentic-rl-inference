"""One v2 episode: plan -> probe -> route -> code -> check -> rank -> grade, with a different model per role.

round r (up to max_rounds):
  1. the planner (large model) samples K plans, each with a probe;
  2. every probe runs in its own sandbox: evidence about the plan's hypothesis, no fix written;
  3. the router (small model) gives each plan p = P(solved | plan, evidence);
  4. the gate (router.py) executes plans with known probabilities q: main plans plus shadows of rejected ones;
     if nothing runs, the planner gets every rejected plan back with the router's reason and the probe
     evidence, and plans again (the last round always executes the best plan);
5. the coder (small model) writes M patches for every executed plan; each patch is applied in a sandbox,
   the probe is re-run and the visible tests run;
6. the router scores every patch; the best-scored patch of a main plan is the submitted answer;
7. every executed patch is graded on the hidden tests in a clean room: shadows and losing siblings included,
   because their outcomes are the training labels for all three roles (credit.py).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from swarmrl.grading.grader import CleanRoomGrader
from swarmrl.inference.chat import ChatFormat
from swarmrl.inference.client import Generation, PolicyClient
from swarmrl.sandbox.base import InfraError, SandboxFactory
from swarmrl.swarm.config import SamplingConfig
from swarmrl.v2 import protocol as P
from swarmrl.v2.records import Call, PatchRecord, PlanRecord, Round, TriEpisode, Verdict
from swarmrl.v2.router import Gate, GateConfig


@dataclass
class RoleModel:
    """A role's own model: its server, its tokenizer/chat format, its budget."""

    policy: PolicyClient
    chat: ChatFormat
    max_new_tokens: int
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass
class TriConfig:
    plans_per_round: int = 4
    max_rounds: int = 2
    patches_per_plan: int = 4
    probe_timeout_s: float = 30.0
    check_timeout_s: float = 120.0
    max_file_chars: int = 20_000
    gate: GateConfig = field(default_factory=GateConfig)


class TriRunner:
    def __init__(self, planner: RoleModel, coder: RoleModel, router: RoleModel, factory: SandboxFactory,
                 grader: CleanRoomGrader, validate: Callable, cfg: TriConfig = TriConfig(),
                 gate: Gate | None = None):  # fmt: skip
        self.roles = {"planner": planner, "coder": coder, "router": router}
        self.factory = factory
        self.grader = grader
        self.validate = validate
        self.cfg = cfg
        self.gate = gate or Gate(cfg.gate)
        self.verdict_tokens = P.VerdictTokens.for_chat(router.chat)

    # --- model calls -------------------------------------------------------------------------

    async def _generate(self, ep: TriEpisode, role: str, system: str, user: str, key: str) -> Call:
        m = self.roles[role]
        ids = m.chat.start(system, user)
        gen = await m.policy.generate(ids, m.sampling, m.chat.stop_ids, f"{ep.episode_id}-{key}:{role}",
                                      m.max_new_tokens)  # fmt: skip
        ep.tokens[role] = ep.tokens.get(role, 0) + len(gen.output_ids)
        text = m.chat.decode(gen.output_ids)
        return Call(role, ids, list(gen.output_ids), list(gen.logprobs), m.chat.closing_ids(gen.output_ids), text,
                    gen.finish_reason)  # fmt: skip

    async def _verdict(self, ep: TriEpisode, system: str, user: str, key: str) -> Verdict:
        call = await self._generate(ep, "router", system, user, key)
        p, valid = P.verdict_probability(Generation(call.output_ids, call.rollout_logprobs, call.finish_reason),
                                         self.verdict_tokens)  # fmt: skip
        reason = call.text.split("\n", 1)[-1].strip() if "\n" in call.text else call.text[3:].strip()
        return Verdict(p, reason[:500], valid, call)

    # --- sandbox work (blocking; run in threads) ----------------------------------------------

    def _listing(self, task) -> str:
        sb = self.factory.create(task)
        try:
            r = sb.exec("git ls-files | head -200", timeout=30)
            return r.stdout.strip()
        finally:
            sb.close()

    def _probe(self, task, plan: PlanRecord) -> None:
        sb = self.factory.create(task)
        try:
            r = sb.exec(plan.probe, timeout=self.cfg.probe_timeout_s)
            files = {}
            for path in plan.files:
                try:
                    files[path] = sb.read_file(path)[: self.cfg.max_file_chars]
                except (FileNotFoundError, OSError):
                    files[path] = None
        finally:
            sb.close()
        passed = r.returncode == 0
        confirmed = None if r.timed_out else (passed == (plan.expect == "pass"))
        plan.probe_result = {"command": plan.probe, "exit_code": r.returncode, "output": r.output[-3000:],
                             "confirmed": confirmed, "files": files}  # fmt: skip

    def _apply_and_check(self, task, plan: PlanRecord, patch: PatchRecord) -> None:
        sb = self.factory.create(task)
        try:
            for path, body in patch.files.items():
                sb.write_file(path, body)
            patch.diff = sb.diff()
            probe = sb.exec(plan.probe, timeout=self.cfg.probe_timeout_s)
            tests = sb.exec("python -m pytest -q -x -p no:cacheprovider 2>&1 | tail -30",
                            timeout=self.cfg.check_timeout_s)  # fmt: skip
        finally:
            sb.close()
        patch.visible = {"probe_exit": probe.returncode, "tests_exit": tests.returncode, "tests_output": tests.output}

    def _grade(self, task, record, patch: PatchRecord) -> None:
        g = self.grader.grade(task, patch.diff, record)
        patch.hidden_resolved = bool(g.resolved and not g.protected_edits)

    # --- the episode ---------------------------------------------------------------------------

    async def _plans(self, ep: TriEpisode, task, listing: str, feedback: list[str], r: int) -> list[PlanRecord]:
        user = P.planner_prompt(task.instruction, listing, feedback)
        calls = await asyncio.gather(*(self._generate(ep, "planner", P.PLANNER_SYSTEM, user, f"r{r}p{k}")
                                       for k in range(self.cfg.plans_per_round)))  # fmt: skip
        plans = []
        for c in calls:
            pp = P.parse_plan(c.text)
            plans.append(PlanRecord(c.text, c, pp.valid, pp.hypothesis, pp.files, pp.probe, pp.expect))
        return plans

    async def _implement(self, ep: TriEpisode, task, record, r: int, k: int, plan: PlanRecord) -> None:
        pp = P.parse_plan(plan.text)
        user = P.coder_prompt(task.instruction, pp, plan.probe_result, plan.probe_result.get("files", {}))
        calls = await asyncio.gather(*(self._generate(ep, "coder", P.CODER_SYSTEM, user, f"r{r}p{k}c{j}")
                                       for j in range(self.cfg.patches_per_plan)))  # fmt: skip
        for c in calls:
            files, err = P.parse_patch(c.text)
            plan.patches.append(PatchRecord(files, c, not err))
        valid = [p for p in plan.patches if p.valid]
        await asyncio.gather(*(asyncio.to_thread(self._apply_and_check, task, plan, p) for p in valid))
        verdicts = await asyncio.gather(*(
            self._verdict(ep, P.ROUTER_PATCH_SYSTEM, P.router_patch_prompt(task.instruction, pp, p.diff, p.visible),
                          f"r{r}p{k}v{j}")
            for j, p in enumerate(valid)))  # fmt: skip
        for p, v in zip(valid, verdicts, strict=True):
            p.verdict = v
        await asyncio.gather(*(asyncio.to_thread(self._grade, task, record, p) for p in valid))
        for p in plan.patches:
            if not p.valid:
                p.hidden_resolved = False  # unparseable patch: an observed failure

    async def run(self, task, seed: int = 0, group_id: str | None = None) -> TriEpisode:
        ep = TriEpisode(uuid.uuid4().hex[:12], task.task_id, group_id or task.task_id, seed)
        try:
            record = await asyncio.to_thread(self.validate, task)
            if not record.valid:
                raise InfraError(f"task failed validation: {record.reason}")
            listing = await asyncio.to_thread(self._listing, task)
            feedback: list[str] = []
            for r in range(self.cfg.max_rounds):
                plans = await self._plans(ep, task, listing, feedback, r)
                valid = [k for k, p in enumerate(plans) if p.valid]
                await asyncio.gather(*(asyncio.to_thread(self._probe, task, plans[k]) for k in valid))
                verdicts = await asyncio.gather(*(
                    self._verdict(ep, P.ROUTER_PLAN_SYSTEM, P.router_plan_prompt(
                        task.instruction, P.parse_plan(plans[k].text), plans[k].probe_result), f"r{r}p{k}v")
                    for k in valid))  # fmt: skip
                for k, v in zip(valid, verdicts, strict=True):
                    plans[k].verdict = v
                dec = self._gate(plans, force=r == self.cfg.max_rounds - 1)
                ep.rounds.append(Round(plans, dec))
                if not any(dec.executed):
                    feedback += self._feedback(plans)
                    continue
                await asyncio.gather(*(self._implement(ep, task, record, r, k, plans[k])
                                       for k, e in enumerate(dec.executed) if e))  # fmt: skip
                break
            self._submit(ep)
        except InfraError as e:
            ep.error = str(e)
        return ep

    def _gate(self, plans: list[PlanRecord], force: bool):
        """Gate the valid plans; invalid ones never run (and are credited as observed failures)."""
        valid = [k for k, p in enumerate(plans) if p.valid]
        n = len(plans)
        sub = self.gate.decide([plans[k].verdict.p if plans[k].verdict else 0.0 for k in valid], force=force)
        q, acc, ex = [0.0] * n, [False] * n, [False] * n
        for i, k in enumerate(valid):
            q[k], acc[k], ex[k] = sub.q[i], sub.accepted[i], sub.executed[i]
        sub.p = [plans[k].verdict.p if plans[k].verdict else 0.0 for k in range(n)]
        sub.q, sub.accepted, sub.executed, sub.main = q, acc, ex, [valid[i] for i in sub.main]
        return sub

    @staticmethod
    def _feedback(plans: list[PlanRecord]) -> list[str]:
        out = []
        for p in plans:
            if not p.valid:
                out.append(f"- an unusable plan (format error): {p.text[:200]!r}")
            else:
                why = p.verdict.reason if p.verdict else "no verdict"
                out.append(f"- rejected (p={p.verdict.p if p.verdict else 0:.2f}): {p.hypothesis} | "
                           f"{P.probe_summary(p.probe_result)[:400]} | router: {why}")  # fmt: skip
        return out

    @staticmethod
    def _submit(ep: TriEpisode) -> None:
        """Submit the best-scored patch among the main plans' patches; reward = its hidden-test outcome."""
        best, best_s = None, -1.0
        for r, k in ep.main_plans():
            for j, patch in enumerate(ep.rounds[r].plans[k].patches):
                s = patch.verdict.p if (patch.verdict and patch.valid) else -0.5
                if s > best_s:
                    best, best_s = (r, k, j), s
        ep.final = best
        fp = ep.final_patch()
        ep.reward = fp.y if fp is not None else 0.0
