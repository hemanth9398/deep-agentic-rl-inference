"""The swarm episode loop: init stage (CPU), run stage (GPU + CPU), reward stage (CPU).

One policy plays every role. Each role has its own token context; a hand-off tool switches
the active role. Every LLM call becomes a Transition with the exact ids and log-probs the
server returned.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import traceback
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from swarmrl.grading.grader import CleanRoomGrader
from swarmrl.grading.reward import RubricTerm, total_reward
from swarmrl.inference.chat import ChatFormat
from swarmrl.inference.client import PolicyClient
from swarmrl.sandbox.base import InfraError, Sandbox, SandboxFactory
from swarmrl.swarm.config import SwarmConfig
from swarmrl.swarm.harness import Harness
from swarmrl.swarm.parser import ParseError, parse_tool_call
from swarmrl.swarm.tools import ToolContext, ToolOutcome, describe_tools, execute
from swarmrl.tasks.validate import ValidationRecord
from swarmrl.traces.schema import AgentSpan, Episode, ErrorKind, GradeSummary, Termination, Transition

log = logging.getLogger(__name__)


@dataclass
class EpisodeState:
    episode: Episode
    task: object
    sandbox: Sandbox | None = None
    harness: Harness | None = None
    contexts: dict[str, list[int]] = field(default_factory=dict)
    role_turns: dict[str, int] = field(default_factory=dict)
    pending_return: dict[str, str] = field(default_factory=dict)
    t0: float = field(default_factory=time.monotonic)
    abandoned: bool = False  # set when the episode was failed/cancelled; late resources are closed at once


class EpisodeRunner:
    def __init__(
        self,
        config: SwarmConfig,
        policy: PolicyClient,
        chat: ChatFormat,
        sandbox_factory: SandboxFactory,
        grader: CleanRoomGrader,
        validation: Callable[[object], ValidationRecord],
        model_checkpoint: str = "base",
        rubric: Sequence[RubricTerm] = (),
        require_submit_for_reward: bool = True,
    ):
        self.cfg = config
        self.policy = policy
        self.chat = chat
        self.factory = sandbox_factory
        self.grader = grader
        self.validation = validation
        self.model_checkpoint = model_checkpoint
        self.rubric = rubric
        self.require_submit = require_submit_for_reward
        self.fingerprint = config.fingerprint()

    # --- prompts -------------------------------------------------------------------

    def _system(self, role: str) -> str:
        rc = self.cfg.role(role)
        return f"{rc.system_prompt}\n\nYour tools:\n{describe_tools(rc.tools)}"

    def _first_user(self, task, sender: str | None, note: str) -> str:
        msg = f"<issue>\n{task.instruction.strip()}\n</issue>"
        if sender:
            msg += f"\n\nMessage from {sender}:\n{note}"
        return msg

    # --- stages --------------------------------------------------------------------

    def new_state(self, task, seed: int, split: str | None = None, group_id: str | None = None) -> EpisodeState:
        ep = Episode(
            episode_id=uuid.uuid4().hex,
            task_id=task.task_id,
            split=split or task.split,
            swarm_fingerprint=self.fingerprint,
            model_checkpoint=self.model_checkpoint,
            seed=seed,
            group_id=group_id,
        )
        return EpisodeState(ep, task)

    def init_stage(self, st: EpisodeState) -> None:
        """CPU: start the task container, reset the repo, build the entry role's prompt."""
        sb = self.factory.create(st.task)
        if st.abandoned:  # timed out while the container was starting
            sb.close()
            return
        st.sandbox = sb
        st.harness = Harness(self.cfg)
        entry = self.cfg.entry_role
        st.contexts[entry] = self.chat.start(self._system(entry), self._first_user(st.task, None, ""))

    async def run_stage(self, st: EpisodeState) -> None:
        cfg, h, ep = self.cfg, st.harness, st.episode
        assert h is not None and st.sandbox is not None
        active = cfg.entry_role
        span = AgentSpan(active, 0, 0)
        termination: Termination | None = None

        while termination is None:
            ctx = st.contexts[active]
            termination, max_new = h.pre_turn(active, len(ctx))
            if termination:
                span.handoff_reason = termination.value
                break
            role_turn = st.role_turns.get(active, 0)
            st.role_turns[active] = role_turn + 1
            gen = await self.policy.generate(
                list(ctx), cfg.sampling, self.chat.stop_ids, f"{ep.episode_id}:{active}", max_new
            )
            h.post_turn(active, len(gen.output_ids))
            span.tokens_used += len(gen.output_ids)
            closing = self.chat.closing_ids(gen.output_ids)
            text = self.chat.decode(gen.output_ids)

            rc = cfg.role(active)
            parsed = parse_tool_call(text, rc.tools)
            outcome: ToolOutcome
            tool_call = None
            if isinstance(parsed, ParseError):
                outcome = ToolOutcome(parsed.message, parsed.kind)
            else:
                tool_call = parsed.as_dict()
                if h.is_repeat(active, parsed.key()):
                    outcome = ToolOutcome(
                        "You already made this exact call and nothing has changed. Try something different.",
                        ErrorKind.REPEAT_CALL,
                    )
                else:
                    outcome = await asyncio.to_thread(execute, parsed, ToolContext(st.sandbox, cfg, rc))

            ep.transitions.append(
                Transition(
                    episode_id=ep.episode_id,
                    role=active,
                    turn=len(ep.transitions),
                    role_turn=role_turn,
                    input_ids=list(ctx),
                    output_ids=list(gen.output_ids),
                    rollout_logprobs=list(gen.logprobs),
                    tool_call=tool_call,
                    observation=outcome.observation,
                    error_kind=outcome.error_kind.value if outcome.error_kind else None,
                    latency_s=gen.latency_s,
                    finish_reason=gen.finish_reason,
                    closing_ids=closing,
                )  # fmt: skip
            )
            st.contexts[active] = ctx + list(gen.output_ids) + closing

            is_fmt = outcome.error_kind is not None and outcome.error_kind.is_format_error
            termination = h.record_format(is_fmt)
            if termination:
                span.handoff_reason = termination.value
                break
            if outcome.submit:
                termination = Termination.SUBMIT
                span.handoff_reason = "submit"
                break
            if outcome.handoff_to:
                target = outcome.handoff_to
                termination = h.record_handoff(active, target)
                span.end_turn, span.handoff_reason, span.handoff_to = len(ep.transitions) - 1, tool_call["name"], target
                ep.spans.append(span)
                if termination:
                    span = AgentSpan(target, len(ep.transitions), len(ep.transitions) - 1, termination.value)
                    break
                # The source sees its tool observation only when control comes back to it.
                st.pending_return[active] = outcome.observation
                self._activate(st, target, sender=active, note=outcome.note)
                active = target
                span = AgentSpan(active, len(ep.transitions), len(ep.transitions))
                continue
            obs = outcome.observation
            if (w := h.warning(active)) is not None:
                obs = f"{obs}\n\n{w}"
            st.contexts[active] = st.contexts[active] + self.chat.user_turn(obs)

        span.end_turn = len(ep.transitions) - 1
        if span.end_turn >= span.start_turn or not ep.spans:
            ep.spans.append(span)
        ep.termination = termination
        ep.repeat_calls = h.repeat_calls
        ep.ping_pong_events = h.ping_pong_events

    def _activate(self, st: EpisodeState, role: str, sender: str, note: str) -> None:
        if role not in st.contexts:
            st.contexts[role] = self.chat.start(self._system(role), self._first_user(st.task, sender, note))
            return
        prefix = st.pending_return.pop(role, "")
        msg = f"{prefix}\n\nMessage from {sender}:\n{note}".strip()
        st.contexts[role] = st.contexts[role] + self.chat.user_turn(msg)
        st.harness.clear_recent(role)

    def reward_stage(self, st: EpisodeState) -> None:
        """CPU: extract the diff, grade it in a clean container, attach the team reward."""
        ep = st.episode
        try:
            if st.sandbox is not None:
                ep.final_diff = st.sandbox.diff()
        finally:
            self._close(st)
        if not ep.termination.scored:
            return
        record = self.validation(st.task)
        if not record.valid:
            raise InfraError(f"task {st.task.task_id} failed validation ({record.reason}); it must not be in any split")
        grade = self.grader.grade(st.task, ep.final_diff, record)
        ep.protected_edits = grade.protected_edits
        ep.grade = GradeSummary(
            grade.resolved, grade.f2p_passed, grade.f2p_total, grade.p2p_passed, grade.p2p_total,
            grade.apply_failed, grade.failed_tests,
        )  # fmt: skip
        reward, details = total_reward(grade, ep, self.rubric)
        if self.require_submit and not ep.submitted:
            reward, details = 0.0, details | {"not_submitted": 1.0}
        ep.team_reward = reward
        ep.metadata["reward_details"] = details

    def _close(self, st: EpisodeState) -> None:
        if st.sandbox is not None:
            try:
                st.sandbox.close()
            finally:
                st.sandbox = None

    def _infra_fail(self, st: EpisodeState, e: BaseException, internal: bool = False) -> None:
        st.abandoned = True
        st.episode.termination = Termination.INFRA_ERROR
        st.episode.team_reward = None
        st.episode.metadata["infra_error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
        if internal:
            st.episode.metadata["internal_error"] = True  # a bug in our code, not the environment: page someone
        self._close(st)

    async def execute(
        self,
        st: EpisodeState,
        timeout_s: float | None = None,
        sems: tuple[asyncio.Semaphore, asyncio.Semaphore, asyncio.Semaphore] | None = None,
    ) -> Episode:
        """Run all three stages with failure isolation: whatever goes wrong, this returns an Episode.

        sems = (init, in-flight, reward) pools from the Dispatcher; None runs unbounded.
        """
        init_sem, run_sem, reward_sem = sems or (contextlib.nullcontext(),) * 3

        async def stages() -> None:
            async with init_sem:
                await asyncio.to_thread(self.init_stage, st)
            await self.run_stage(st)
            async with reward_sem:
                await asyncio.to_thread(self.reward_stage, st)

        async with run_sem:  # the wall clock starts once the episode holds an in-flight slot
            st.t0 = time.monotonic()
            try:
                await asyncio.wait_for(stages(), timeout_s)
            except InfraError as e:
                log.warning("episode %s (%s): infra error: %s", st.episode.episode_id, st.task.task_id, e)
                self._infra_fail(st, e)
            except TimeoutError:
                log.warning(
                    "episode %s (%s): wall-clock timeout %ss", st.episode.episode_id, st.task.task_id, timeout_s
                )
                self._infra_fail(st, TimeoutError(f"episode exceeded {timeout_s}s wall clock"))
            except Exception as e:
                log.exception("episode %s (%s): internal error", st.episode.episode_id, st.task.task_id)
                self._infra_fail(st, e, internal=True)
        st.episode.wall_time_s = time.monotonic() - st.t0
        return st.episode

    async def run(self, task, seed: int = 0, split: str | None = None, group_id: str | None = None,
                  timeout_s: float | None = None) -> Episode:  # fmt: skip
        """One episode, all stages back to back (the Dispatcher interleaves stages across episodes)."""
        return await self.execute(self.new_state(task, seed, split, group_id), timeout_s or task.agent_timeout_s)
