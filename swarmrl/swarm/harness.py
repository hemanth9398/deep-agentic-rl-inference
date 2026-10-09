"""Harness controls, identical in eval and training."""

from __future__ import annotations

from collections import Counter, deque

from swarmrl.swarm.config import SwarmConfig
from swarmrl.traces.schema import Termination

MIN_NEW_TOKENS = 32


class Harness:
    def __init__(self, cfg: SwarmConfig):
        self.cfg = cfg
        self.episode_turns = 0
        self.role_turns: Counter[str] = Counter()
        self.tokens_generated = 0
        self.consecutive_format_errors = 0
        self.recent_calls: dict[str, deque[str]] = {}
        self.pair_handoffs: Counter[frozenset[str]] = Counter()
        self.repeat_calls = 0
        self.ping_pong_events = 0

    def remaining_turns(self, role: str) -> int:
        return min(
            self.cfg.episode_turn_budget - self.episode_turns,
            self.cfg.role(role).turn_budget - self.role_turns[role],
        )

    def pre_turn(self, role: str, context_len: int) -> tuple[Termination | None, int]:
        """Check budgets before a turn. Returns (termination, max_new_tokens)."""
        if self.remaining_turns(role) <= 0:
            return Termination.TURN_BUDGET, 0
        token_room = self.cfg.episode_token_budget - self.tokens_generated
        if token_room < MIN_NEW_TOKENS:
            return Termination.TOKEN_BUDGET, 0
        ctx_room = self.cfg.context_limit - context_len
        if ctx_room < MIN_NEW_TOKENS:
            return Termination.CONTEXT_LIMIT, 0
        return None, min(self.cfg.sampling.max_new_tokens, token_room, ctx_room)

    def post_turn(self, role: str, n_tokens: int) -> None:
        self.episode_turns += 1
        self.role_turns[role] += 1
        self.tokens_generated += n_tokens

    def warning(self, role: str) -> str | None:
        left = self.remaining_turns(role)
        if left == self.cfg.warn_turns_remaining:
            return f"[harness] {left} turns remain. Wrap up: finish, hand off or submit."
        return None

    def record_format(self, is_format_error: bool) -> Termination | None:
        self.consecutive_format_errors = self.consecutive_format_errors + 1 if is_format_error else 0
        if self.consecutive_format_errors >= self.cfg.max_consecutive_format_errors:
            return Termination.FORMAT_ERRORS
        return None

    def is_repeat(self, role: str, call_key: str) -> bool:
        q = self.recent_calls.setdefault(role, deque(maxlen=self.cfg.repeat_call_window))
        rep = call_key in q
        q.append(call_key)
        if rep:
            self.repeat_calls += 1
        return rep

    def clear_recent(self, role: str) -> None:
        """New information arrived (a hand-off back); re-running an earlier call is legitimate."""
        self.recent_calls.pop(role, None)

    def record_handoff(self, src: str, dst: str) -> Termination | None:
        pair = frozenset((src, dst))
        self.pair_handoffs[pair] += 1
        if self.pair_handoffs[pair] > self.cfg.max_handoffs_per_pair:
            self.ping_pong_events += 1
            return Termination.PING_PONG
        return None
