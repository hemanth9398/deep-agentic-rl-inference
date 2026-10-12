"""Policy clients: token ids in, sampled ids + per-token log-probs out."""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from swarmrl.sandbox.base import InfraError
from swarmrl.swarm.config import SamplingConfig


@dataclass
class Generation:
    output_ids: list[int]
    logprobs: list[float]
    finish_reason: str  # "stop" | "length" | "abort"
    latency_s: float = 0.0


class PolicyClient(abc.ABC):
    """Token-in, token-out generation. Implementations must return the sampled ids and their
    log-probs exactly as the sampler produced them (never decoded and re-encoded)."""

    @abc.abstractmethod
    async def generate(
        self, input_ids: list[int], sampling: SamplingConfig, stop_ids: list[int], session_key: str, max_new_tokens: int
    ) -> Generation:
        """`session_key` is "<episode_id>:<role>"; servers use it for sticky routing."""


class StickyRouter:
    """Pins every turn of one (episode, role) to the same server so RadixAttention keeps its prefix.

    Rendezvous hashing: adding or removing a server only moves the sessions that lived on it.
    """

    def __init__(self, servers: list[str]):
        if not servers:
            raise ValueError("no servers")
        self.servers = list(servers)

    def route(self, session_key: str) -> str:
        def score(server: str) -> int:
            return int.from_bytes(hashlib.blake2b(f"{server}|{session_key}".encode(), digest_size=8).digest(), "big")

        return max(self.servers, key=score)


def _post(url: str, payload: dict, timeout: float, headers: dict[str, str] | None = None) -> dict:
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class SGLangClient(PolicyClient):
    """Native SGLang /generate with input_ids and return_logprob (no chat endpoint, no detokenize round trip)."""

    def __init__(
        self,
        servers: list[str] | str,
        timeout_s: float = 600.0,
        retries: int = 3,
        lora_name: str | None = None,
        routing_header: str | None = None,
    ):
        self.router = StickyRouter([servers] if isinstance(servers, str) else servers)
        self.timeout_s = timeout_s
        self.retries = retries
        self.lora_name = lora_name  # set by LoRA weight sync when adapters are served unmerged
        # When `servers` is a router or gateway rather than the engines themselves, the session key is sent in
        # this header so the router keeps a role's turns on one engine (prefix-cache hits): e.g.
        # "X-SMG-Routing-Key" for sgl-router consistent hashing, "x-session-id" for llm-d session affinity.
        self.routing_header = routing_header

    def _payload(self, input_ids, sampling: SamplingConfig, stop_ids, max_new_tokens) -> dict:
        p = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": stop_ids,
                "skip_special_tokens": False,
                "no_stop_trim": True,  # keep the end-of-turn token: it carries loss
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        }
        if self.lora_name:
            p["lora_path"] = self.lora_name
        return p

    @staticmethod
    def parse_response(resp: dict) -> Generation:
        meta = resp.get("meta_info", {})
        lps = meta.get("output_token_logprobs") or []
        ids = [int(x[1]) for x in lps]
        logprobs = [float(x[0]) for x in lps]
        if "output_ids" in resp and list(resp["output_ids"]) != ids:
            raise InfraError("SGLang output_ids disagree with output_token_logprobs ids")
        fr = meta.get("finish_reason") or {}
        reason = fr.get("type", "stop") if isinstance(fr, dict) else str(fr)
        return Generation(ids, logprobs, reason)

    def _generate_sync(self, input_ids, sampling, stop_ids, session_key, max_new_tokens) -> Generation:
        url = self.router.route(session_key).rstrip("/") + "/generate"
        payload = self._payload(input_ids, sampling, stop_ids, max_new_tokens)
        last: Exception | None = None
        for attempt in range(self.retries):
            t0 = time.monotonic()
            try:
                hdrs = {self.routing_header: session_key} if self.routing_header else None
                g = self.parse_response(_post(url, payload, self.timeout_s, hdrs))
                g.latency_s = time.monotonic() - t0
                return g
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                time.sleep(min(2**attempt, 8) * (0.5 + random.random()))
        raise InfraError(f"SGLang unreachable at {url}: {last}")

    async def generate(self, input_ids, sampling, stop_ids, session_key, max_new_tokens) -> Generation:
        return await asyncio.to_thread(self._generate_sync, input_ids, sampling, stop_ids, session_key, max_new_tokens)


class ScriptedPolicy(PolicyClient):
    """Test/dry-run policy. `script(role, role_turn, context_text) -> reply text` decides each turn.

    Returns plausible log-probs so the full data path (traces, packing, losses) can be exercised
    without a GPU. Session keys are "<episode_id>:<role>".
    """

    def __init__(self, chat, script: Callable[[str, int, str], str], append_eot: bool = True):
        self.chat = chat
        self.script = script
        self.append_eot = append_eot
        self._turns: dict[str, int] = {}
        self.calls: list[tuple[str, int]] = []

    async def generate(self, input_ids, sampling, stop_ids, session_key, max_new_tokens) -> Generation:
        role = session_key.rsplit(":", 1)[-1]
        n = self._turns.get(session_key, 0)
        self._turns[session_key] = n + 1
        self.calls.append((session_key, len(input_ids)))
        text = self.script(role, n, self.chat.decode(input_ids))
        ids = self.chat.tok.encode(text)
        reason = "stop"
        if len(ids) >= max_new_tokens:
            ids, reason = ids[:max_new_tokens], "length"
        elif self.append_eot:
            ids = ids + [self.chat.end_of_turn_id]
        # Deterministic, varied log-probs so alignment and mismatch checks are meaningful in tests.
        lps = [-0.05 - ((tok * 31 + len(input_ids) + i * 17) % 97) / 50 for i, tok in enumerate(ids)]
        return Generation(ids, lps, reason)
