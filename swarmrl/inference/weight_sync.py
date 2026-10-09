"""Weight sync after each RL update.

| Mode          | When                      | Mechanism                                                         |
| Colocated     | stages 1 and 2 (sync)     | SGLang sleeps during training (weights + KV released), then wakes |
|               |                           | and receives the new weights in place                             |
| Disaggregated | stage 3 (async)           | separate GPU pools; NCCL broadcast every K steps (SkyRL-train's   |
|               |                           | inference-engine client drives it; see integrations/skyrl.py)     |

In both modes the KV cache is flushed after an update, so no prefix computed by old weights
leaks into new rollouts. LoRA is served one of two ways; pick one and run
scripts/verify/weight_sync.py against it:
  - "merged": merge the adapter into full weights on disk, /update_weights_from_disk
  - "adapter": /load_lora_adapter under a new name per version, route requests to it
"""

from __future__ import annotations

import abc
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Literal

from swarmrl.sandbox.base import InfraError


class SGLangAdmin:
    def __init__(self, servers: list[str], timeout_s: float = 900.0):
        self.servers = [s.rstrip("/") for s in servers]
        self.timeout_s = timeout_s

    def _call(self, path: str, payload: dict | None = None) -> list[dict]:
        out = []
        for s in self.servers:
            req = urllib.request.Request(
                s + path, data=json.dumps(payload or {}).encode(), headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    body = r.read()
            except Exception as e:
                raise InfraError(f"{s}{path} failed: {e}") from e
            try:
                out.append(json.loads(body) if body else {})
            except json.JSONDecodeError:
                out.append({"raw": body.decode(errors="replace")})
        return out

    def flush_cache(self) -> None:
        self._call("/flush_cache")

    def release_memory(self, tags: list[str] | None = None) -> None:
        self._call("/release_memory_occupation", {"tags": tags} if tags else {})

    def resume_memory(self, tags: list[str] | None = None) -> None:
        self._call("/resume_memory_occupation", {"tags": tags} if tags else {})

    def update_weights_from_disk(self, model_path: str, weight_version: str | None = None) -> None:
        res = self._call(
            "/update_weights_from_disk",
            {"model_path": model_path, "weight_version": weight_version, "abort_all_requests": True},
        )
        for r in res:
            if r.get("success") is False:
                raise InfraError(f"update_weights_from_disk failed: {r.get('message')}")

    def load_lora_adapter(self, name: str, path: str) -> None:
        for r in self._call("/load_lora_adapter", {"lora_name": name, "lora_path": path}):
            if r.get("success") is False:
                raise InfraError(f"load_lora_adapter failed: {r.get('error_message') or r}")

    def unload_lora_adapter(self, name: str) -> None:
        self._call("/unload_lora_adapter", {"lora_name": name})

    def model_info(self) -> dict:
        req = urllib.request.Request(self.servers[0] + "/get_model_info")
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())


class WeightSync(abc.ABC):
    """Moves trainer weights into the inference servers between RL steps."""

    @abc.abstractmethod
    def before_training(self) -> None:
        """Called before the trainer runs (colocated: free the servers' GPU memory)."""

    @abc.abstractmethod
    def sync(self, checkpoint_path: str, version: int) -> None:
        """Load `checkpoint_path` as policy `version` everywhere and flush every KV cache."""


@dataclass
class DiskWeightSync(WeightSync):
    """Push a checkpoint written by the trainer into every SGLang server, then flush the KV cache."""

    admin: SGLangAdmin
    mode: Literal["merged", "adapter"] = "merged"
    client: object | None = None  # SGLangClient, to switch lora_name in adapter mode
    colocated: bool = True
    history: list[str] = field(default_factory=list)
    released: bool = False

    def before_training(self) -> None:
        """Colocated: free GPU memory for the trainer."""
        if self.colocated and not self.released:
            self.admin.release_memory(["kv_cache", "weights"])
            self.released = True

    def sync(self, checkpoint_path: str, version: int) -> None:
        if self.colocated and self.released:
            self.admin.resume_memory(["weights"])
        if self.mode == "merged":
            self.admin.update_weights_from_disk(checkpoint_path, weight_version=str(version))
        else:
            name = f"policy-v{version}"
            self.admin.load_lora_adapter(name, checkpoint_path)
            if self.client is not None:
                self.client.lora_name = name
            if self.history:
                self.admin.unload_lora_adapter(self.history[-1])
            self.history.append(name)
        if self.colocated and self.released:
            self.admin.resume_memory(["kv_cache"])
            self.released = False
        self.admin.flush_cache()  # never skip: stale prefixes would leak old-policy activations


@dataclass
class NullWeightSync(WeightSync):
    """Records calls without touching a server: dry runs, and stage 0 when the server already holds the weights."""

    calls: list[tuple[str, int]] = field(default_factory=list)

    def before_training(self) -> None:
        self.calls.append(("before_training", -1))

    def sync(self, checkpoint_path: str, version: int) -> None:
        self.calls.append((checkpoint_path, version))
