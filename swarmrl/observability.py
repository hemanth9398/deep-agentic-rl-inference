"""Logging setup and a metrics sink shared by every entrypoint."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any


def setup_logging(level: str | int | None = None, json_format: bool | None = None) -> None:
    """Configure root logging once. SWARMRL_LOG_LEVEL / SWARMRL_LOG_JSON override the defaults."""
    level = level or os.environ.get("SWARMRL_LOG_LEVEL", "INFO")
    if json_format is None:
        json_format = os.environ.get("SWARMRL_LOG_JSON", "0") == "1"
    handler = logging.StreamHandler(sys.stderr)
    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": record.created, "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d)


class MetricsSink:
    """Appends one JSON object per call to a JSONL file (and logs a one-line summary).

    Usable as RLLoop.log. Thread-safe; values that are not JSON-native are stringified.
    """

    SUMMARY = ("step", "reward_mean", "resolve_rate", "mismatch_kl", "informative_groups")

    def __init__(self, path: str | None = None, summary_keys: tuple[str, ...] = SUMMARY):
        self.path = path
        self.summary_keys = summary_keys
        self._lock = threading.Lock()
        self._log = logging.getLogger("swarmrl.metrics")
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def __call__(self, metrics: dict[str, Any]) -> None:
        row = {"time": time.time(), **metrics}
        if self.path:
            with self._lock, open(self.path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        parts = []
        for k in self.summary_keys:
            if k in metrics:
                v = metrics[k]
                parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
        self._log.info(" ".join(parts))
