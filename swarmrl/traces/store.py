"""Trace store: one JSON file per episode plus an append-only JSONL index.

The local-directory layout is the reference implementation; object storage (S3/GCS)
mounts the same layout. The store is the single input to evals, SFT and RL.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from swarmrl.traces.schema import Episode


class TraceStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        self._index = self.root / "index.jsonl"
        self._lock = threading.Lock()

    def _path(self, episode_id: str) -> Path:
        # Shard by prefix so a directory never holds millions of files.
        return self.root / "episodes" / episode_id[:2] / f"{episode_id}.json"

    def put(self, episode: Episode) -> None:
        path = self._path(episode.episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(episode.to_json())
        tmp.replace(path)
        with self._lock, self._index.open("a") as f:
            f.write(json.dumps(episode.summary()) + "\n")

    def get(self, episode_id: str) -> Episode:
        return Episode.from_json(self._path(episode_id).read_text())

    def index(self) -> list[dict[str, Any]]:
        """Latest index row per episode (re-puts overwrite earlier rows)."""
        if not self._index.exists():
            return []
        rows: dict[str, dict[str, Any]] = {}
        with self._index.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["episode_id"]] = row
        return list(rows.values())

    def query(self, **equals: Any) -> list[dict[str, Any]]:
        return [r for r in self.index() if all(r.get(k) == v for k, v in equals.items())]

    def episodes(
        self, where: Callable[[dict[str, Any]], bool] | None = None, **equals: Any
    ) -> Iterator[Episode]:
        for row in self.query(**equals):
            if where is None or where(row):
                yield self.get(row["episode_id"])
