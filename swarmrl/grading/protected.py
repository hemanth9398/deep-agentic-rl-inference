"""Protected paths: edits to tests or test configuration never reach grading."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable

_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def is_protected(path: str, patterns: Iterable[str]) -> bool:
    path = path.lstrip("./")
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        # `**/x` should also match a top-level `x`
        if pat.startswith("**/") and fnmatch.fnmatch(path, pat[3:]):
            return True
    return False


def split_diff(diff: str) -> list[tuple[str, str]]:
    """Split a git diff into (path, file_patch) pieces, one per file."""
    starts = [m.start() for m in _DIFF_HEADER.finditer(diff)]
    out = []
    for i, s in enumerate(starts):
        chunk = diff[s : starts[i + 1] if i + 1 < len(starts) else len(diff)]
        m = _DIFF_HEADER.match(chunk)
        assert m
        # For renames the destination path decides protection; check both to be safe.
        a, b = m.group(1), m.group(2)
        out.append((b if b != "/dev/null" else a, chunk if chunk.endswith("\n") else chunk + "\n"))
        if a != b:
            out[-1] = (f"{a}\0{b}", out[-1][1])
    return out


def filter_diff(diff: str, patterns: Iterable[str]) -> tuple[str, list[str]]:
    """Drop every file patch that touches a protected path. Returns (kept_diff, dropped_paths)."""
    patterns = tuple(patterns)
    kept, dropped = [], []
    for path, patch in split_diff(diff):
        paths = path.split("\0")
        if any(is_protected(p, patterns) for p in paths):
            dropped.extend(paths)
        else:
            kept.append(patch)
    return "".join(kept), dropped


def changed_paths(diff: str) -> list[str]:
    return [p for path, _ in split_diff(diff) for p in path.split("\0")]
