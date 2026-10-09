"""Procedural mutants of training repos.

Inject one small, realistic bug (flipped comparison, swapped operator, off-by-one constant,
and/or swap) into a source file, keep it only if some test fails and the suite still
collects, and emit a Harbor task whose oracle is the reverse patch. The instruction states
symptoms (failing tests and their errors), never the fix.
"""

from __future__ import annotations

import ast
import difflib
import io
import random
import shutil
import tempfile
import tokenize
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from swarmrl.grading.protected import is_protected
from swarmrl.grading.pytest_runner import run_pytest
from swarmrl.sandbox.local import LocalSandbox
from swarmrl.swarm.config import SwarmConfig
from swarmrl.tasks.harbor import write_task

_SWAPS = {
    "<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "==",
    "+": "-", "-": "+", "*": "/", "//": "/", "and": "or", "or": "and",
}  # fmt: skip


@dataclass(frozen=True)
class Mutation:
    path: str
    line: int  # 1-based
    col: int
    end_col: int
    old: str
    new: str

    def describe(self) -> str:
        return f"{self.path}:{self.line} {self.old!r} -> {self.new!r}"


def _tokens(src: str) -> list[tokenize.TokenInfo]:
    return list(tokenize.generate_tokens(io.StringIO(src).readline))


def _op_between(tokens, line: int, start_col: int, end_col: int, wanted: set[str]):
    for t in tokens:
        if t.start[0] == line and start_col <= t.start[1] and t.end[1] <= end_col and t.string in wanted:
            return t
    return None


def find_mutations(path: str, src: str) -> list[Mutation]:
    tree = ast.parse(src)
    toks = _tokens(src)
    out: list[Mutation] = []
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != getattr(node, "end_lineno", None):
            continue  # single-line nodes only, so positions are unambiguous
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = node.left, node.comparators[0]
            t = _op_between(toks, node.lineno, left.end_col_offset, right.col_offset, set(_SWAPS))
            if t and t.string in ("<", "<=", ">", ">=", "==", "!="):
                out.append(Mutation(path, t.start[0], t.start[1], t.end[1], t.string, _SWAPS[t.string]))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
            t = _op_between(toks, node.lineno, node.left.end_col_offset, node.right.col_offset, {"+", "-", "*", "//"})
            if t:
                out.append(Mutation(path, t.start[0], t.start[1], t.end[1], t.string, _SWAPS[t.string]))
        elif isinstance(node, ast.BoolOp) and len(node.values) == 2:
            a, b = node.values
            t = _op_between(toks, node.lineno, a.end_col_offset, b.col_offset, {"and", "or"})
            if t:
                out.append(Mutation(path, t.start[0], t.start[1], t.end[1], t.string, _SWAPS[t.string]))
        elif isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1, 2):
            old = src.splitlines()[node.lineno - 1][node.col_offset : node.end_col_offset]
            if old == str(node.value):
                out.append(Mutation(path, node.lineno, node.col_offset, node.end_col_offset, old, str(node.value + 1)))
    return out


def apply_mutation(src: str, m: Mutation) -> str:
    lines = src.splitlines(keepends=True)
    line = lines[m.line - 1]
    assert line[m.col : m.end_col] == m.old, (line, m)
    lines[m.line - 1] = line[: m.col] + m.new + line[m.end_col :]
    return "".join(lines)


def git_style_patch(path: str, before: str, after: str) -> str:
    body = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{path}", f"b/{path}", n=3
    )
    return f"diff --git a/{path} b/{path}\n" + "".join(body)


def candidate_files(repo: Path, protected: tuple[str, ...]) -> Iterator[Path]:
    for p in sorted(repo.rglob("*.py")):
        rel = p.relative_to(repo).as_posix()
        if not is_protected(rel, protected) and "/." not in f"/{rel}":
            yield p


def _symptoms(run) -> str:
    failed = sorted(run.failed)[:5]
    tail = run.exec_result.output.strip().splitlines()[-15:] if run.exec_result else []
    lines = "\n".join(f"- `{t}`" for t in failed)
    return (
        "Some behaviour in this repository regressed. These tests now fail:\n\n"
        f"{lines}\n\nTest output (tail):\n```\n" + "\n".join(tail) + "\n```\n\n"
        "Find and fix the bug in the source code. Do not modify the tests."
    )


def generate_mutant_tasks(
    repo: str | Path,
    tests_dir: str,
    pytest_targets: list[str],
    out_dir: str | Path,
    repo_name: str,
    config: SwarmConfig,
    max_tasks: int = 20,
    seed: int = 0,
    split: str = "train",
) -> list[Path]:
    repo = Path(repo)
    rng = random.Random(seed)
    cands: list[Mutation] = []
    for f in candidate_files(repo, config.protected_paths):
        try:
            cands.extend(find_mutations(f.relative_to(repo).as_posix(), f.read_text()))
        except SyntaxError:
            continue
    rng.shuffle(cands)
    written: list[Path] = []
    for m in cands:
        if len(written) >= max_tasks:
            break
        original = (repo / m.path).read_text()
        mutated = apply_mutation(original, m)
        work = Path(tempfile.mkdtemp(prefix="swarmrl-mutant-"))
        try:
            shutil.copytree(repo, work / "repo", ignore=shutil.ignore_patterns(".git", "__pycache__"))
            (work / "repo" / m.path).write_text(mutated)
            sb = LocalSandbox(work / "repo")
            try:
                run = run_pytest(sb, pytest_targets, timeout=300)
            finally:
                sb.close()
            if run.collection_error or not run.failed:
                continue  # equivalent mutant, or one that breaks imports: not a useful task
            tid = f"{repo_name}-mut-{len(written):04d}"
            overlay = work / "overlay"
            shutil.copytree(repo / tests_dir, overlay / tests_dir, ignore=shutil.ignore_patterns("__pycache__"))
            written.append(
                write_task(
                    Path(out_dir) / tid,
                    instruction=_symptoms(run),
                    repo_src=work / "repo",
                    test_overlay_src=overlay,
                    oracle_patch=git_style_patch(m.path, mutated, original),
                    pytest_targets=pytest_targets,
                    split=split,
                    source="procedural-mutant",
                    repo=repo_name,
                    extra_metadata={"task_id": tid, "mutation": m.describe(), "seed": seed},
                )
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return written
