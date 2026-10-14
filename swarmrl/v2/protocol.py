"""Prompts and output formats for the v2 roles. Roles exchange text, never tokens: each model keeps its own tokenizer.

Planner (large model): one plan per sample, with a probe that tests the plan's hypothesis in the sandbox
without fixing anything (e.g. reproduce the bug). Coder (small model): full new contents of each changed file.
Router (small model): YES or NO as the very first token, then a short reason; P(YES) is its value estimate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from swarmrl.inference.chat import ChatFormat
from swarmrl.inference.client import Generation

PLANNER_SYSTEM = """You are the Planner of a software team. Read the task and the repository listing,
then propose ONE plan. Format, exactly:
<hypothesis>what is wrong and where, in one or two sentences</hypothesis>
<files>comma-separated repository paths the coder must change</files>
<steps>the change to make, precisely, without writing the code</steps>
<probe>ONE shell command (repository root, under 30 seconds) that checks your hypothesis WITHOUT fixing anything</probe>
<expect>fail if the probe should fail before the fix (it reproduces the problem), pass if it should pass</expect>"""

CODER_SYSTEM = """You are the Coder. Implement the plan exactly. For every file you change, output its COMPLETE
new content as
<file path="relative/path.py">
...entire file...
</file>
Output nothing else."""

ROUTER_PLAN_SYSTEM = """You are the Router. Decide whether this plan, implemented by the coder, will solve the task
(every hidden test passes). The probe already ran in the sandbox; weigh its evidence. Answer YES or NO as the very
first word, then give the reason in one or two sentences: what is missing or wrong, or why it will work."""

ROUTER_PATCH_SYSTEM = """You are the Router. Decide whether this patch solves the task (every hidden test passes).
Answer YES or NO as the very first word, then give the reason in one or two sentences."""

_TAG = r"<{0}>\s*(.*?)\s*</{0}>"
_FILE = re.compile(r'<file path="([^"]+)">\n?(.*?)</file>', re.DOTALL)


def _tag(text: str, name: str) -> str | None:
    m = re.search(_TAG.format(name), text, re.DOTALL)
    return m.group(1).strip() if m else None


@dataclass
class ParsedPlan:
    hypothesis: str = ""
    files: list[str] = field(default_factory=list)
    steps: str = ""
    probe: str = ""
    expect: str = "fail"
    valid: bool = False
    error: str = ""


def safe_path(p: str) -> bool:
    return bool(p) and not p.startswith(("/", "~")) and ".." not in p.split("/") and "\x00" not in p


def parse_plan(text: str) -> ParsedPlan:
    fields = {k: _tag(text, k) for k in ("hypothesis", "files", "steps", "probe", "expect")}
    missing = [k for k, v in fields.items() if not v and k != "expect"]
    if missing:
        return ParsedPlan(error=f"missing {', '.join(missing)}")
    files = [f.strip() for f in re.split(r"[,\n]", fields["files"]) if f.strip()]
    bad = [f for f in files if not safe_path(f)]
    if bad:
        return ParsedPlan(error=f"unsafe paths {bad}")
    expect = (fields["expect"] or "fail").lower()
    return ParsedPlan(fields["hypothesis"], files, fields["steps"], fields["probe"],
                      "pass" if expect.startswith("pass") else "fail", True)  # fmt: skip


def parse_patch(text: str) -> tuple[dict[str, str], str]:
    files = {path.strip(): body for path, body in _FILE.findall(text)}
    if not files:
        return {}, "no <file> blocks"
    bad = [p for p in files if not safe_path(p)]
    if bad:
        return {}, f"unsafe paths {bad}"
    return files, ""


@dataclass(frozen=True)
class VerdictTokens:
    yes: int
    no: int

    @classmethod
    def for_chat(cls, chat: ChatFormat) -> VerdictTokens:
        yes, no = chat.tok.encode("YES")[0], chat.tok.encode("NO")[0]
        if yes == no:
            raise ValueError("YES and NO start with the same token for this tokenizer; choose other verdict words")
        return cls(yes, no)


def verdict_probability(gen: Generation, tokens: VerdictTokens) -> tuple[float, bool]:
    """P(YES) at the first output token, from the sampled token's log-prob (two-way: the rest of the vocabulary
    is assumed negligible at this position). The trainer computes the exact two-way softmax for the loss."""
    if not gen.output_ids:
        return 0.0, False
    first, lp = gen.output_ids[0], gen.logprobs[0] if gen.logprobs else 0.0
    if first == tokens.yes:
        return min(max(math.exp(lp), 0.0), 1.0), True
    if first == tokens.no:
        return min(max(1.0 - math.exp(lp), 0.0), 1.0), True
    return 0.0, False  # not a verdict: treated as a rejection, and not used as a training label


def planner_prompt(instruction: str, listing: str, feedback: list[str]) -> str:
    fb = "\n\nEarlier plans and why they were not used:\n" + "\n".join(feedback) if feedback else ""
    return f"Task:\n{instruction}\n\nRepository files:\n{listing}{fb}"


def probe_summary(probe: dict) -> str:
    if not probe:
        return "(probe not run)"
    conf = {True: "confirms", False: "contradicts", None: "is inconclusive for"}[probe.get("confirmed")]
    return (f"probe `{probe.get('command', '')}` exited {probe.get('exit_code')} ({conf} the hypothesis):\n"
            f"{probe.get('output', '')[-1500:]}")  # fmt: skip


def plan_block(p: ParsedPlan) -> str:
    return f"Hypothesis: {p.hypothesis}\nFiles: {', '.join(p.files)}\nSteps: {p.steps}"


def coder_prompt(instruction: str, plan: ParsedPlan, probe: dict, files: dict[str, str | None]) -> str:
    shown = "\n\n".join(f"--- {path}\n{body if body is not None else '(new file)'}" for path, body in files.items())
    return f"Task:\n{instruction}\n\nPlan:\n{plan_block(plan)}\n\n{probe_summary(probe)}\n\nCurrent files:\n{shown}"


def router_plan_prompt(instruction: str, plan: ParsedPlan, probe: dict) -> str:
    return f"Task:\n{instruction}\n\nPlan:\n{plan_block(plan)}\n\n{probe_summary(probe)}"


def router_patch_prompt(instruction: str, plan: ParsedPlan, diff: str, visible: dict) -> str:
    checks = (f"probe now exits {visible.get('probe_exit')}; visible tests exit {visible.get('tests_exit')}:\n"
              f"{visible.get('tests_output', '')[-1500:]}")  # fmt: skip
    return f"Task:\n{instruction}\n\nPlan:\n{plan_block(plan)}\n\nPatch:\n{diff[-6000:]}\n\nChecks: {checks}"
