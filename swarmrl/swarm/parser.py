"""Parse exactly one tool call from a model turn. Decoded text is used only here."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from swarmrl.traces.schema import ErrorKind

_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass
class ParsedCall:
    name: str
    arguments: dict[str, Any]

    def key(self) -> str:
        return json.dumps({"name": self.name, "arguments": self.arguments}, sort_keys=True)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ParseError:
    kind: ErrorKind
    message: str


def parse_tool_call(text: str, allowed: tuple[str, ...]) -> ParsedCall | ParseError:
    m = _CALL.search(text)
    if not m:
        return ParseError(ErrorKind.NO_TOOL_CALL, "No tool call found. End your reply with one <tool_call>{...}</tool_call>.")
    body = m.group(1)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return ParseError(ErrorKind.BAD_JSON, f"The tool call is not valid JSON ({e.msg} at char {e.pos}). Resend it.")
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return ParseError(ErrorKind.BAD_JSON, 'The tool call must be an object with "name" and "arguments".')
    name = obj["name"]
    if name not in allowed:
        return ParseError(ErrorKind.UNKNOWN_TOOL, f"Unknown tool {name!r}. Your tools: {', '.join(allowed)}.")
    args = obj.get("arguments", {})
    if not isinstance(args, dict):
        return ParseError(ErrorKind.BAD_ARGS, '"arguments" must be a JSON object.')
    return ParsedCall(name, args)
