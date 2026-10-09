"""Token-level chat formatting.

Each role's context is a growing list of token ids. Model outputs are appended verbatim
(the ids SGLang sampled); only new user/observation turns are tokenized. Nothing is ever
decoded and re-encoded, so the trainer sees exactly the tokens the sampler produced.

`scripts/verify/gemma_template.py` checks that these pieces reproduce the tokenizer's own
`apply_chat_template` for a multi-turn conversation; run it for every new model/tokenizer.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Token ids for `text`, special-token strings mapped to their ids, no BOS/EOS added."""

    @abc.abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Text for `ids`, special tokens kept (used only to parse tool calls)."""

    @abc.abstractmethod
    def token_id(self, token: str) -> int:
        """Id of a single special token; KeyError if it is not one token."""


@dataclass(frozen=True)
class ChatTemplate:
    """String pieces of a turn-based template (defaults: Gemma-style turns)."""

    bos: str = "<bos>"
    user_open: str = "<start_of_turn>user\n"
    model_open: str = "<start_of_turn>model\n"
    turn_close: str = "<end_of_turn>"
    between_turns: str = "\n"
    system_in_first_user: bool = True  # Gemma 3: no system role; Gemma 4: verify
    system_open: str = "<start_of_turn>system\n"


class ChatFormat:
    def __init__(self, tokenizer: Tokenizer, template: ChatTemplate = ChatTemplate()):
        self.tok = tokenizer
        self.t = template
        self.end_of_turn_id = tokenizer.token_id(template.turn_close)

    @property
    def stop_ids(self) -> list[int]:
        return [self.end_of_turn_id]

    def _enc(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def start(self, system: str, user: str) -> list[int]:
        t = self.t
        if t.system_in_first_user:
            text = f"{t.bos}{t.user_open}{system}\n\n{user}{t.turn_close}{t.between_turns}{t.model_open}"
        else:
            text = (
                f"{t.bos}{t.system_open}{system}{t.turn_close}{t.between_turns}"
                f"{t.user_open}{user}{t.turn_close}{t.between_turns}{t.model_open}"
            )
        return self._enc(text)

    def closing_ids(self, output_ids: list[int]) -> list[int]:
        """Ids needed to close the model turn if the server did not return the end-of-turn token."""
        return [] if output_ids and output_ids[-1] == self.end_of_turn_id else [self.end_of_turn_id]

    def user_turn(self, text: str) -> list[int]:
        t = self.t
        return self._enc(f"{t.between_turns}{t.user_open}{text}{t.turn_close}{t.between_turns}{t.model_open}")

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)


class ByteTokenizer(Tokenizer):
    """Deterministic byte-level tokenizer with special tokens. For tests and dry runs only."""

    SPECIALS = ("<bos>", "<start_of_turn>", "<end_of_turn>", "<eos>")

    def __init__(self) -> None:
        self._special = {s: 256 + i for i, s in enumerate(self.SPECIALS)}
        self._inv = {v: k for k, v in self._special.items()}
        self._split = re.compile("(" + "|".join(re.escape(s) for s in self.SPECIALS) + ")")

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for part in self._split.split(text):
            if part in self._special:
                out.append(self._special[part])
            elif part:
                out.extend(part.encode())
        return out

    def decode(self, ids: list[int]) -> str:
        buf, out = bytearray(), []
        for i in ids:
            if i in self._inv:
                out.append(buf.decode(errors="replace"))
                buf = bytearray()
                out.append(self._inv[i])
            else:
                buf.append(i)
        out.append(buf.decode(errors="replace"))
        return "".join(out)

    def token_id(self, token: str) -> int:
        return self._special[token]


class HFTokenizer(Tokenizer):
    """Adapter over a Hugging Face tokenizer (lazy import)."""

    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer

        self.hf = AutoTokenizer.from_pretrained(name_or_path)

    def encode(self, text: str) -> list[int]:
        return self.hf.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.hf.decode(ids, skip_special_tokens=False)

    def token_id(self, token: str) -> int:
        i = self.hf.convert_tokens_to_ids(token)
        if i is None or i == self.hf.unk_token_id:
            raise KeyError(f"{token!r} is not a single token in this tokenizer")
        return i
