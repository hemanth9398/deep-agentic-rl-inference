"""Verify Gemma chat-template support: our token-level ChatFormat must reproduce the tokenizer's
own apply_chat_template for a multi-turn conversation, token for token.

Usage: python scripts/verify/gemma_template.py --tokenizer google/gemma-... [--system-role]

If it fails, adjust swarmrl.inference.chat.ChatTemplate (e.g. system_in_first_user=False when the
model has a native system role) until it passes. Training on a mismatched template silently
trains a different protocol from the one deployed.
"""

import argparse
import sys

from swarmrl.inference.chat import ChatFormat, ChatTemplate, HFTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--system-role", action="store_true", help="model has a native system turn")
    a = ap.parse_args()
    tok = HFTokenizer(a.tokenizer)
    chat = ChatFormat(tok, ChatTemplate(system_in_first_user=not a.system_role))
    sys_p, u1, m1, u2 = "You are the Coder.", "Fix the bug.", "Looking.\n<tool_call>{}</tool_call>", "obs: ok"
    ours = chat.start(sys_p, u1)
    ours += tok.encode(m1) + [chat.end_of_turn_id]
    ours += chat.user_turn(u2)
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": u1},
            {"role": "assistant", "content": m1}, {"role": "user", "content": u2}]
    ref = tok.hf.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(ref, "input_ids"):
        ref = ref["input_ids"]
    ok = list(ours) == list(ref)
    if not ok:
        i = next((k for k, (x, y) in enumerate(zip(ours, ref)) if x != y), min(len(ours), len(ref)))
        print(f"first difference at token {i}:")
        print("  ours:", repr(tok.decode(ours[max(0, i - 8): i + 8])))
        print("  ref: ", repr(tok.decode(ref[max(0, i - 8): i + 8])))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
