"""Verify the weight-sync path (doc: "Gemma ... LoRA weight sync"; risk: stale weights in SGLang).

Usage: python scripts/verify/weight_sync.py --server URL --tokenizer TOK \
           --new-checkpoint /ckpt/step1 --base-checkpoint /ckpt/base [--mode merged|adapter]

Pushes a different checkpoint, checks the log-probs of a fixed probe change, pushes the base back,
and checks they return to the original values: proves both the update and the KV-cache flush.
"""

import argparse
import sys

from swarmrl.inference.chat import ChatFormat, HFTokenizer
from swarmrl.inference.client import SGLangClient
from swarmrl.inference.weight_sync import DiskWeightSync, SGLangAdmin
from swarmrl.swarm.config import SamplingConfig


def probe(client, chat, prompt):
    out = client._generate_sync(prompt, SamplingConfig(0.0, 1.0, 16), chat.stop_ids, "probe", 16)
    return out.output_ids, out.logprobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--new-checkpoint", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--mode", choices=["merged", "adapter"], default="merged")
    ap.add_argument("--colocated", action="store_true")
    a = ap.parse_args()
    chat = ChatFormat(HFTokenizer(a.tokenizer))
    client = SGLangClient(a.server)
    sync = DiskWeightSync(SGLangAdmin([a.server]), a.mode, client, colocated=a.colocated)
    prompt = chat.start("Answer with one word.", "What colour is the sky on a clear day?")
    ids0, lp0 = probe(client, chat, prompt)
    sync.before_training()
    sync.sync(a.new_checkpoint, 1)
    ids1, lp1 = probe(client, chat, prompt)
    sync.before_training()
    sync.sync(a.base_checkpoint, 2)
    ids2, lp2 = probe(client, chat, prompt)
    changed = ids1 != ids0 or max(abs(x - y) for x, y in zip(lp0, lp1)) > 1e-3
    restored = ids2 == ids0 and max(abs(x - y) for x, y in zip(lp0, lp2)) < 1e-3
    print(f"update changed outputs: {changed}; base restored exactly: {restored}")
    ok = changed and restored
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
