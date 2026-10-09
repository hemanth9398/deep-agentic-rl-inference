"""Verify item 1: SGLang token-in/token-out with per-token log-probs, and trainer-equivalent re-scoring.

Usage: python scripts/verify/sglang_tito.py --server http://localhost:30000 --tokenizer google/gemma-...

Checks
  1. /generate with input_ids returns output ids aligned 1:1 with output log-probs.
  2. The end-of-turn token is kept in output_ids (no_stop_trim) so it can carry loss.
  3. Re-scoring prompt+output as input (logprob_start_len) reproduces the sampled log-probs:
     the same gap the trainer's stage-0 check measures, isolated to the server.
"""

import argparse
import json
import sys
import urllib.request

from swarmrl.inference.chat import ChatFormat, HFTokenizer
from swarmrl.inference.client import SGLangClient
from swarmrl.swarm.config import SamplingConfig


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tol", type=float, default=0.02, help="max mean |delta logprob|")
    a = ap.parse_args()
    chat = ChatFormat(HFTokenizer(a.tokenizer))
    prompt = chat.start("You are terse.", "Name three prime numbers, then stop.")
    client = SGLangClient(a.server)
    body = client._payload(prompt, SamplingConfig(1.0, 1.0, 64), chat.stop_ids, 64)
    resp = post(a.server.rstrip("/") + "/generate", body)
    g = SGLangClient.parse_response(resp)
    ok = True
    print(f"[1] {len(g.output_ids)} ids, {len(g.logprobs)} logprobs, finish={g.finish_reason}")
    ok &= len(g.output_ids) == len(g.logprobs) > 0
    if g.finish_reason == "stop":
        kept = g.output_ids[-1] == chat.end_of_turn_id
        print(f"[2] end-of-turn kept in output_ids: {kept}")
        ok &= kept
    rescore = post(
        a.server.rstrip("/") + "/generate",
        {
            "input_ids": prompt + g.output_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 1.0},
            "return_logprob": True,
            "logprob_start_len": len(prompt),
        },
    )
    inp = rescore["meta_info"]["input_token_logprobs"]
    lp = [x[0] for x in inp if x[0] is not None][-len(g.output_ids) :]
    gap = sum(abs(x - y) for x, y in zip(lp, g.logprobs, strict=False)) / max(len(lp), 1)
    print(f"[3] mean |rescored - sampled| logprob = {gap:.4f} (tol {a.tol})")
    ok &= gap <= a.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
