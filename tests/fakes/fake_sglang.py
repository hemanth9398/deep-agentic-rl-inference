"""A stub SGLang server for end-to-end tests: plays a scripted swarm over real HTTP.

The role is read from the system prompt in the context and the turn from the number of model
turns so far, exactly what a model would see. Admin endpoints (weight sync, flush) succeed and
are recorded.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from swarmrl.inference.chat import ByteTokenizer

ROLE_MARKERS = {
    "You are the Planner": "planner",
    "You are the Coder": "coder",
    "You are the Tester": "tester",
    "You are the Reviewer": "reviewer",
    "You are a software engineer": "agent",
}


class StubSGLang:
    def __init__(self, script):
        self.script = script
        self.tok = ByteTokenizer()
        self.calls: list[tuple[str, dict]] = []
        self.headers: list[dict[str, str]] = []
        stub = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                stub.calls.append((self.path, body))
                stub.headers.append(dict(self.headers))
                resp = stub.generate(body) if self.path == "/generate" else {"success": True}
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def generate(self, body):
        text = self.tok.decode(body["input_ids"])
        role = next(r for m, r in ROLE_MARKERS.items() if m in text)
        turn = text.count("<start_of_turn>model") - 1
        ids = self.tok.encode(self.script(role, turn, text)) + [self.tok.token_id("<end_of_turn>")]
        lps = [-0.05 - ((t * 31 + i * 17) % 97) / 50 for i, t in enumerate(ids)]
        return {
            "output_ids": ids,
            "meta_info": {
                "output_token_logprobs": [[lp, t, None] for lp, t in zip(lps, ids, strict=True)],
                "finish_reason": {"type": "stop"},
            },
        }

    def close(self):
        self.server.shutdown()
