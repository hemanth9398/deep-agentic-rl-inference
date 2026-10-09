import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from swarmrl.inference.chat import ByteTokenizer, ChatFormat
from swarmrl.inference.client import SGLangClient, StickyRouter
from swarmrl.inference.weight_sync import DiskWeightSync, SGLangAdmin
from swarmrl.swarm.config import SamplingConfig


class FakeSGLang(BaseHTTPRequestHandler):
    calls: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        FakeSGLang.calls.append((self.path, body))
        if self.path == "/generate":
            out = [10, 11, 258]  # ends with end-of-turn (ByteTokenizer id 258)
            resp = {
                "output_ids": out,
                "meta_info": {
                    "output_token_logprobs": [[-0.1, i, None] for i in out],
                    "finish_reason": {"type": "stop", "matched": 258},
                },
            }
        else:
            resp = {"success": True}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def serve():
    srv = HTTPServer(("127.0.0.1", 0), FakeSGLang)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_sglang_client_token_in_token_out():
    srv, url = serve()
    try:
        chat = ChatFormat(ByteTokenizer())
        g = asyncio.run(
            SGLangClient(url).generate([1, 2, 3], SamplingConfig(1.0, 1.0, 64), chat.stop_ids, "ep:coder", 64)
        )
        assert g.output_ids == [10, 11, 258] and g.logprobs == [-0.1] * 3 and g.finish_reason == "stop"
        path, body = FakeSGLang.calls[-1]
        assert path == "/generate" and body["input_ids"] == [1, 2, 3] and body["return_logprob"] is True
        sp = body["sampling_params"]
        assert sp["stop_token_ids"] == [258] and sp["no_stop_trim"] is True and sp["skip_special_tokens"] is False
    finally:
        srv.shutdown()


def test_weight_sync_flushes_cache():
    srv, url = serve()
    try:
        FakeSGLang.calls.clear()
        client = SGLangClient(url)
        sync = DiskWeightSync(SGLangAdmin([url]), "adapter", client, colocated=True)
        sync.before_training()
        sync.sync("/ckpt/1", 1)
        sync.sync("/ckpt/2", 2)
        paths = [p for p, _ in FakeSGLang.calls]
        assert paths[0] == "/release_memory_occupation"
        assert paths.count("/flush_cache") == 2 and paths[-1] == "/flush_cache"
        assert "/unload_lora_adapter" in paths and client.lora_name == "policy-v2"
    finally:
        srv.shutdown()


def test_sticky_router_is_stable():
    r = StickyRouter(["a", "b", "c"])
    keys = [f"ep{i}:coder" for i in range(200)]
    first = {k: r.route(k) for k in keys}
    assert all(r.route(k) == v for k, v in first.items())
    assert len(set(first.values())) == 3
    r2 = StickyRouter(["a", "b", "c", "d"])
    moved = sum(r2.route(k) != first[k] for k in keys)
    assert all(r2.route(k) == "d" for k in keys if r2.route(k) != first[k])
    assert moved < 100


def test_chat_format_byte_roundtrip():
    chat = ChatFormat(ByteTokenizer())
    ids = chat.start("sys", "hello")
    assert chat.decode(ids).endswith("<start_of_turn>model\n")
    assert chat.closing_ids([1, 2]) == [chat.end_of_turn_id]
    assert chat.closing_ids([1, chat.end_of_turn_id]) == []
