"""로컬 mock 추론 서버 (vLLM OpenAI 호환 / Ollama native).

실제 엔진 동작이 아니라 HTTP 형식만 흉내낸다.

- vLLM mock: `/v1/chat/completions` (JSON / SSE), `/v1/models`, `/version`, `/metrics`.
  `sched_policy="fcfs"` 일 때 0 이 아닌 priority 를 400 으로 거부한다 — 실제 vLLM 의 동작은 버전마다
  확인해야 하며 (probe_engine.py), 여기서는 거부되는 경우의 처리 경로를 테스트하기 위한 가정이다.
- Ollama mock: `/api/chat`, `/api/version`, `/api/tags`, `/api/ps`.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def default_responder(prompt: str) -> str:
    return "Paris is the answer"


class MockServer:
    def __init__(
        self,
        kind: str,
        responder=default_responder,
        api_key: str | None = None,
        sched_policy: str = "fcfs",
        model: str = "mock-model",
    ):
        self.kind = kind
        self.responder = responder
        self.api_key = api_key
        self.sched_policy = sched_policy
        self.model = model
        self.requests: list[dict] = []
        self.n_chat = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, obj=None, text=None, ctype="application/json"):
                body = (text if text is not None else json.dumps(obj)).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self):
                if server.api_key is None:
                    return True
                return self.headers.get("Authorization") == f"Bearer {server.api_key}"

            def do_GET(self):
                server.requests.append(
                    {"method": "GET", "path": self.path, "headers": dict(self.headers)}
                )
                if server.kind == "vllm":
                    if not self._authorized():
                        return self._send(401, {"error": "Unauthorized"})
                    if self.path == "/v1/models":
                        return self._send(
                            200,
                            {
                                "data": [
                                    {
                                        "id": server.model,
                                        "root": server.model,
                                        "max_model_len": 6144,
                                    }
                                ]
                            },
                        )
                    if self.path == "/version":
                        return self._send(200, {"version": "0.0.0-mock"})
                    if self.path == "/metrics":
                        text = (
                            "# HELP vllm:num_requests_running x\n"
                            f'vllm:num_requests_running{{model_name="{server.model}"}} 0.0\n'
                            f'vllm:num_preemptions_total{{model_name="{server.model}"}} 0.0\n'
                            f'vllm:request_queue_time_seconds_sum{{model_name="{server.model}"}} '
                            f"{0.01 * server.n_chat}\n"
                        )
                        return self._send(200, text=text, ctype="text/plain")
                else:
                    if self.path == "/api/version":
                        return self._send(200, {"version": "0.0.0-mock"})
                    if self.path == "/api/tags":
                        return self._send(
                            200,
                            {
                                "models": [
                                    {
                                        "name": server.model,
                                        "model": server.model,
                                        "digest": "abc123",
                                        "details": {
                                            "quantization_level": "Q4_K_M",
                                            "parameter_size": "8.0B",
                                        },
                                    }
                                ]
                            },
                        )
                    if self.path == "/api/ps":
                        return self._send(
                            200, {"models": [{"name": server.model, "size_vram": 5_000_000_000}]}
                        )
                self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "body": body,
                        "headers": dict(self.headers),
                    }
                )
                if server.kind == "vllm" and self.path == "/v1/chat/completions":
                    return self._vllm_chat(body)
                if server.kind == "vllm" and self.path == "/reset_prefix_cache":
                    return self._send(200, {})
                if server.kind == "ollama" and self.path == "/api/chat":
                    server.n_chat += 1
                    prompt = body["messages"][-1]["content"]
                    return self._send(
                        200,
                        {
                            "model": body["model"],
                            "message": {
                                "role": "assistant",
                                "content": " " + server.responder(prompt),
                            },
                            "done": True,
                            "done_reason": "stop",
                            "prompt_eval_count": len(prompt.split()),
                            "eval_count": 4,
                            "total_duration": 2_000_000_000,
                            "load_duration": 100_000_000,
                            "prompt_eval_duration": 500_000_000,
                            "eval_duration": 1_400_000_000,
                        },
                    )
                self._send(404, {"error": "not found"})

            def _vllm_chat(self, body):
                if not self._authorized():
                    return self._send(401, {"error": "Unauthorized"})
                if body.get("priority", 0) != 0 and server.sched_policy != "priority":
                    return self._send(400, {"error": {"message": "priority not enabled (mock)"}})
                server.n_chat += 1
                prompt = body["messages"][-1]["content"]
                text = server.responder(prompt)
                usage = {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(text.split()),
                    "total_tokens": len(prompt.split()) + len(text.split()),
                }
                rid = f"chatcmpl-{server.n_chat}"
                if not body.get("stream"):
                    return self._send(
                        200,
                        {
                            "id": rid,
                            "model": body["model"],
                            "usage": usage,
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": text},
                                }
                            ],
                        },
                    )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def emit(obj):
                    payload = obj if isinstance(obj, str) else json.dumps(obj)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()

                base = {"id": rid, "model": body["model"]}
                emit({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
                for i, word in enumerate(text.split(" ")):
                    emit(
                        {
                            **base,
                            "choices": [
                                {"index": 0, "delta": {"content": (" " if i else "") + word}}
                            ],
                        }
                    )
                emit({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                if (body.get("stream_options") or {}).get("include_usage"):
                    emit({**base, "choices": [], "usage": usage})
                emit("[DONE]")
                self.close_connection = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def chat_requests(self) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST"]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def mock_server():
    """mock_server(kind, **opts) 로 서버를 띄운다. 테스트가 끝나면 종료."""
    servers = []

    def make(kind: str, **opts) -> MockServer:
        s = MockServer(kind, **opts).__enter__()
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.__exit__(None, None, None)
