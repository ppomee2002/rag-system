"""vLLM `InferenceEngine` (OpenAI 호환 `/v1/chat/completions`).

- chat API 를 쓴다: Ollama `/api/chat` 과 같이 서버가 모델의 chat template 을 적용한다.
- `stream=True` 면 SSE 로 받아 TTFT 를 기록하고, `stream_options.include_usage` 로 토큰 수를 받는다.
- priority 는 `send_priority=True` 일 때만 요청 body 에 넣는다.
  vLLM 이 `--scheduling-policy priority` 가 아닐 때 priority 를 어떻게 처리하는지는
  버전마다 확인이 필요하다 (`scripts/server/probe_engine.py` 의 priority 점검).
- API key 는 환경변수 (`api_key_env`) 에서만 읽는다. config / 로그에 남기지 않는다.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from rag.llm.engine import EngineError, HttpTransport, LLMRequest, LLMResponse, safe_call


class VLLMEngine:
    name = "vllm"

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = 0,
        stream: bool = True,
        send_priority: bool = False,
        api_key_env: str | None = "VLLM_API_KEY",
        timeout: float = 600.0,
        extra_body: dict | None = None,
        transport: HttpTransport | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.server_root = self.base_url.removesuffix("/v1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.stream = stream
        self.send_priority = send_priority
        self.api_key_env = api_key_env
        self.extra_body = extra_body or {}
        self.transport = transport or HttpTransport(timeout)

    # ------------------------------------------------------------------ request
    def _headers(self, request: LLMRequest | None = None) -> dict:
        headers = {}
        key = os.getenv(self.api_key_env) if self.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if request is not None and request.request_id:
            headers["X-Request-Id"] = request.request_id
        return headers

    def build_payload(self, request: LLMRequest) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens if request.max_tokens is not None else self.max_tokens,
            "temperature": (
                request.temperature if request.temperature is not None else self.temperature
            ),
            "stream": self.stream,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.stream:
            payload["stream_options"] = {"include_usage": True}
        if self.send_priority and request.priority is not None:
            payload["priority"] = int(request.priority)
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        payload.update(self.extra_body)
        return payload

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self.build_payload(request)
        url = f"{self.base_url}/chat/completions"
        out = LLMResponse(text="", engine=self.name, priority_sent=payload.get("priority"))
        out.t_start = time.perf_counter()
        if self.stream:
            self._complete_stream(url, payload, request, out)
        else:
            res, _ = self.transport.post_json(url, payload, self._headers(request))
            self._check_error(res)
            choice = res["choices"][0]
            out.text = (choice["message"].get("content") or "").strip()
            out.finish_reason = choice.get("finish_reason")
            self._apply_meta(res, out)
        out.t_end = time.perf_counter()
        return out

    def _complete_stream(self, url: str, payload: dict, request: LLMRequest, out: LLMResponse):
        parts: list[str] = []
        for raw in self.transport.post_stream(url, payload, self._headers(request)):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            self._check_error(chunk)
            self._apply_meta(chunk, out)
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if out.t_first_token is None:
                        out.t_first_token = time.perf_counter()
                    parts.append(piece)
                if choice.get("finish_reason"):
                    out.finish_reason = choice["finish_reason"]
        out.text = "".join(parts).strip()

    @staticmethod
    def _check_error(obj: dict):
        if isinstance(obj, dict) and obj.get("error"):
            err = obj["error"]
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise EngineError(f"vLLM error: {msg}", body=json.dumps(obj)[:2000])

    @staticmethod
    def _apply_meta(obj: dict, out: LLMResponse):
        out.request_id = obj.get("id", out.request_id)
        out.model = obj.get("model", out.model)
        usage = obj.get("usage")
        if usage:
            out.prompt_tokens = usage.get("prompt_tokens")
            out.completion_tokens = usage.get("completion_tokens")

    # ------------------------------------------------------------------ server info
    def models(self) -> dict:
        return self.transport.get_json(f"{self.base_url}/models", self._headers())

    def version(self) -> dict:
        return self.transport.get_json(f"{self.server_root}/version", self._headers())

    def metrics_text(self) -> str:
        """Prometheus `/metrics` 원문. 지표 이름은 vLLM 버전마다 다를 수 있다."""
        return self.transport.get_text(f"{self.server_root}/metrics", self._headers())

    def reset_prefix_cache(self) -> dict:
        """조건 사이 prefix cache 초기화 (`POST /reset_prefix_cache`).

        vLLM 0.33.0: `VLLM_SERVER_DEV_MODE=1` 로 띄운 서버에만 있고 (없으면 404),
        `{"success": bool}` 을 돌려준다. 실행 중인 요청이 block 을 잡고 있으면 success=false.
        실패해도 예외 대신 결과를 돌려주고 runner 가 run_end.json 에 기록한다.
        """
        try:
            with self.transport._request(
                f"{self.server_root}/reset_prefix_cache", {}, self._headers(), "POST"
            ) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                body = json.loads(raw) if raw.strip().startswith("{") else {}
                success = body.get("success")
                return {"ok": success is not False, "success": success, "status": resp.status}
        except EngineError as e:
            hint = " (dev mode 아님: launch_vllm.py --dev-mode)" if e.status == 404 else ""
            return {"ok": False, "status": e.status, "error": str(e)[:300] + hint}

    def describe(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "stream": self.stream,
                "send_priority": self.send_priority,
                "extra_body": self.extra_body,
            },
            "api_key_configured": bool(self.api_key_env and os.getenv(self.api_key_env)),
            "version": safe_call(self.version),
            "models": safe_call(self.models),  # max_model_len, root (HF 모델 경로)
        }


def parse_prometheus(text: str, prefix: str = "vllm:") -> dict[str, float]:
    """`name{labels} value` 줄을 {name{labels}: value} 로. 같은 이름은 label 별로 따로 남는다."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        key, _, value = line.rpartition(" ")
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out
