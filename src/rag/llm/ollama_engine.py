"""Ollama `InferenceEngine`.

기존 실험 (L-CL, L-CS) 과 요청이 같도록 `OllamaLLM` 을 그대로 호출하고,
응답의 서버 측 시간·토큰 수만 추가로 꺼낸다. Ollama 는 요청 단위 priority 를 지원하지 않으므로
priority 는 보내지 않는다 (`priority_sent=None`).
"""

from __future__ import annotations

import time
from typing import Any

from rag.llm.engine import HttpTransport, LLMRequest, LLMResponse, safe_call
from rag.llm.ollama_llm import OllamaLLM

NS = 1e-9


class OllamaEngine:
    name = "ollama"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        num_ctx: int = 8192,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 600.0,
        transport: HttpTransport | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.transport = transport or HttpTransport(timeout)

    def complete(self, request: LLMRequest) -> LLMResponse:
        if request.response_format is not None:
            raise ValueError(
                "OllamaEngine 은 기존 실험 payload 를 유지하므로 response_format 미지원"
            )
        box: dict = {}

        def post(payload: dict) -> dict:
            box["res"], _ = self.transport.post_json(f"{self.host}/api/chat", payload)
            return box["res"]

        llm = OllamaLLM(
            self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            num_ctx=self.num_ctx,
            host=self.host,
            timeout=self.timeout,
            post=post,
        )
        kwargs = {}
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        t_start = time.perf_counter()
        # prompt 잘림 감지 시 RuntimeError (기존 OllamaLLM 동작)
        text = llm.generate(request.prompt, **kwargs)
        t_end = time.perf_counter()
        res = box["res"]
        return LLMResponse(
            text=text,
            engine=self.name,
            model=res.get("model", self.model),
            prompt_tokens=res.get("prompt_eval_count"),
            completion_tokens=res.get("eval_count"),
            finish_reason=res.get("done_reason"),
            t_start=t_start,
            t_end=t_end,
            server={
                "total": res.get("total_duration", 0) * NS,
                "load": res.get("load_duration", 0) * NS,
                "prompt_eval": res.get("prompt_eval_duration", 0) * NS,
                "eval": res.get("eval_duration", 0) * NS,
            },
        )

    def describe(self) -> dict[str, Any]:
        def model_entry(path: str) -> Any:
            models = self.transport.get_json(f"{self.host}{path}").get("models", [])
            return next(
                (m for m in models if self.model in (m.get("name"), m.get("model"))),
                {"error": f"{self.model} not listed in {path}"},
            )

        return {
            "engine": self.name,
            "host": self.host,
            "model": self.model,
            "params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "num_ctx": self.num_ctx,
                "seed": 0,  # OllamaLLM 고정값
            },
            "version": safe_call(self.transport.get_json, f"{self.host}/api/version"),
            # digest / quantization_level (기본 llama3.1:8b 는 4bit) -> 엔진 간 차이 분석에 필요
            "model_info": safe_call(model_entry, "/api/tags"),
            "placement": safe_call(model_entry, "/api/ps"),  # size_vram: GPU 적재 여부
        }
