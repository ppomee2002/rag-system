"""vLLM `/v1/completions` (raw prompt) 엔진.

공식 CRAG_Inference.py 는 vLLM offline `LLM.generate([prompt])` 로 chat template 없이 prompt 원문을
넣는다. 기존 `VLLMEngine` 은 chat API (서버가 chat template 적용) 이므로, 공식 재현 (MODE A) 에서는
이 엔진으로 같은 서버의 completions endpoint 를 쓴다. 기존 VLLMEngine 은 수정하지 않는다.
"""

from __future__ import annotations

import json
import time

from rag.llm.engine import LLMRequest, LLMResponse
from rag.llm.vllm_engine import VLLMEngine


class VLLMCompletionEngine(VLLMEngine):
    name = "vllm_completions"

    def __init__(self, *args, top_p: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_p = top_p

    def build_payload(self, request: LLMRequest) -> dict:
        payload = super().build_payload(request)
        payload.pop("messages")
        payload["prompt"] = request.prompt
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self.build_payload(request)
        url = f"{self.base_url}/completions"
        out = LLMResponse(text="", engine=self.name, priority_sent=payload.get("priority"))
        out.t_start = time.perf_counter()
        if self.stream:
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
                    piece = choice.get("text")
                    if piece:
                        if out.t_first_token is None:
                            out.t_first_token = time.perf_counter()
                        parts.append(piece)
                    if choice.get("finish_reason"):
                        out.finish_reason = choice["finish_reason"]
            out.text = "".join(parts)
        else:
            res, _ = self.transport.post_json(url, payload, self._headers(request))
            self._check_error(res)
            choice = res["choices"][0]
            out.text = choice.get("text") or ""
            out.finish_reason = choice.get("finish_reason")
            self._apply_meta(res, out)
        out.t_end = time.perf_counter()
        return out  # 원본처럼 strip 하지 않는다 (후처리는 postprocess_answer)

    def describe(self) -> dict:
        d = super().describe()
        d["engine"] = self.name
        d["params"]["top_p"] = self.top_p
        d["api"] = "completions (chat template 미적용)"
        return d


def build_generator_engine(llm_cfg: dict):
    """llm.api: chat (기본, 기존 build_engine) | completions (공식처럼 raw prompt)."""
    from rag.llm.factory import build_engine

    if llm_cfg.get("api", "chat") == "chat":
        return build_engine(llm_cfg)
    if llm_cfg.get("provider") != "vllm":
        raise ValueError("llm.api=completions 는 provider=vllm 만 지원")
    return VLLMCompletionEngine(
        model=llm_cfg["model"],
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:8000/v1"),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 100),
        seed=llm_cfg.get("seed", 0),
        stream=llm_cfg.get("stream", True),
        send_priority=llm_cfg.get("send_priority", False),
        api_key_env=llm_cfg.get("api_key_env", "VLLM_API_KEY"),
        timeout=llm_cfg.get("timeout", 600.0),
        extra_body=llm_cfg.get("extra_body"),
        top_p=llm_cfg.get("top_p"),
    )
