"""추론 엔진 추상화 계층.

기존 `LLM.generate(prompt) -> str` 는 그대로 두고, 측정·스케줄링에 필요한 정보를 주고받는
하위 계층을 둔다.

    engine.complete(LLMRequest) -> LLMResponse   # 토큰 수, TTFT, 서버 측 시간, priority 등
    EngineLLM(engine).generate(prompt) -> str     # 기존 pipeline 이 사용 (rag.llm.engine_llm)

시각은 모두 `time.perf_counter()` 값이다 (`rag.utils.timing.stopwatch` 와 같은 시계). 호출 기록에서
실험 t0 기준 상대 시각으로 바꾼다 (`rag.telemetry.calls`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMRequest:
    prompt: str
    kind: str = "other"  # grade | rewrite | generate | grade_batch | other
    max_tokens: int | None = None  # None 이면 엔진 기본값
    temperature: float | None = None
    priority: int | None = (
        None  # 엔진에 전달할 priority (vLLM: 낮을수록 먼저). None 이면 보내지 않음
    )
    response_format: dict | None = None
    request_id: str | None = None


@dataclass
class LLMResponse:
    text: str
    engine: str
    model: str | None = None
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    t_start: float = 0.0
    t_first_token: float | None = None  # streaming 일 때만
    t_end: float = 0.0
    priority_sent: int | None = None  # 실제로 요청에 넣은 priority
    server: dict[str, float | None] = field(default_factory=dict)  # 엔진이 보고한 시간 (초)

    @property
    def wall(self) -> float:
        return self.t_end - self.t_start

    @property
    def ttft(self) -> float | None:
        return None if self.t_first_token is None else self.t_first_token - self.t_start


class InferenceEngine(Protocol):
    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def describe(self) -> dict[str, Any]:
        """실행 메타데이터용 엔진 정보 (버전, 모델, 서버 설정). 실패해도 예외 대신 error 필드."""
        ...


class EngineError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


class HttpTransport:
    """표준 라이브러리 HTTP. 재시도하지 않는다 (latency 측정을 왜곡하므로 호출자가 결정)."""

    def __init__(self, timeout: float = 600.0):
        self.timeout = timeout

    def _request(self, url: str, payload: dict | None, headers: dict | None, method: str):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:2000]
            raise EngineError(f"HTTP {e.code} {url}: {body}", status=e.code, body=body) from None
        except urllib.error.URLError as e:
            raise EngineError(f"connection failed {url}: {e.reason}") from None

    def post_json(self, url: str, payload: dict, headers: dict | None = None) -> tuple[dict, dict]:
        with self._request(url, payload, headers, "POST") as resp:
            return json.load(resp), dict(resp.headers)

    def post_stream(self, url: str, payload: dict, headers: dict | None = None) -> Iterator[bytes]:
        """응답 본문을 줄 단위로 yield (SSE / NDJSON)."""
        with self._request(url, payload, headers, "POST") as resp:
            yield from resp

    def get_json(self, url: str, headers: dict | None = None) -> dict:
        with self._request(url, None, headers, "GET") as resp:
            return json.load(resp)

    def get_text(self, url: str, headers: dict | None = None) -> str:
        with self._request(url, None, headers, "GET") as resp:
            return resp.read().decode("utf-8", errors="replace")


def safe_call(fn, *args, **kwargs) -> Any:
    """describe() 용: 실패하면 {"error": ...} 반환."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - 메타데이터 수집은 실험을 멈추지 않는다
        return {"error": f"{type(e).__name__}: {e}"}
