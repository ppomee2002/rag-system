"""LLM 호출 단위 기록.

필드 이름은 기존 correction_scheduling trace
(exp/correction-scheduling `common.py` 의 `QueryLLM`) 와 같게 유지한다.
그래서 기존 simulator 의 `job_from_row` 가 Ollama trace 를 그대로 읽을 수 있다.
vLLM 은 요청 단위 서버 시간이 없어 `server_*` 가 None 이다.

Source (call_kind): exp/correction-scheduling:experiments/correction_scheduling/common.py
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass

from rag.pipelines import prompts

_KIND_PREFIXES = {
    "grade": prompts.GRADE.partition("{")[0],
    "rewrite": prompts.REWRITE.partition("{")[0],
    "generate": prompts.GENERATE.partition("{")[0],
    "grade_batch": prompts.GRADE_BATCH.partition("{")[0],
}


def infer_call_kind(prompt: str) -> str:
    """prompt template 으로 호출 종류 (= C-RAG stage) 를 구분한다."""
    for kind, prefix in _KIND_PREFIXES.items():
        if prompt.startswith(prefix):
            return kind
    return "other"


@dataclass
class CallRecord:
    kind: str
    idx: int  # query 안에서 같은 kind 의 몇 번째 호출인지
    priority: float | None  # client gate 우선순위 (낮을수록 먼저)
    engine_priority: int | None  # 엔진에 실제로 보낸 priority
    t_enqueue: float  # 이하 시각은 실험 t0 기준 초
    t_start: float  # gate 통과 = 엔진 전송 시각
    t_first_token: float | None
    t_end: float
    gate_wait: float
    wall: float  # 전송 ~ 응답 완료 (엔진 내부 대기 포함)
    ttft: float | None
    prompt_chars: int
    prompt_tokens: int | None
    out_tokens: int | None
    finish_reason: str | None
    engine: str
    model: str | None
    request_id: str | None
    server_total: float | None = None  # Ollama total_duration (부하 없을 때 서비스 시간)
    server_load: float | None = None
    server_prompt_eval: float | None = None
    server_eval: float | None = None
    error: str | None = None
    dispatch_key: list | None = None  # SchedulingGate 가 dispatch 할 때의 정책 key


class CallRecorder:
    """query 하나의 호출 기록 (thread-safe). t0 는 동시 실행 query 끼리 공유한다."""

    def __init__(self, t0: float | None = None):
        self.t0 = time.perf_counter() if t0 is None else t0
        self._lock = threading.Lock()
        self._issued: dict[str, int] = {}
        self.calls: list[CallRecord] = []

    def next_idx(self, kind: str) -> int:
        with self._lock:
            idx = self._issued.get(kind, 0)
            self._issued[kind] = idx + 1
            return idx

    def add(self, record: CallRecord) -> None:
        with self._lock:
            self.calls.append(record)

    def rel(self, t: float | None) -> float | None:
        return None if t is None else t - self.t0

    def to_dicts(self) -> list[dict]:
        with self._lock:
            return [asdict(c) for c in sorted(self.calls, key=lambda c: c.t_start)]

    def summary(self) -> dict:
        """query 행에 넣을 합계 (호출 수, 토큰, 시간)."""
        calls = self.to_dicts()
        by_kind: dict[str, int] = {}
        for c in calls:
            by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1

        def total(key):
            vals = [c[key] for c in calls if c[key] is not None]
            return sum(vals) if vals else None

        return {
            "n_calls": len(calls),
            "n_calls_by_kind": by_kind,
            "prompt_tokens": total("prompt_tokens"),
            "out_tokens": total("out_tokens"),
            "llm_wall": total("wall"),
            "gate_wait": total("gate_wait"),
            "server_total": total("server_total"),
            "n_errors": sum(c["error"] is not None for c in calls),
        }
