"""`InferenceEngine` 을 기존 `LLM` Protocol (`generate(prompt) -> str`) 로 감싸는 adapter.

기존 pipeline (`CRAG`, `VanillaRAG`) 을 수정하지 않고 엔진·호출 기록·스케줄링 gate 를 붙인다.
동시 실행 시 query 마다 `EngineLLM` 을 하나씩 만들고 (자기 `CallRecorder`), 엔진과 gate 는 공유한다.

스케줄링 연결 방법은 두 가지다.
  scheduler + query_state : `rag.scheduling.SchedulingGate`
    (동적 우선순위, K slots, vLLM priority 전달)
  gate + priority_fn      : 기존 `LLMGate` (exp/correction-scheduling) 와 같은 정적 인터페이스
                            `acquire(priority)`, `release()`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from rag.llm.engine import InferenceEngine, LLMRequest
from rag.telemetry.calls import CallRecord, CallRecorder, infer_call_kind


class Gate(Protocol):
    def acquire(self, priority: float) -> None: ...

    def release(self) -> None: ...


# (kind, idx, prompt) -> client 우선순위 (낮을수록 먼저). None 이면 enqueue 시각 (FIFO)
PriorityFn = Callable[[str, int, str], float | None]
# client 우선순위 -> 엔진 priority (int). None 이면 엔진에 보내지 않음
EnginePriorityFn = Callable[[float], int | None]


class EngineLLM:
    def __init__(
        self,
        engine: InferenceEngine,
        recorder: CallRecorder | None = None,
        priority_fn: PriorityFn | None = None,
        engine_priority_fn: EnginePriorityFn | None = None,
        gate: Gate | None = None,
        kind_fn: Callable[[str], str] = infer_call_kind,
        scheduler=None,  # rag.scheduling.SchedulingGate
        query_state=None,  # rag.scheduling.QueryState (scheduler 사용 시 필수, 없어도 stage 기록용)
    ):
        if scheduler is not None and query_state is None:
            raise ValueError("scheduler 를 쓰려면 query_state 가 필요")
        if scheduler is not None and gate is not None:
            raise ValueError("scheduler 와 gate 는 함께 쓰지 않는다")
        self.engine = engine
        self.recorder = recorder
        self.priority_fn = priority_fn
        self.engine_priority_fn = engine_priority_fn
        self.gate = gate
        self.kind_fn = kind_fn
        self.scheduler = scheduler
        self.query_state = query_state

    def generate(self, prompt: str, **kwargs) -> str:
        kind = kwargs.get("kind") or self.kind_fn(prompt)
        idx = self.recorder.next_idx(kind) if self.recorder else 0
        if self.query_state is not None and hasattr(self.query_state, "note_call"):
            self.query_state.note_call(kind)  # 호출 종류로 stage 보정 (그래프 없는 파이프라인)
        t_enqueue = time.perf_counter()
        ticket = None
        priority = None
        engine_priority = None
        if self.scheduler is not None:
            ticket = self.scheduler.acquire(self.query_state, kind, idx, len(prompt))
            priority = ticket.base
            engine_priority = ticket.engine_priority
        else:
            priority = self.priority_fn(kind, idx, prompt) if self.priority_fn else None
            if priority is None:
                priority = t_enqueue
            if self.engine_priority_fn:
                engine_priority = self.engine_priority_fn(priority)
        request = LLMRequest(
            prompt=prompt,
            kind=kind,
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            priority=engine_priority,
            response_format=kwargs.get("response_format"),
        )
        if self.gate is not None:
            self.gate.acquire(priority)
        t_start = time.perf_counter()
        response = None
        error = None
        try:
            response = self.engine.complete(request)
            return response.text
        except Exception as e:
            error = f"{type(e).__name__}: {e}"[:500]
            raise
        finally:
            t_end = time.perf_counter()
            if self.gate is not None:
                self.gate.release()
            if ticket is not None:
                self.scheduler.release(ticket)
            if self.recorder is not None:
                self._record(
                    kind, idx, priority, prompt, t_enqueue, t_start, t_end, response, error, ticket
                )

    def _record(
        self, kind, idx, priority, prompt, t_enqueue, t_start, t_end, response, error, ticket=None
    ):
        r = self.recorder
        server = response.server if response else {}
        t_first = response.t_first_token if response else None
        uses_priority = self.gate is not None or self.priority_fn or ticket is not None
        self.recorder.add(
            CallRecord(
                kind=kind,
                idx=idx,
                priority=priority if uses_priority else None,
                dispatch_key=list(ticket.key_at_grant) if ticket else None,
                output_text=(response.text[:2000] if response and r.record_texts else None),
                engine_priority=response.priority_sent if response else None,
                t_enqueue=r.rel(t_enqueue),
                t_start=r.rel(t_start),
                t_first_token=r.rel(t_first),
                t_end=r.rel(t_end),
                gate_wait=t_start - t_enqueue,
                wall=t_end - t_start,
                ttft=None if t_first is None else t_first - t_start,
                prompt_chars=len(prompt),
                prompt_tokens=response.prompt_tokens if response else None,
                out_tokens=response.completion_tokens if response else None,
                finish_reason=response.finish_reason if response else None,
                engine=self.engine.name,
                model=response.model if response else getattr(self.engine, "model", None),
                request_id=response.request_id if response else None,
                server_total=server.get("total"),
                server_load=server.get("load"),
                server_prompt_eval=server.get("prompt_eval"),
                server_eval=server.get("eval"),
                error=error,
            )
        )
