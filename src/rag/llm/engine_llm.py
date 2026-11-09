"""`InferenceEngine` 을 기존 `LLM` Protocol (`generate(prompt) -> str`) 로 감싸는 adapter.

기존 pipeline (`CRAG`, `VanillaRAG`) 을 수정하지 않고 엔진·호출 기록·스케줄링 gate 를 붙인다.
동시 실행 시 query 마다 `EngineLLM` 을 하나씩 만들고 (자기 `CallRecorder`), 엔진과 gate 는 공유한다.

gate 는 기존 `LLMGate` (exp/correction-scheduling) 와 같은 인터페이스:
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
    ):
        self.engine = engine
        self.recorder = recorder
        self.priority_fn = priority_fn
        self.engine_priority_fn = engine_priority_fn
        self.gate = gate
        self.kind_fn = kind_fn

    def generate(self, prompt: str, **kwargs) -> str:
        kind = kwargs.get("kind") or self.kind_fn(prompt)
        idx = self.recorder.next_idx(kind) if self.recorder else 0
        t_enqueue = time.perf_counter()
        priority = self.priority_fn(kind, idx, prompt) if self.priority_fn else None
        if priority is None:
            priority = t_enqueue
        engine_priority = self.engine_priority_fn(priority) if self.engine_priority_fn else None
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
            if self.recorder is not None:
                self._record(
                    kind, idx, priority, prompt, t_enqueue, t_start, t_end, response, error
                )

    def _record(self, kind, idx, priority, prompt, t_enqueue, t_start, t_end, response, error):
        r = self.recorder
        server = response.server if response else {}
        t_first = response.t_first_token if response else None
        self.recorder.add(
            CallRecord(
                kind=kind,
                idx=idx,
                priority=priority if self.gate is not None or self.priority_fn else None,
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
