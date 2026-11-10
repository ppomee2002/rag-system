"""요청 (query) 단위 실행 상태.

필드는 두 종류로 나뉜다.

정책 입력 (POLICY_FIELDS) — 스케줄링 정책은 `PolicyView` 를 통해 이 값들만 읽을 수 있다.
  request_key, query_id, arrival, s_pop      요청 식별 / 도착 / 인기도
  retrieval_scores, retrieval_chars, n_docs  최초 검색 직후 특징 (cost model 입력)
  stage                                      현재 실행 stage (STAGES)
  corrected                                  correction 경로 여부 (grading 전에는 None)
  n_graded                                   평가한 문서 수 (grading 후)
  cum_wait                                   gate 에서 기다린 시간 누적 (실행·검색 시간 제외)

기록 전용 — 정책은 읽지 않는다. 분석·검증용.
  action, n_dispatched, est_remaining (정책이 마지막으로 계산한 추정치), retrieval_doc_ids,
  stage_history [(stage, t)], est_history [{kind, idx, value, t}]

갱신 주체:
  LangGraph CRAG (`rag.graph.crag_graph`)  enter_stage / on_retrieved / on_graded
  EngineLLM                                 note_call (호출 종류로 stage 보정 — 기존 CRAG 클래스용)
  SchedulingGate                            cum_wait, n_dispatched, record_estimate
  Runner                                    finish

시각은 실험 t0 기준 초 (time.perf_counter).
출처: exp/correction-scheduling:experiments/correction_scheduling/common.py `QueryState` 를 확장.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field

STAGES = (
    "arrived",
    "retrieval",
    "grading",
    "decided",
    "rewrite",
    "web_search",
    "generation",
    "done",
)
KIND_TO_STAGE = {
    "grade": "grading",
    "grade_batch": "grading",
    "rewrite": "rewrite",
    "generate": "generation",
}
POLICY_FIELDS = (
    "request_key",
    "query_id",
    "arrival",
    "s_pop",
    "retrieval_scores",
    "retrieval_chars",
    "n_docs",
    "stage",
    "corrected",
    "n_graded",
    "cum_wait",
)
_KEYS = itertools.count()


@dataclass(eq=False)
class QueryState:
    query_id: str
    arrival: float = 0.0
    s_pop: float | None = None
    t0: float | None = None  # 상대 시각 기준. None 이면 생성 시각
    request_key: str = ""
    # ---- 정책 입력
    retrieval_scores: list[float] = field(default_factory=list)
    retrieval_chars: list[int] = field(default_factory=list)
    n_docs: int = 0
    stage: str = "arrived"
    corrected: bool | None = None
    n_graded: int | None = None
    cum_wait: float = 0.0
    # ---- 기록 전용
    action: str | None = None
    retrieval_doc_ids: list[str] = field(default_factory=list)
    n_dispatched: int = 0
    est_remaining: float | None = None
    stage_history: list = field(default_factory=list)
    est_history: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        if self.t0 is None:
            self.t0 = time.perf_counter()
        if not self.request_key:
            self.request_key = f"{self.query_id}#{next(_KEYS)}"
        self.stage_history.append((self.stage, self._now()))

    def _now(self) -> float:
        return time.perf_counter() - self.t0

    # ------------------------------------------------------------ updates
    def enter_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        with self.lock:
            if stage != self.stage:
                self.stage = stage
                self.stage_history.append((stage, self._now()))

    def note_call(self, kind: str) -> None:
        """LLM 호출 종류로 stage 를 맞춘다 (그래프가 stage 를 관리하지 않는 파이프라인용)."""
        stage = KIND_TO_STAGE.get(kind)
        if stage is not None:
            self.enter_stage(stage)

    def on_retrieved(self, docs) -> None:
        with self.lock:
            if not self.retrieval_scores:  # 최초 retrieval 만 (correction 검색 제외)
                self.retrieval_scores = [d.score if d.score is not None else 0.0 for d in docs]
                self.retrieval_chars = [len(d.text) for d in docs]
                self.n_docs = len(docs)
                self.retrieval_doc_ids = [d.id for d in docs]

    def on_graded(self, result) -> None:
        with self.lock:
            self.action = result.action
            self.n_graded = result.n_graded
            self.corrected = result.action != "correct"
        self.enter_stage("decided")

    def record_estimate(self, kind: str, idx: int, value: float) -> None:
        with self.lock:
            self.est_remaining = value
            self.est_history.append({"kind": kind, "idx": idx, "value": value, "t": self._now()})

    def finish(self, action: str | None = None) -> None:
        """요청 종료. 그래프를 쓰지 않는 파이프라인은 결과의 action 으로 correction 여부를 기록."""
        if action is not None and self.action is None:
            with self.lock:
                self.action = action
                self.corrected = action != "correct"
        self.enter_stage("done")

    # ------------------------------------------------------------ views
    def view(self) -> PolicyView:
        return PolicyView(self)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "request_key": self.request_key,
                "stage": self.stage,
                "corrected": self.corrected,
                "action": self.action,
                "n_graded": self.n_graded,
                "retrieval_doc_ids": list(self.retrieval_doc_ids),
                "cum_wait": self.cum_wait,
                "n_dispatched": self.n_dispatched,
                "est_remaining": self.est_remaining,
                "stage_history": list(self.stage_history),
                "est_history": list(self.est_history),
            }


class PolicyView:
    """정책에 넘기는 읽기 전용 view. POLICY_FIELDS 이외의 속성은 AttributeError."""

    __slots__ = ("_state",)

    def __init__(self, state: QueryState):
        object.__setattr__(self, "_state", state)

    def __setattr__(self, name, value):
        raise AttributeError("PolicyView is read-only")

    def __repr__(self):
        return f"PolicyView({self._state.request_key}, stage={self._state.stage})"


for _name in POLICY_FIELDS:
    setattr(PolicyView, _name, property(lambda self, _n=_name: getattr(self._state, _n)))
