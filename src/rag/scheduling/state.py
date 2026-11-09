"""스케줄링 정책이 보는 query 단위 상태.

LangGraph CRAG 가 `on_retrieved` / `on_graded` 를 호출해 갱신하고 (`rag.graph.crag_graph`),
`SchedulingGate` 가 호출 대기시간을 누적한다. 시각은 실험 t0 기준 초.

출처: exp/correction-scheduling:experiments/correction_scheduling/common.py `QueryState`
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class QueryState:
    query_id: str
    arrival: float = 0.0
    s_pop: float | None = None
    retrieval_scores: list[float] = field(default_factory=list)
    retrieval_chars: list[int] = field(default_factory=list)
    n_docs: int = 0
    action: str | None = None  # grading 이후 확정: correct | ambiguous | incorrect
    n_graded: int | None = None
    cum_wait: float = 0.0  # gate 에서 기다린 시간 누적 (실행·검색 시간 제외)
    n_dispatched: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_retrieved(self, docs) -> None:
        with self.lock:
            if not self.retrieval_scores:  # 최초 retrieval 만 (correction 검색 제외)
                self.retrieval_scores = [d.score if d.score is not None else 0.0 for d in docs]
                self.retrieval_chars = [len(d.text) for d in docs]
                self.n_docs = len(docs)

    def on_graded(self, result) -> None:
        with self.lock:
            self.action = result.action
            self.n_graded = result.n_graded

    @property
    def corrected(self) -> bool | None:
        return None if self.action is None else self.action != "correct"
