"""Grading 전략 공통 인터페이스.

모든 전략은 `grade(question, docs, llm, upper, lower) -> GradingResult` 하나만 구현한다.
LangGraph CRAG (`rag.graph.crag_graph`) 의 grading 노드가 이 인터페이스로 전략을 호출하므로,
그래프 구조와 stage 이름은 전략과 무관하게 같다.

판정 규칙은 기존 `rag.pipelines.crag.CRAG` 와 같다 (평가한 문서만 대상으로):
  correct   : score >= upper 인 문서가 하나라도 있음
  incorrect : 평가한 문서가 모두 score < lower
  ambiguous : 그 외
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag.llm.base import LLM
from rag.pipelines import prompts
from rag.types import Document

ACTIONS = ("correct", "ambiguous", "incorrect")


@dataclass
class GradingResult:
    scores: list[float | None]  # 검색 순서대로. None = 평가하지 않음
    action: str
    contexts: list[Document]  # correction 전 남길 문서 (incorrect 면 [])
    n_llm_calls: int
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def graded_scores(self) -> list[float]:
        return [s for s in self.scores if s is not None]

    @property
    def n_graded(self) -> int:
        return len(self.graded_scores)


class GradingStrategy(Protocol):
    name: str

    def grade(
        self, question: str, docs: list[Document], llm: LLM, upper: float, lower: float
    ) -> GradingResult: ...


def parse_score(text: str) -> float:
    """기존 CRAG.grade 와 같은 파서: 첫 숫자를 [0, 1] 로 clip, 없으면 0."""
    m = re.search(r"\d*\.?\d+", text)
    return min(max(float(m.group()), 0.0), 1.0) if m else 0.0


def grade_one(question: str, doc: Document, llm: LLM) -> float:
    return parse_score(llm.generate(prompts.GRADE.format(question=question, document=doc.text)))


def grade_many(question: str, docs: list[Document], llm: LLM, parallel: bool) -> list[float]:
    """문서별 개별 호출. parallel=True 면 동시에 보낸다 (기존 CRAG parallel_grading 과 같음)."""
    if parallel and len(docs) > 1:
        with ThreadPoolExecutor(max_workers=len(docs)) as pool:
            return list(pool.map(lambda d: grade_one(question, d, llm), docs))
    return [grade_one(question, d, llm) for d in docs]


def decide_action(scores: list[float | None], upper: float, lower: float) -> str:
    graded = [s for s in scores if s is not None]
    if any(s >= upper for s in graded):
        return "correct"
    if all(s < lower for s in graded):
        return "incorrect"
    return "ambiguous"


def select_contexts(
    docs: list[Document], scores: list[float | None], action: str, upper: float, lower: float
) -> list[Document]:
    """평가한 문서 중 correct 면 >= upper, ambiguous 면 >= lower (기존 CRAG 와 같음)."""
    if action == "incorrect":
        return []
    cut = upper if action == "correct" else lower
    return [d for d, s in zip(docs, scores, strict=True) if s is not None and s >= cut]
