"""문서 단위 Grader 공통 인터페이스 (공식 T5 Evaluator / 기존 Llama Grader).

두 grader 모두 "질문 + 문서 하나" 에 점수 하나를 낸다. 점수의 척도·의미는 다르다.

  grader_type    score_scale            기본 threshold (upper / lower)  출처
  official_t5    t5_regression_logit    0.592 / -0.995 (PopQA)          공식 run_crag_inference.sh
  llama_prompt   llm_0_1_parsed         0.7 / 0.3                       기존 crag_system config

문서별 flag 와 질문 판정 규칙은 두 grader 에 같은 함수를 쓴다 (rag.crag_official.evaluator):
  flag 2 (pass) : score >= upper / flag 1 (partial) : score >= lower / flag 0 (fail)
  correct : flag 2 가 하나라도 / ambiguous : 아니면 flag 1 이 하나라도 / incorrect : 그 외
기존 CRAG 판정 (any >= upper -> correct, all < lower -> incorrect, 그 외 ambiguous) 과 논리적으로
같다 (tests/test_official_crag.py 에서 확인). 다른 것은 점수 모델·척도와 threshold 이다.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from rag.crag_official import source
from rag.crag_official.evaluator import (
    OfficialRetrievalEvaluator,
    PairScore,
    official_action,
    official_doc_flag,
)
from rag.grading.base import parse_score
from rag.pipelines import prompts
from rag.types import Document

FLAG_LABEL = {2: "pass", 1: "partial", 0: "fail"}


@dataclass
class DocJudgment:
    rank: int  # 검색 순서 (0부터)
    doc_id: str
    score: float
    flag: int
    label: str
    latency: float  # 이 문서 평가의 wall 시간 (초)
    input_tokens: int | None = None
    output_tokens: int | None = None
    source: str = "model"  # model | empty_rule | cache | mock | llm
    compute_seconds: float | None = None  # backend 보고 (cache 면 기록 당시 값)
    raw_output: str | None = None  # llama 출력 원문 (record_texts 일 때)

    def to_dict(self) -> dict:
        return asdict(self)


def decide(judgments: list[DocJudgment]) -> str:
    return official_action(j.flag for j in judgments)


class DocumentGrader(Protocol):
    grader_type: str
    score_scale: str
    upper: float
    lower: float

    def judge(self, question: str, doc: Document, rank: int, llm=None) -> DocJudgment: ...

    def score_strips(self, question: str, strips: list[str], llm=None) -> list[PairScore]: ...

    def describe(self) -> dict[str, Any]: ...


class OfficialT5Grader:
    grader_type = "official_t5"
    score_scale = "t5_regression_logit"

    def __init__(self, evaluator: OfficialRetrievalEvaluator):
        self.evaluator = evaluator
        self.upper = evaluator.upper
        self.lower = evaluator.lower

    def judge(self, question, doc, rank, llm=None) -> DocJudgment:
        t0 = time.perf_counter()
        p = self.evaluator.score_documents(question, [doc.text])[0]
        flag = official_doc_flag(p.score, self.upper, self.lower)
        return DocJudgment(
            rank=rank,
            doc_id=doc.id,
            score=p.score,
            flag=flag,
            label=FLAG_LABEL[flag],
            latency=time.perf_counter() - t0,
            input_tokens=p.n_tokens,
            output_tokens=None,  # 회귀 출력 (생성 토큰 없음)
            source=p.source,
            compute_seconds=p.compute_seconds,
        )

    def score_strips(self, question, strips, llm=None) -> list[PairScore]:
        return self.evaluator.score_strips(question, strips)

    def describe(self) -> dict:
        return {"grader_type": self.grader_type, **self.evaluator.describe()}


class LlamaPromptGrader:
    """기존 grader: rag.pipelines.prompts.GRADE + rag.grading.base.parse_score (0~1 clip)."""

    grader_type = "llama_prompt"
    score_scale = "llm_0_1_parsed"

    def __init__(self, upper: float = 0.7, lower: float = 0.3, record_texts: bool = False):
        self.upper = upper
        self.lower = lower
        self.record_texts = record_texts

    def _call(self, question: str, text: str, llm) -> tuple[float, int | None, int | None, str]:
        if llm is None:
            raise ValueError("llama_prompt grader 에는 llm 이 필요")
        out = llm.generate(prompts.GRADE.format(question=question, document=text), kind="grade")
        rec = getattr(llm, "recorder", None)
        last = rec.calls[-1] if rec is not None and rec.calls else None
        if last is not None and last.kind != "grade":
            last = None
        return (
            parse_score(out),
            last.prompt_tokens if last else None,
            last.out_tokens if last else None,
            out,
        )

    def judge(self, question, doc, rank, llm=None) -> DocJudgment:
        t0 = time.perf_counter()
        score, tin, tout, raw = self._call(question, doc.text, llm)
        flag = official_doc_flag(score, self.upper, self.lower)
        return DocJudgment(
            rank=rank,
            doc_id=doc.id,
            score=score,
            flag=flag,
            label=FLAG_LABEL[flag],
            latency=time.perf_counter() - t0,
            input_tokens=tin,
            output_tokens=tout,
            source="llm",
            raw_output=raw[:500] if self.record_texts else None,
        )

    def score_strips(self, question, strips, llm=None) -> list[PairScore]:
        """official 정책을 Llama 점수로 돌릴 때 (비공식 조합). 공식 짧은 strip 규칙만 따른다."""
        out = []
        for s in strips:
            if len(s.split()) < source.SHORT_STRIP_MIN_WORDS:
                out.append(PairScore(source.SHORT_STRIP_SCORE, None, 0.0, "short_strip_rule"))
                continue
            t0 = time.perf_counter()
            score, tin, _, _ = self._call(question, s, llm)
            out.append(PairScore(score, tin, time.perf_counter() - t0, "llm"))
        return out

    def describe(self) -> dict:
        return {
            "grader_type": self.grader_type,
            "score_scale": self.score_scale,
            "upper": self.upper,
            "lower": self.lower,
            "prompt": "rag.pipelines.prompts.GRADE",
            "parser": "rag.grading.base.parse_score",
        }


class MemoStripScorer:
    """같은 질문에서 이미 평가한 문서 텍스트는 다시 부르지 않는 strip scorer (Llama 비용 절약).

    selection 모드에서 strip == 문서 전체이므로 grading 에서 얻은 점수를 재사용한다.
    공식 T5 에는 쓰지 않는다 (공식은 판정용 점수 (truncation 없음) 와 refinement 점수
    (truncation=True) 를 따로 계산한다).
    """

    def __init__(self, grader: LlamaPromptGrader, memo: dict[str, float], llm):
        self.grader, self.memo, self.llm = grader, memo, llm

    def score_strips(self, question, strips):
        out: list[PairScore | None] = [None] * len(strips)
        todo = []
        for i, s in enumerate(strips):
            key = s.strip()
            if key in self.memo and len(s.split()) >= source.SHORT_STRIP_MIN_WORDS:
                out[i] = PairScore(self.memo[key], None, 0.0, "memo")
            else:
                todo.append(i)
        got = self.grader.score_strips(question, [strips[i] for i in todo], self.llm)
        for i, p in zip(todo, got, strict=True):
            out[i] = p
        return out


def build_grader(cfg: dict, evaluator_factory=None) -> DocumentGrader:
    """cfg (`grader:` 블록): {type: official_t5 | llama_prompt, upper, lower, ...}"""
    gtype = cfg.get("type", "official_t5")
    if gtype == "official_t5":
        if evaluator_factory is None:
            raise ValueError("official_t5 grader 에는 evaluator_factory 가 필요")
        return OfficialT5Grader(evaluator_factory(cfg))
    if gtype == "llama_prompt":
        return LlamaPromptGrader(
            upper=cfg.get("upper", 0.7),
            lower=cfg.get("lower", 0.3),
            record_texts=bool(cfg.get("record_texts", False)),
        )
    raise ValueError(f"grader.type: official_t5 | llama_prompt (got {gtype})")
