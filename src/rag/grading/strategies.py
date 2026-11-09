"""Full / Lazy / Batch / Adaptive grading.

Lazy 규칙의 출처: exp/grading-optimization:experiments/grading_optimization/run_live.py,
exp/correction-scheduling:experiments/correction_scheduling/run_lazy.py
  lazy_hit  : 검색 순서로 평가하다 처음 upper 이상이 나오면 중단, 그 문서 1개만 사용
  lazy_rest : 같은 지점에서 중단, 그 문서 + 뒤의 (평가하지 않은) 문서 사용
"""

from __future__ import annotations

import json
import re

from rag.grading.base import (
    GradingResult,
    decide_action,
    grade_many,
    grade_one,
    select_contexts,
)
from rag.pipelines import prompts


class FullGrading:
    """모든 문서를 문서별 호출로 평가 (기존 CRAG). parallel=True 면 k 개 호출을 동시에 전송."""

    def __init__(self, parallel: bool = False):
        self.parallel = parallel
        self.name = "full_parallel" if parallel else "full"

    def grade(self, question, docs, llm, upper, lower) -> GradingResult:
        scores: list[float | None] = list(grade_many(question, docs, llm, self.parallel))
        action = decide_action(scores, upper, lower)
        return GradingResult(
            scores=scores,
            action=action,
            contexts=select_contexts(docs, scores, action, upper, lower),
            n_llm_calls=len(docs),
        )


class LazyGrading:
    def __init__(self, mode: str = "hit"):
        if mode not in ("hit", "rest"):
            raise ValueError(f"lazy mode: hit | rest (got {mode})")
        self.mode = mode
        self.name = f"lazy_{mode}"

    def grade(self, question, docs, llm, upper, lower) -> GradingResult:
        scores: list[float | None] = [None] * len(docs)
        for i, d in enumerate(docs):
            scores[i] = grade_one(question, d, llm)
            if scores[i] >= upper:
                contexts = [d] if self.mode == "hit" else docs[i:]
                return GradingResult(scores, "correct", contexts, i + 1, {"hit_index": i})
        # hit 이 없으면 전부 평가한 것 = Full 과 같은 판정
        action = decide_action(scores, upper, lower)
        return GradingResult(
            scores, action, select_contexts(docs, scores, action, upper, lower), len(docs)
        )


# ---------------------------------------------------------------- batch
_JSON_OBJ = re.compile(r"\{.*?\}", re.S)
_JSON_LIST = re.compile(r"\[[^\[\]]*\]", re.S)
_INDEXED = re.compile(r"(?:document\s*)?\[?(\d+)\]?\s*[:=\-]\s*(\d*\.?\d+)", re.I)


def _valid(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    return float(x) if 0.0 <= x <= 1.0 else None


def parse_batch_scores(text: str, n: int) -> tuple[list[float | None], str]:
    """batch grading 출력 -> (문서별 score 또는 None, 파싱 상태).

    허용 형식 (앞에서부터 시도):
      1. JSON object {"scores": [...]} — 길이가 n 이어야 함
      2. JSON list [...] — 길이가 n 이어야 함
      3. 번호 형식 "[1]: 0.8" / "Document 2 = 0.1" — 번호별로 채움
    길이가 맞지 않는 목록은 위치 대응을 믿을 수 없으므로 전체를 무효로 본다.
    [0, 1] 밖의 값은 clip 하지 않고 무효 (None) 로 둔다 → fallback 대상.
    """
    for m in _JSON_OBJ.finditer(text):
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("scores"), list):
            vals = obj["scores"]
            if len(vals) != n:
                return [None] * n, f"length_mismatch:{len(vals)}"
            out = [_valid(v) for v in vals]
            return out, "ok" if all(v is not None for v in out) else "partial"
    for m in _JSON_LIST.finditer(text):
        try:
            vals = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        if isinstance(vals, list) and len(vals) == n:
            out = [_valid(v) for v in vals]
            return out, "ok_list" if all(v is not None for v in out) else "partial"
    out: list[float | None] = [None] * n
    for idx, val in _INDEXED.findall(text):
        i = int(idx) - 1
        if 0 <= i < n and out[i] is None:
            out[i] = _valid(float(val))
    if any(v is not None for v in out):
        return out, "indexed" if all(v is not None for v in out) else "partial"
    return out, "unparsed"


class BatchGrading:
    """batch_size 개 문서를 한 호출로 평가 (기본: 전부 = 5개).

    fallback (파싱 실패 / 누락 / 범위 밖 값):
      individual : 해당 문서만 기존 GRADE prompt 로 개별 평가 (기본)
      zero       : 0 으로 간주 (호출 추가 없음)
    """

    def __init__(
        self,
        batch_size: int | None = None,
        fallback: str = "individual",
        max_tokens: int | None = None,
    ):
        if fallback not in ("individual", "zero"):
            raise ValueError(f"fallback: individual | zero (got {fallback})")
        self.batch_size = batch_size
        self.fallback = fallback
        self.max_tokens = max_tokens
        self.name = "batch" if batch_size is None else f"batch{batch_size}"

    def grade_batches(self, question, docs, llm) -> tuple[list[float | None], int, dict]:
        size = self.batch_size or max(len(docs), 1)
        scores: list[float | None] = []
        statuses, raw = [], []
        n_calls = 0
        for start in range(0, len(docs), size):
            part = docs[start : start + size]
            prompt = prompts.GRADE_BATCH.format(
                n=len(part), question=question, documents=prompts.format_batch_documents(part)
            )
            kwargs = {"kind": "grade_batch"}
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            out = llm.generate(prompt, **kwargs)
            n_calls += 1
            vals, status = parse_batch_scores(out, len(part))
            scores += vals
            statuses.append(status)
            raw.append(out[:500])
        fallback_idx = [i for i, s in enumerate(scores) if s is None]
        for i in fallback_idx:
            if self.fallback == "individual":
                scores[i] = grade_one(question, docs[i], llm)
                n_calls += 1
            else:
                scores[i] = 0.0
        info = {
            "batch_status": statuses,
            "fallback_indices": fallback_idx,
            "fallback": self.fallback,
            "batch_raw": raw,
        }
        return scores, n_calls, info

    def grade(self, question, docs, llm, upper, lower) -> GradingResult:
        if not docs:
            return GradingResult([], "incorrect", [], 0)
        scores, n_calls, info = self.grade_batches(question, docs, llm)
        action = decide_action(scores, upper, lower)
        return GradingResult(
            scores, action, select_contexts(docs, scores, action, upper, lower), n_calls, info
        )


# ---------------------------------------------------------------- adaptive
class AdaptiveGrading:
    """근거 충분성에 따라 평가량을 조절한다.

    임계값은 GPU 실험 후 dev split 에서 튜닝한다 (기본은 비활성).

    1. skip_below_retrieval: 검색 점수가 이 값 미만인 문서는 평가하지 않는다 (None=비활성)
    2. probe_k: 검색 상위 probe_k 개를 순서대로 개별 평가, upper 이상이 나오면 중단 (correct)
    3. early_incorrect_retrieval: probe 가 모두 lower 미만이고 남은 문서의 검색 점수가 모두 이 값
       미만이면 나머지를 평가하지 않고 판정 (None=비활성)
    4. 남은 문서를 rest_mode 로 평가: batch (기본) | sequential | parallel
    """

    def __init__(
        self,
        probe_k: int = 1,
        rest_mode: str = "batch",
        skip_below_retrieval: float | None = None,
        early_incorrect_retrieval: float | None = None,
        batch_fallback: str = "individual",
    ):
        if rest_mode not in ("batch", "sequential", "parallel"):
            raise ValueError(f"rest_mode: batch | sequential | parallel (got {rest_mode})")
        if probe_k < 0:
            raise ValueError("probe_k >= 0")
        self.probe_k = probe_k
        self.rest_mode = rest_mode
        self.skip_below = skip_below_retrieval
        self.early_incorrect = early_incorrect_retrieval
        self.batch = BatchGrading(fallback=batch_fallback)
        self.name = "adaptive"

    def grade(self, question, docs, llm, upper, lower) -> GradingResult:
        scores: list[float | None] = [None] * len(docs)
        eligible = [
            i
            for i, d in enumerate(docs)
            if self.skip_below is None or d.score is None or d.score >= self.skip_below
        ]
        info: dict = {"skipped_low_retrieval": [i for i in range(len(docs)) if i not in eligible]}
        n_calls = 0

        def finish(reason: str) -> GradingResult:
            action = decide_action(scores, upper, lower)
            info["stop_reason"] = reason
            contexts = select_contexts(docs, scores, action, upper, lower)
            return GradingResult(scores, action, contexts, n_calls, info)

        if not eligible:
            return finish("all_skipped")
        probes, rest = eligible[: self.probe_k], eligible[self.probe_k :]
        for i in probes:
            scores[i] = grade_one(question, docs[i], llm)
            n_calls += 1
            if scores[i] >= upper:
                return finish("probe_hit")
        if (
            self.early_incorrect is not None
            and probes
            and all(scores[i] < lower for i in probes)
            and all(
                docs[i].score is not None and docs[i].score < self.early_incorrect for i in rest
            )
        ):
            return finish("early_incorrect")
        if rest:
            rest_docs = [docs[i] for i in rest]
            if self.rest_mode == "batch":
                vals, calls, binfo = self.batch.grade_batches(question, rest_docs, llm)
                info["batch"] = binfo
            else:
                vals = grade_many(question, rest_docs, llm, self.rest_mode == "parallel")
                calls = len(rest_docs)
            n_calls += calls
            for i, v in zip(rest, vals, strict=True):
                scores[i] = v
        return finish("graded_rest" if rest else "probes_only")


def build_grader(cfg: dict | None):
    """config `grading:` -> 전략. 예) {strategy: batch, fallback: individual}"""
    cfg = dict(cfg or {})
    strategy = cfg.pop("strategy", "full")
    if strategy == "full":
        return FullGrading(parallel=cfg.pop("parallel", False), **cfg)
    if strategy in ("lazy", "lazy_hit"):
        return LazyGrading(mode=cfg.pop("mode", "hit"), **cfg)
    if strategy == "lazy_rest":
        return LazyGrading(mode="rest", **cfg)
    if strategy == "batch":
        return BatchGrading(**cfg)
    if strategy == "adaptive":
        return AdaptiveGrading(**cfg)
    raise ValueError(f"지원하지 않는 grading strategy: {strategy}")
