"""C3: Evidence Budget Control 인터페이스.

두 제어 변수를 분리한다.
  stop_rule  : grading 을 언제 멈출지 (평가 중단 시점)
                 full          모든 문서 평가
                 first_pass    처음 flag 2 (score >= upper) 가 나오면 중단  (기존 Lazy)
                 max_docs:N    앞 N 개만 평가
  retention  : 멈춘 뒤 내부 문서 중 무엇을 generation 에 남길지 (문서 보존 비율)
                 policy        evidence policy 규칙을 따름 (기본)
                 hit_only      first pass 문서 1개 (기존 Lazy-hit)
                 hit_and_rest  first pass 문서 + 그 뒤 (평가하지 않은) 문서 (기존 Lazy-rest)
                 all           검색 문서 전부
                 ratio         ratio, base (rest | all): 후보의 앞 ceil(ratio x 후보 수) 개
                               ** 정의 미확정 (placeholder). "Lazy-rest 25%" 의 정확한 정의는
                               C3 담당자가 확정 후 Retention 을 추가/수정한다 **
  hit 이 없으면 (first_pass 인데 끝까지 통과 문서 없음) hit_only / hit_and_rest / ratio(rest) 는
  None 을 돌려 policy 규칙으로 돌아간다 (기존 LazyGrading 과 같음).

preset (기존 이름과 대응):
  full_grading = full + policy / lazy_hit = first_pass + hit_only /
  lazy_rest = first_pass + hit_and_rest / keep_all = full + all

"첫 통과 문서" 와 공식 Correct 의 관계:
  first_pass 의 통과 조건 (flag 2) 은 공식 Correct 의 충분조건과 같다 (하나라도 있으면 Correct).
  따라서 first_pass 로 멈춰도 **판정 (action)** 은 공식과 같다.
  그러나 공식 Correct 의 **evidence** 는 모든 문서의 refinement 점수 상위 3개라서 lazy 로 얻을 수
  없다 -> official 정책과 first_pass / max_docs / retention != policy 의 조합은 거부한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from rag.evidence.state import GradingSnapshot
from rag.types import Document


class StopRule(Protocol):
    name: str

    def should_stop(self, snap: GradingSnapshot) -> bool: ...


class FullStop:
    name = "full"

    def should_stop(self, snap):
        return False


class FirstPassStop:
    name = "first_pass"

    def should_stop(self, snap):
        return snap.n_passed > 0


class MaxDocsStop:
    def __init__(self, n: int):
        if n < 1:
            raise ValueError("max_docs >= 1")
        self.n = n
        self.name = f"max_docs:{n}"

    def should_stop(self, snap):
        return snap.n_evaluated >= self.n


class Retention(Protocol):
    name: str

    def select(
        self, docs: list[Document], hit_rank: int | None, action: str
    ) -> list[Document] | None: ...


class PolicyRetention:
    name = "policy"

    def select(self, docs, hit_rank, action):
        return None


class HitOnly:
    name = "hit_only"

    def select(self, docs, hit_rank, action):
        return None if hit_rank is None else [docs[hit_rank]]


class HitAndRest:
    name = "hit_and_rest"

    def select(self, docs, hit_rank, action):
        return None if hit_rank is None else list(docs[hit_rank:])


class KeepAll:
    name = "all"

    def select(self, docs, hit_rank, action):
        return list(docs)


class RatioRetention:
    """placeholder 정의 (README / 모듈 docstring 참고)."""

    def __init__(self, ratio: float, base: str = "rest"):
        if not 0 < ratio <= 1:
            raise ValueError("ratio: (0, 1]")
        if base not in ("rest", "all"):
            raise ValueError("ratio base: rest | all")
        self.ratio, self.base = ratio, base
        self.name = f"ratio:{ratio}:{base}"

    def select(self, docs, hit_rank, action):
        if self.base == "rest":
            if hit_rank is None:
                return None
            cand = list(docs[hit_rank:])
        else:
            cand = list(docs)
        if not cand:
            return []
        return cand[: max(1, math.ceil(self.ratio * len(cand)))]


@dataclass
class BudgetPolicy:
    name: str
    stop: StopRule
    retention: Retention

    @property
    def is_full_default(self) -> bool:
        return self.stop.name == "full" and self.retention.name == "policy"

    def describe(self) -> dict:
        return {"name": self.name, "stop_rule": self.stop.name, "retention": self.retention.name}


PRESETS = {
    "full_grading": ("full", "policy"),
    "lazy_hit": ("first_pass", "hit_only"),
    "lazy_rest": ("first_pass", "hit_and_rest"),
    "keep_all": ("full", "all"),
}


def _stop(spec) -> StopRule:
    if spec in (None, "full"):
        return FullStop()
    if spec == "first_pass":
        return FirstPassStop()
    if isinstance(spec, str) and spec.startswith("max_docs:"):
        return MaxDocsStop(int(spec.split(":", 1)[1]))
    raise ValueError(f"stop_rule: full | first_pass | max_docs:N (got {spec})")


def _retention(spec, cfg: dict) -> Retention:
    if spec in (None, "policy"):
        return PolicyRetention()
    if spec == "hit_only":
        return HitOnly()
    if spec == "hit_and_rest":
        return HitAndRest()
    if spec == "all":
        return KeepAll()
    if spec == "ratio":
        return RatioRetention(float(cfg["ratio"]), cfg.get("ratio_base", "rest"))
    raise ValueError(f"retention: policy | hit_only | hit_and_rest | all | ratio (got {spec})")


def build_budget(cfg: dict | None) -> BudgetPolicy:
    """cfg (`budget:`): {preset: lazy_rest} 또는 {stop_rule: ..., retention: ..., ratio: ...}.
    preset 과 개별 값을 같이 주면 개별 값이 우선한다."""
    cfg = dict(cfg or {})
    preset = cfg.get("preset")
    stop, ret = PRESETS.get(preset, (None, None)) if preset else (None, None)
    if preset and preset not in PRESETS:
        raise ValueError(f"budget.preset: {tuple(PRESETS)} (got {preset})")
    stop = cfg.get("stop_rule", stop)
    ret = cfg.get("retention", ret)
    s, r = _stop(stop), _retention(ret, cfg)
    name = preset if preset and (s.name, r.name) == PRESETS[preset] else f"{s.name}+{r.name}"
    if (s.name, r.name) == ("full", "policy"):
        name = "full_grading"
    return BudgetPolicy(name, s, r)
