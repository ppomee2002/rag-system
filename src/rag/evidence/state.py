"""C2: 문서별 Grading 중간 상태 (event / snapshot).

grading 을 검색 순서대로 한 문서씩 하면서, 매 문서 뒤의 상태를 snapshot 으로 남긴다.
C2 는 이것으로 "여기서 멈췄다면" 의 판정·evidence·비용을 재구성해 추가 grading 의 필요성과 중단 시
품질 손실을 분석한다.

공식 판정과의 관계 (공식 Evaluator 는 문서별 점수가 서로 독립):
  - k 번째 snapshot 의 점수들은 전체 평가 때의 앞 k 개 점수와 같다.
  - provisional_action 이 correct 이면 최종 판정도 correct (flag 2 는 남은 문서와 무관하게 확정).
  - provisional_action 이 ambiguous / incorrect 이면 남은 문서에 따라 바뀔 수 있다
    (action_final=False). 이 경우 snapshot 판정은 공식 판정이 아니다.
  - 공식 Correct 의 evidence (refinement 상위 3개) 는 모든 문서의 refinement 점수가 필요하므로
    중간 snapshot 의 passed_ids 는 공식 evidence 가 아니다 (기존 CRAG-rule 기준 후보일 뿐).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from rag.crag_official.evaluator import official_action
from rag.evidence.graders import DocJudgment


@dataclass
class GradingSnapshot:
    n_evaluated: int  # 평가 완료 문서 수 (= 다음에 평가할 문서의 rank)
    n_total: int
    n_passed: int  # flag 2 누적
    n_partial: int  # flag 1 누적
    n_failed: int
    cum_input_tokens: int | None  # 하나라도 모르면 None
    cum_output_tokens: int | None
    cum_latency: float
    passed_ids: list[str]  # 지금까지 통과 (flag 2) 문서 (검색 순서)
    partial_ids: list[str]
    evaluated_ids: list[str]
    provisional_action: str
    action_final: bool  # 이 시점 판정이 남은 문서와 무관하게 확정인지
    first_pass_rank: int | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GradingTracker:
    query_id: str
    n_total: int
    judgments: list[DocJudgment] = field(default_factory=list)
    snapshots: list[GradingSnapshot] = field(default_factory=list)

    def add(self, j: DocJudgment) -> GradingSnapshot:
        self.judgments.append(j)
        snap = self._snapshot()
        self.snapshots.append(snap)
        return snap

    @property
    def last(self) -> GradingSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    def _snapshot(self) -> GradingSnapshot:
        js = self.judgments

        def cum(attr):
            vals = [getattr(j, attr) for j in js]
            return None if any(v is None for v in vals) else int(sum(vals))

        action = official_action(j.flag for j in js)
        first = next((j.rank for j in js if j.flag == 2), None)
        return GradingSnapshot(
            n_evaluated=len(js),
            n_total=self.n_total,
            n_passed=sum(j.flag == 2 for j in js),
            n_partial=sum(j.flag == 1 for j in js),
            n_failed=sum(j.flag == 0 for j in js),
            cum_input_tokens=cum("input_tokens"),
            cum_output_tokens=cum("output_tokens"),
            cum_latency=sum(j.latency for j in js),
            passed_ids=[j.doc_id for j in js if j.flag == 2],
            partial_ids=[j.doc_id for j in js if j.flag == 1],
            evaluated_ids=[j.doc_id for j in js],
            provisional_action=action,
            action_final=action == "correct" or len(js) == self.n_total,
            first_pass_rank=first,
        )

    def events(self, request_id: str | None = None, grader_type: str | None = None) -> list[dict]:
        return [
            {
                "query_id": self.query_id,
                "request_id": request_id,
                "grader_type": grader_type,
                "eval_order": i,
                **j.to_dict(),
            }
            for i, j in enumerate(self.judgments)
        ]
