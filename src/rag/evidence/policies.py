"""C1: Evidence 정책 정의.

정책은 grading 판정 (correct / ambiguous / incorrect) 별로
  (1) 내부 (검색) 문서를 어떻게 남길지, (2) web correction 을 할지, (3) web 결과를 어떻게 쓸지
를 정한다. 정의의 출처는 기존 코드와 공식 코드이다.

  정책                  correct           ambiguous             incorrect        web 사용
  vanilla               (grading 없음) 검색 top-k 전부, correction 없음                    -
  crag_rule             score>=upper      score>=lower + web    web 만           raw 문서
  no_correct_filter     top-k 전부        score>=lower + web    web 만           raw 문서
  no_incorrect_discard  score>=upper      score>=lower + web    top-k + web      raw 문서
  keep_all              top-k 전부        top-k + web           top-k + web      raw 문서
  official              refine(전부)      refine + web refine   web refine       refine

  crag_rule            = rag.graph.CRAGGraph / rag.pipelines.crag.CRAG (기존 Full CRAG)
  no_correct_filter    = experiments/crag_system/ablation.py correct_keep_topk (Ablation B)
  no_incorrect_discard = experiments/crag_system/ablation.py incorrect_keep_faiss (Ablation A)
  keep_all             = 신규. 문서를 버리지 않고 correction 만 추가
                         (grading 은 web 추가 여부만 결정)
  official             = 공식 CRAG. refine = rag.crag_official.refinement
                         internal: 모든 문서의 strip 중 evaluator 점수 상위 3
                         external: web 문단 중 상위 5, ambiguous: 결합 템플릿

같은 정책 이름이라도 grader 가 다르면 의미가 달라지므로 결과에는 policy_id = "<정책>@<grader>" 와
threshold·척도를 함께 기록한다. 특히 official@llama_prompt 는 공식 정책 구조에 Llama 0~1 점수를 넣은
비공식 조합이다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

INTERNAL_MODES = ("filter_upper", "filter_lower", "all", "none", "refine")


@dataclass(frozen=True)
class EvidencePolicySpec:
    name: str
    grading: bool  # False 면 grading 단계 자체를 건너뜀 (vanilla)
    correct: str
    ambiguous: str
    incorrect: str
    web_on: tuple[str, ...]  # web correction 을 하는 판정
    web_form: str  # raw | refine
    source: str

    def internal_mode(self, action: str) -> str:
        return getattr(self, action)

    def uses_web(self, action: str) -> bool:
        return action in self.web_on

    def to_dict(self) -> dict:
        return asdict(self)


_CORR = ("ambiguous", "incorrect")

POLICIES: dict[str, EvidencePolicySpec] = {
    p.name: p
    for p in (
        EvidencePolicySpec(
            "vanilla", False, "all", "all", "all", (), "raw", "rag.pipelines.vanilla.VanillaRAG"
        ),
        EvidencePolicySpec(
            "crag_rule",
            True,
            "filter_upper",
            "filter_lower",
            "none",
            _CORR,
            "raw",
            "rag.graph.CRAGGraph (기존 Full CRAG)",
        ),
        EvidencePolicySpec(
            "no_correct_filter",
            True,
            "all",
            "filter_lower",
            "none",
            _CORR,
            "raw",
            "crag_system/ablation.py correct_keep_topk (Ablation B)",
        ),
        EvidencePolicySpec(
            "no_incorrect_discard",
            True,
            "filter_upper",
            "filter_lower",
            "all",
            _CORR,
            "raw",
            "crag_system/ablation.py incorrect_keep_faiss (Ablation A)",
        ),
        EvidencePolicySpec(
            "keep_all",
            True,
            "all",
            "all",
            "all",
            _CORR,
            "raw",
            "신규 (문서 유지 + correction 추가)",
        ),
        EvidencePolicySpec(
            "official",
            True,
            "refine",
            "refine",
            "none",
            _CORR,
            "refine",
            "HuskyInSalt/CRAG@de7c296 (knowledge refinement + combined knowledge)",
        ),
    )
}


def get_policy(name: str) -> EvidencePolicySpec:
    if name not in POLICIES:
        raise ValueError(f"evidence_policy: {tuple(POLICIES)} (got {name})")
    return POLICIES[name]


def policy_id(policy: str, grader_type: str | None, budget_name: str = "full_grading") -> str:
    base = policy if grader_type is None else f"{policy}@{grader_type}"
    return base if budget_name == "full_grading" else f"{base}+{budget_name}"


def check_combination(policy: EvidencePolicySpec, grader_type: str | None, budget) -> dict:
    """정책 x grader x budget 조합 검사.

    공식 의미가 없는 조합은 거부하고, 비공식 조합은 표시한다.
    """
    if not policy.grading:
        if not budget.is_full_default:
            raise ValueError("vanilla 정책에는 budget (grading 제어) 를 쓸 수 없음")
        return {"official_policy_semantics": False, "note": "grading 없음"}
    if policy.name == "official" and not budget.is_full_default:
        raise ValueError(
            "official 정책은 모든 문서의 refinement 점수가 필요해 Lazy/부분 grading 과 조합할 수 없음 "  # noqa: E501
            "(판정은 같아도 evidence 가 공식과 다름). C3 실험은 legacy 정책 (crag_rule 등) 으로"
        )
    semantics = policy.name == "official" and grader_type == "official_t5"
    note = None
    if policy.name == "official" and grader_type != "official_t5":
        note = "비공식 조합: 공식 정책 구조에 다른 grader 점수 사용"
    return {"official_policy_semantics": semantics, "note": note}
