"""진단용 Ablation: CRAG 의 문서 필터링 / 폐기 규칙만 바꾼다.

기존 CRAGGraph·Runner 는 수정하지 않는다 (하위 클래스).

설정 (config 의 최상위 `ablation:` 키, run_ablation.py 로만 실행):
  incorrect_keep_faiss : Incorrect 판정이어도 FAISS 최초 검색 top-k 를 버리지 않고
                         contexts = FAISS top-k (검색 순서) + web 문서 (Ambiguous 와 같은 순서)
  correct_keep_topk    : Correct 판정이어도 upper 미만 문서를 지우지 않고
                         contexts = FAISS top-k (검색 순서)  -> Vanilla 와 같은 context

바뀌지 않는 것: grading 호출과 점수, corrective action 판정과 기록, rewrite / web search 호출 여부,
stage 이름과 StageTimer, LLM 호출 수. 판정 이후 generation 에 넣는 contexts 만 다르다.

구분 표시: queries.jsonl 의 pipeline = "crag_graph:<grading>+abl_<이름>",
extra.ablation = {설정}, config.yaml 의 ablation 키.
"""

from __future__ import annotations

from rag.graph import CRAGGraph

ABLATIONS = ("incorrect_keep_faiss", "correct_keep_topk")
SHORT = {"incorrect_keep_faiss": "inc_keep_faiss", "correct_keep_topk": "cor_keep_topk"}


def parse_ablation(cfg: dict) -> dict[str, bool]:
    """config 의 ablation 키 검증. 없거나, 모르는 키가 있거나, 켜진 항목이 없으면 오류."""
    spec = cfg.get("ablation")
    if not isinstance(spec, dict) or not spec:
        raise ValueError(
            "ablation 설정이 없음. 예: --set ablation.incorrect_keep_faiss=true "
            "(Ablation 없는 실행은 run.py 를 사용)"
        )
    unknown = set(spec) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"알 수 없는 ablation 키: {sorted(unknown)} (가능: {ABLATIONS})")
    bad = {k: v for k, v in spec.items() if not isinstance(v, bool)}
    if bad:
        raise ValueError(f"ablation 값은 true/false: {bad}")
    out = {k: bool(spec.get(k, False)) for k in ABLATIONS}
    if not any(out.values()):
        raise ValueError("켜진 ablation 이 없음")
    if cfg.get("pipeline", "crag_graph") != "crag_graph":
        raise ValueError(f"ablation 은 pipeline=crag_graph 에서만 (got {cfg.get('pipeline')})")
    return out


class AblationCRAGGraph(CRAGGraph):
    """CRAGGraph 의 decide / web_search 노드 결과에서 contexts 만 바꾼다."""

    def __init__(self, *args, ablation: dict[str, bool], **kwargs):
        self.ablation = dict(ablation)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_graph(cls, g: CRAGGraph, ablation: dict[str, bool]) -> AblationCRAGGraph:
        tag = "+".join(f"abl_{SHORT[k]}" for k in ABLATIONS if ablation.get(k))
        return cls(
            g.retriever,
            g.llm,
            g.web_search,
            grader=g.grader,
            grader_llm=g.grader_llm,
            top_k=g.top_k,
            web_top_k=g.web_top_k,
            upper_threshold=g.upper,
            lower_threshold=g.lower,
            name=f"{g.name}+{tag}",
            ablation=ablation,
        )

    def _decide(self, state, config) -> dict:
        out = super()._decide(state, config)
        if self.ablation.get("correct_keep_topk") and out["action"] == "correct":
            out["contexts"] = list(state["docs"])
        return out

    def _web_search(self, state, config) -> dict:
        out = super()._web_search(state, config)
        if self.ablation.get("incorrect_keep_faiss") and state["action"] == "incorrect":
            out["contexts"] = list(state["docs"]) + list(out["web_docs"])
        return out

    def run(self, *args, **kwargs):
        res = super().run(*args, **kwargs)
        res.extra["ablation"] = dict(self.ablation)
        return res
