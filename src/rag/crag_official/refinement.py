"""공식 CRAG Knowledge Refinement (decompose-then-recompose) 재구현.

근거 (기준 commit):
  extract_strips    internal_knowledge_preparation.py:11-56
  select_relevants  utils.py:274-299
  internal (Correct)   internal_knowledge_preparation.py:58-85 — 검색 문서 전부를 strip 으로 나눠
                       evaluator 점수순 상위 top_n (selection 3 / 그 외 6) 을 '; ' 로 결합.
                       ** threshold 로 문서를 거르지 않는다 ** (기존 CRAG-rule 과 다른 점)
  external (Incorrect) external_knowledge_preparation.py:96-162 — 검색 결과 페이지 문단
                       ("제목: 문단", 10자 초과) 을 같은 방식으로 상위 5개.
                       문단이 없으면 snippet 결합
  ambiguous            combined_knowledge_preparation.py:14
                       "Knowledge1: 내부 [sep] Knowledge2: 외부"

공식과 다르게 처리한 것 (README 표에 기록):
  - fixed_num 에서 문서가 10 단어 미만이면 원본은 IndexError -> 여기서는 strip 하나로 둔다
  - external 의 페이지 수집 (Google/serper + HTML 파싱) 대신 주어진 web 문서의 줄 단위 문단 사용
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rag.crag_official import source
from rag.crag_official.evaluator import OfficialRetrievalEvaluator, PairScore

DECOMPOSE_MODES = ("selection", "excerption", "fixed_num")


def extract_strips(psg: str, mode: str = "excerption") -> list[str]:
    if mode == "fixed_num":
        final: list[str] = []
        buf: list[str] = []
        for w in psg.split(" "):
            buf.append(w)
            if len(buf) == source.FIXED_NUM_WINDOW:
                final.append(" ".join(buf))
                buf = []
        if buf:
            if len(buf) < 10 and final:
                final[-1] += " " + " ".join(buf)
            else:  # 원본: final 이 비어 있으면 IndexError (deviation)
                final.append(" ".join(buf))
        return final
    if mode == "excerption":
        origin: list[str] = []
        for qs in psg.split("?"):
            origin += qs.split(". ")
        strips: list[str] = []
        for s in origin:
            if s in strips:
                continue
            if not strips:
                strips.append(s)
            elif len(s.split()) > 5:
                strips.append(s)
            else:
                strips[-1] += s
        final, buf = [], []
        for s in strips:
            buf.append(s)
            if len(buf) == source.EXCERPTION_CONCAT:
                final.append(" ".join(buf))
                buf = []
        if buf:
            final.append(" ".join(buf))
        return final
    if mode == "selection":
        return [psg]
    raise ValueError(f"decompose_mode: {DECOMPOSE_MODES} (got {mode})")


@dataclass
class Strip:
    index: int  # 전체 strip 목록에서의 위치 (공식 idxs)
    text: str
    origin: str | None  # 출처 문서 id (있으면)
    score: float
    score_source: str


@dataclass
class RefinementResult:
    text: str  # 공식 형식 ('; ' 결합, 줄바꿈 -> 공백)
    selected: list[Strip]
    strips: list[Strip]  # 전체 (점수 포함)
    ranking: list[int]  # 점수 내림차순 strip index (공식 idxs)
    top_n: int
    decompose_mode: str
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_origins(self) -> list[str]:
        seen: list[str] = []
        for s in self.selected:
            if s.origin is not None and s.origin not in seen:
                seen.append(s.origin)
        return seen

    def to_dict(self, with_text: bool = True) -> dict:
        d = asdict(self)
        if not with_text:
            for s in d["selected"] + d["strips"]:
                s["text"] = None
        return d


def select_relevants(
    strips: list[str],
    origins: list[str | None],
    query: str,
    evaluator: OfficialRetrievalEvaluator,
    top_n: int,
    decompose_mode: str = "selection",
) -> RefinementResult:
    scores: list[PairScore] = evaluator.score_strips(query, strips)
    data = [
        Strip(i, s, o, p.score, p.source)
        for i, (s, o, p) in enumerate(zip(strips, origins, scores, strict=True))
    ]
    ranked = sorted(data, key=lambda x: x.score, reverse=True)  # 안정 정렬 (원본과 같음)
    selected = ranked[:top_n]
    text = source.STRIP_JOIN.join(s.text for s in selected).replace("\n", " ")
    return RefinementResult(
        text=text,
        selected=selected,
        strips=data,
        ranking=[s.index for s in ranked],
        top_n=top_n,
        decompose_mode=decompose_mode,
        info={
            "n_tokens": [p.n_tokens for p in scores],
            "compute_seconds": sum(p.compute_seconds or 0.0 for p in scores),
            "n_errors": sum(p.source == "error" for p in scores),
        },
    )


def internal_knowledge(
    query: str,
    passages: list[str],
    evaluator: OfficialRetrievalEvaluator,
    decompose_mode: str = source.DEFAULT_DECOMPOSE_MODE,
    origins: list[str | None] | None = None,
    top_n: int | None = None,
) -> RefinementResult:
    """Correct 경로 지식. passages 는 검색 순서 전체 (필터링하지 않는다)."""
    origins = origins or [None] * len(passages)
    strips: list[str] = []
    strip_origins: list[str | None] = []
    for p, o in zip(passages, origins, strict=True):
        parts = extract_strips(p, decompose_mode)
        strips += parts
        strip_origins += [o] * len(parts)
    n = top_n if top_n is not None else source.INTERNAL_TOP_N[decompose_mode]
    return select_relevants(strips, strip_origins, query, evaluator, n, decompose_mode)


def official_passages_from_line(retrieved_psgs_line: str) -> list[str]:
    """공식 retrieved_psgs 한 줄 -> passage 목록 (internal_knowledge_preparation.py:103 과 같이
    '[sep]' 로만 나눠 앞뒤 공백을 남긴다)."""
    return retrieved_psgs_line.strip().split("[sep]")


def web_strips(web_docs) -> tuple[list[str], list[str | None]]:
    """web 문서 -> "제목: 문단" strip (공식 test_page_loader 의 10자 초과 문단 규칙)."""
    strips: list[str] = []
    origins: list[str | None] = []
    for d in web_docs:
        title = (d.metadata or {}).get("title") or ""
        for para in d.text.split("\n"):
            if len(para) > 10:
                strips.append(f"{title}: {para}" if title else para)
                origins.append(d.id)
    return strips, origins


def external_knowledge(
    query: str,
    web_docs,
    evaluator: OfficialRetrievalEvaluator,
    top_n: int = source.EXTERNAL_TOP_N,
) -> RefinementResult:
    """Incorrect 경로 지식. 문단이 하나도 없으면 원본처럼 문서 본문 (snippet) 을 '; ' 로 결합."""
    strips, origins = web_strips(web_docs)
    if not strips:
        text = source.STRIP_JOIN.join(d.text for d in web_docs).replace("\n", " ")
        return RefinementResult(
            text=text,
            selected=[],
            strips=[],
            ranking=[],
            top_n=top_n,
            decompose_mode="web_paragraph",
            info={"fallback": "snippets" if web_docs else "no_web_docs"},
        )
    res = select_relevants(strips, origins, query, evaluator, top_n, "web_paragraph")
    return res


def ambiguous_knowledge(internal: str, external: str) -> str:
    return source.AMBIGUOUS_TEMPLATE.format(internal=internal, external=external)
