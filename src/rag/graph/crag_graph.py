"""LangGraph Corrective RAG (공통 파이프라인).

    retrieve -> grade -> decide (Evidence Sufficiency) -> [rewrite -> web_search] -> generate

- grading 노드는 `GradingStrategy` 를 호출한다 (Full / Lazy / Batch / Adaptive 교체). 그래프 구조와
  stage 이름 (`rag.pipelines.crag.STAGES`) 은 전략과 무관하게 같다.
- prompt, 판정, context 구성은 기존 `rag.pipelines.crag.CRAG` 와 같다.
  Full 전략에서 두 구현의 동등성은 tests/test_crag_graph.py 가 확인한다
  (기존 CRAG 클래스는 기준 구현으로 보존).
- `run(ex)` 가 `PipelineResult` 를 돌려주므로 기존 runner 와 호환된다.
- 동시 실행: 그래프는 한 번만 compile 하고, query 마다 다른 llm (호출 기록 / scheduler 연결) 과
  query_state 를 `run(..., llm=, grader_llm=, query_state=)` 로 넘긴다. 이 값들과 StageTimer 는
  graph state 가 아니라 `config["configurable"]["ctx"]` 로 전달된다.

출처: exp/grading-optimization:experiments/grading_optimization/run_live.py 의 그래프를 공통화.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from rag.grading.base import GradingResult, GradingStrategy
from rag.grading.strategies import FullGrading
from rag.llm.base import LLM
from rag.pipelines import prompts
from rag.pipelines.crag import STAGES
from rag.retrieval.base import Retriever, WebSearch
from rag.types import Document, PipelineResult, QAExample
from rag.utils.timing import StageTimer, stopwatch


class CRAGState(TypedDict, total=False):
    question: str
    docs: list[Document]
    grading: GradingResult
    action: str
    contexts: list[Document]
    rewritten: str
    web_docs: list[Document]
    answer: str


@dataclass
class RunContext:
    llm: LLM
    grader_llm: LLM
    timer: StageTimer
    query_state: Any = None  # rag.scheduling.state.QueryState (duck typing)
    web_search: Any = (
        None  # 요청별 WebSearch (예: query_id 에 묶인 replay). None 이면 그래프 기본값
    )


def _ctx(config) -> RunContext:
    return config["configurable"]["ctx"]


def _enter(ctx: RunContext, stage: str) -> None:
    """QueryState 에 현재 stage 를 알린다 (Scheduler 정책 입력)."""
    qs = ctx.query_state
    if qs is not None and hasattr(qs, "enter_stage"):
        qs.enter_stage(stage)


class CRAGGraph:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        web_search: WebSearch,
        grader: GradingStrategy | None = None,
        grader_llm: LLM | None = None,
        top_k: int = 5,
        web_top_k: int = 5,
        upper_threshold: float = 0.7,
        lower_threshold: float = 0.3,
        name: str | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.grader_llm = grader_llm
        self.web_search = web_search
        self.grader = grader or FullGrading()
        self.top_k = top_k
        self.web_top_k = web_top_k
        self.upper = upper_threshold
        self.lower = lower_threshold
        self.name = name or f"crag_graph:{self.grader.name}"
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _retrieve(self, state: CRAGState, config) -> dict:
        ctx = _ctx(config)
        _enter(ctx, "retrieval")
        with ctx.timer.stage("retrieval"):
            docs = self.retriever.retrieve(state["question"], self.top_k)
        if ctx.query_state is not None:
            ctx.query_state.on_retrieved(docs)
        return {"docs": docs}

    def _grade(self, state: CRAGState, config) -> dict:
        ctx = _ctx(config)
        _enter(ctx, "grading")
        with ctx.timer.stage("grading"):
            result = self.grader.grade(
                state["question"], state["docs"], ctx.grader_llm, self.upper, self.lower
            )
        if ctx.query_state is not None:
            ctx.query_state.on_graded(result)
        return {"grading": result}

    def _decide(self, state: CRAGState, config) -> dict:
        """Evidence Sufficiency Decision: grading 결과로 correction 여부를 정한다."""
        g = state["grading"]
        return {"action": g.action, "contexts": list(g.contexts)}

    @staticmethod
    def _route(state: CRAGState) -> str:
        return "generate" if state["action"] == "correct" else "rewrite"

    def _rewrite(self, state: CRAGState, config) -> dict:
        ctx = _ctx(config)
        _enter(ctx, "rewrite")
        with ctx.timer.stage("rewrite"):
            query = ctx.llm.generate(prompts.REWRITE.format(question=state["question"]))
        return {"rewritten": query}

    def _web_search(self, state: CRAGState, config) -> dict:
        ctx = _ctx(config)
        _enter(ctx, "web_search")
        with ctx.timer.stage("web_search"):
            web = ctx.web_search or self.web_search
            web_docs = web.search(state["rewritten"].strip(), self.web_top_k)
        contexts = web_docs if state["action"] == "incorrect" else state["contexts"] + web_docs
        return {"web_docs": web_docs, "contexts": contexts}

    def _generate(self, state: CRAGState, config) -> dict:
        ctx = _ctx(config)
        _enter(ctx, "generation")
        with ctx.timer.stage("generation"):
            answer = ctx.llm.generate(
                prompts.GENERATE.format(
                    context=prompts.format_context(state["contexts"]), question=state["question"]
                )
            )
        return {"answer": answer}

    def _build(self):
        g = StateGraph(CRAGState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("decide", self._decide)
        g.add_node("rewrite", self._rewrite)
        g.add_node("web_search", self._web_search)
        g.add_node("generate", self._generate)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_edge("grade", "decide")
        g.add_conditional_edges(
            "decide", self._route, {"generate": "generate", "rewrite": "rewrite"}
        )
        g.add_edge("rewrite", "web_search")
        g.add_edge("web_search", "generate")
        g.add_edge("generate", END)
        return g.compile()

    # ------------------------------------------------------------------ run
    def run(
        self,
        ex: QAExample,
        llm: LLM | None = None,
        grader_llm: LLM | None = None,
        query_state: Any = None,
        web_search: Any = None,
    ) -> PipelineResult:
        llm = llm or self.llm
        ctx = RunContext(
            llm=llm,
            grader_llm=grader_llm or self.grader_llm or llm,
            timer=StageTimer(),
            query_state=query_state,
            web_search=web_search,
        )
        with stopwatch() as sw:
            out = self.graph.invoke({"question": ex.question}, {"configurable": {"ctx": ctx}})
        g: GradingResult = out["grading"]
        corrected = out["action"] != "correct"
        return PipelineResult(
            query_id=ex.id,
            pipeline=self.name,
            question=ex.question,
            answer=out["answer"],
            contexts=out["contexts"],
            e2e_latency=sw["elapsed"],
            stage_latencies={s: ctx.timer.totals.get(s, 0.0) for s in STAGES},
            corrective_action=out["action"],
            grader_scores=g.graded_scores,
            n_corrective_steps=int(corrected),
            extra={
                "stage_counts": dict(ctx.timer.counts),
                "retrieval_scores": [d.score for d in out["docs"]],
                "retrieval_doc_ids": [d.id for d in out["docs"]],  # 조건 간 검색 결과 동일성 확인용
                "grading": {
                    "strategy": self.grader.name,
                    "scores_full": g.scores,  # 검색 순서, None = 평가 안 함
                    "n_graded": g.n_graded,
                    "n_llm_calls": g.n_llm_calls,
                    "info": g.info,
                },
                "rewritten_query": out.get("rewritten"),
                "web_doc_ids": [d.id for d in out.get("web_docs", [])],
            },
        )
