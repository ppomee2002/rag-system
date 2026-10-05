"""Corrective RAG (Yan et al., 2024).

retrieval -> grading -> action 결정
  correct   : 검색 문서 중 relevant 한 것만 사용
  incorrect : query rewrite -> web search 결과만 사용
  ambiguous : relevant 문서 + rewrite/web search 결과 함께 사용
-> generation

각 단계는 StageTimer 의 stage 이름으로 기록된다:
retrieval / grading / rewrite / web_search / generation
"""

import re

from rag.llm.base import LLM
from rag.pipelines import prompts
from rag.retrieval.base import Retriever, WebSearch
from rag.types import Document, PipelineResult, QAExample
from rag.utils.timing import StageTimer, stopwatch

STAGES = ("retrieval", "grading", "rewrite", "web_search", "generation")


class CRAG:
    name = "crag"

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        web_search: WebSearch,
        grader_llm: LLM | None = None,
        top_k: int = 5,
        web_top_k: int = 5,
        upper_threshold: float = 0.7,
        lower_threshold: float = 0.3,
    ):
        self.retriever = retriever
        self.llm = llm
        self.grader_llm = grader_llm or llm
        self.web_search = web_search
        self.top_k = top_k
        self.web_top_k = web_top_k
        self.upper = upper_threshold
        self.lower = lower_threshold

    def grade(self, question: str, doc: Document) -> float:
        out = self.grader_llm.generate(
            prompts.GRADE.format(question=question, document=doc.text)
        )
        m = re.search(r"\d*\.?\d+", out)
        return min(max(float(m.group()), 0.0), 1.0) if m else 0.0

    def decide(self, scores: list[float]) -> str:
        if any(s >= self.upper for s in scores):
            return "correct"
        if all(s < self.lower for s in scores):
            return "incorrect"
        return "ambiguous"

    def run(self, ex: QAExample) -> PipelineResult:
        timer = StageTimer()
        n_corrective = 0
        with stopwatch() as sw:
            with timer.stage("retrieval"):
                docs = self.retriever.retrieve(ex.question, self.top_k)

            with timer.stage("grading"):
                scores = [self.grade(ex.question, d) for d in docs]
            action = self.decide(scores)

            contexts = [d for d, s in zip(docs, scores) if s >= self.lower]
            if action == "correct":
                contexts = [d for d, s in zip(docs, scores) if s >= self.upper]
            else:
                with timer.stage("rewrite"):
                    query = self.llm.generate(prompts.REWRITE.format(question=ex.question))
                with timer.stage("web_search"):
                    web_docs = self.web_search.search(query.strip(), self.web_top_k)
                n_corrective += 1
                contexts = web_docs if action == "incorrect" else contexts + web_docs

            with timer.stage("generation"):
                answer = self.llm.generate(
                    prompts.GENERATE.format(
                        context=prompts.format_context(contexts), question=ex.question
                    )
                )

        return PipelineResult(
            query_id=ex.id,
            pipeline=self.name,
            question=ex.question,
            answer=answer,
            contexts=contexts,
            e2e_latency=sw["elapsed"],
            stage_latencies={s: timer.totals.get(s, 0.0) for s in STAGES},
            corrective_action=action,
            grader_scores=scores,
            n_corrective_steps=n_corrective,
            extra={
                "stage_counts": dict(timer.counts),
                "retrieval_scores": [d.score for d in docs],  # 최초 retrieval score (난이도 기준)
            },
        )
