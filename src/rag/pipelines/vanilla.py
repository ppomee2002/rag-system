from rag.llm.base import LLM
from rag.pipelines import prompts
from rag.retrieval.base import Retriever
from rag.types import PipelineResult, QAExample
from rag.utils.timing import StageTimer, stopwatch


class VanillaRAG:
    name = "vanilla"

    def __init__(self, retriever: Retriever, llm: LLM, top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def run(self, ex: QAExample) -> PipelineResult:
        timer = StageTimer()
        with stopwatch() as sw:
            with timer.stage("retrieval"):
                docs = self.retriever.retrieve(ex.question, self.top_k)
            with timer.stage("generation"):
                answer = self.llm.generate(
                    prompts.GENERATE.format(
                        context=prompts.format_context(docs), question=ex.question
                    )
                )
        return PipelineResult(
            query_id=ex.id,
            pipeline=self.name,
            question=ex.question,
            answer=answer,
            contexts=docs,
            e2e_latency=sw["elapsed"],
            stage_latencies=dict(timer.totals),
        )
