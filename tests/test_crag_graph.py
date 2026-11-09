"""LangGraph CRAG 의 분기·상태 전이와 기존 CRAG 와의 동등성 (기능 검증용, 성능·정확도 결과 아님)."""

import threading

import pytest

from rag.grading import AdaptiveGrading, BatchGrading, FullGrading, LazyGrading
from rag.graph import CRAGGraph
from rag.pipelines import prompts
from rag.pipelines.crag import CRAG, STAGES
from rag.telemetry.calls import infer_call_kind
from rag.types import Document, QAExample

Q = "Who wrote Hamlet?"
DOCS = [Document(id=f"d{i}", text=f"doc{i}", score=0.9 - 0.1 * i) for i in range(5)]
WEB = [Document(id=f"w{i}", text=f"web{i}") for i in range(2)]


class ScriptedLLM:
    """prompt -> 응답. grade 는 문서별 점수표, batch 는 지정 문자열, 나머지는 고정."""

    def __init__(self, scores, batch_reply=None):
        self.scores = dict(zip((d.text for d in DOCS), scores, strict=True))
        self.batch_reply = batch_reply
        self.prompts = []
        self.lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        with self.lock:
            self.prompts.append(prompt)
        kind = kwargs.get("kind") or infer_call_kind(prompt)
        if kind == "grade":
            doc = prompt.rsplit("Document: ", 1)[1].split("\n", 1)[0]
            return str(self.scores[doc])
        if kind == "grade_batch":
            return self.batch_reply
        if kind == "rewrite":
            return " rewritten query "
        return "Shakespeare"


class Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def retrieve(self, query, top_k):
        self.queries.append(query)
        return self.docs[:top_k]

    def search(self, query, top_k):
        self.queries.append(query)
        return WEB[:top_k]


EX = QAExample(id="q1", question=Q, answers=["Shakespeare"])
BRANCHES = {
    "correct": [0.2, 0.9, 0.1, 0.8, 0.0],
    "incorrect": [0.1, 0.2, 0.0, 0.1, 0.2],
    "ambiguous": [0.4, 0.2, 0.1, 0.5, 0.2],
}


@pytest.mark.parametrize("branch", list(BRANCHES))
@pytest.mark.parametrize("parallel", [False, True])
def test_full_graph_matches_reference_crag(branch, parallel):
    scores = BRANCHES[branch]
    old_llm, new_llm = ScriptedLLM(scores), ScriptedLLM(scores)
    old_web, new_web = Retriever(DOCS), Retriever(DOCS)
    old = CRAG(Retriever(DOCS), old_llm, old_web, parallel_grading=parallel).run(EX)
    new = CRAGGraph(Retriever(DOCS), new_llm, new_web, grader=FullGrading(parallel)).run(EX)

    assert new.corrective_action == old.corrective_action == branch
    assert new.answer == old.answer
    assert [d.id for d in new.contexts] == [d.id for d in old.contexts]
    assert new.grader_scores == old.grader_scores
    assert new.n_corrective_steps == old.n_corrective_steps
    assert set(new.stage_latencies) == set(STAGES)
    assert new.extra["stage_counts"] == old.extra["stage_counts"]
    assert new_web.queries == old_web.queries  # rewrite 결과로 같은 web search
    if parallel:
        assert sorted(new_llm.prompts) == sorted(old_llm.prompts)
    else:
        assert new_llm.prompts == old_llm.prompts  # prompt 내용과 호출 순서까지 동일


def test_graph_state_transitions_per_branch():
    for branch, scores in BRANCHES.items():
        llm = ScriptedLLM(scores)
        res = CRAGGraph(Retriever(DOCS), llm, Retriever(DOCS)).run(EX)
        kinds = [infer_call_kind(p) for p in llm.prompts]
        if branch == "correct":
            assert kinds == ["grade"] * 5 + ["generate"]
            assert [d.id for d in res.contexts] == ["d1", "d3"]
            assert res.stage_latencies["rewrite"] == 0.0
        else:
            assert kinds == ["grade"] * 5 + ["rewrite", "generate"]
            assert res.extra["rewritten_query"] == " rewritten query "
            assert res.extra["web_doc_ids"] == ["w0", "w1"]
        if branch == "incorrect":
            assert [d.id for d in res.contexts] == ["w0", "w1"]
        if branch == "ambiguous":
            assert [d.id for d in res.contexts] == ["d0", "d3", "w0", "w1"]


def test_grader_is_swappable_and_reports_calls():
    scores = BRANCHES["correct"]
    lazy = CRAGGraph(
        Retriever(DOCS), ScriptedLLM(scores), Retriever(DOCS), grader=LazyGrading("hit")
    ).run(EX)
    assert lazy.extra["grading"]["n_llm_calls"] == 2
    assert lazy.extra["grading"]["scores_full"] == [0.2, 0.9, None, None, None]
    assert [d.id for d in lazy.contexts] == ["d1"]
    assert lazy.pipeline == "crag_graph:lazy_hit"

    reply = '{"scores": [0.2, 0.9, 0.1, 0.8, 0.0]}'
    batch = CRAGGraph(
        Retriever(DOCS), ScriptedLLM(scores, reply), Retriever(DOCS), grader=BatchGrading()
    ).run(EX)
    assert batch.extra["grading"]["n_llm_calls"] == 1
    assert batch.corrective_action == "correct"

    adaptive = CRAGGraph(
        Retriever(DOCS),
        ScriptedLLM(scores, '{"scores": [0.9, 0.1, 0.8, 0.0]}'),
        Retriever(DOCS),
        grader=AdaptiveGrading(),
    ).run(EX)
    assert adaptive.extra["grading"]["n_llm_calls"] == 2  # probe 1 + batch 1
    assert adaptive.extra["grading"]["info"]["stop_reason"] == "graded_rest"


class StateSpy:
    def __init__(self):
        self.events = []

    def on_retrieved(self, docs):
        self.events.append(("retrieved", len(docs)))

    def on_graded(self, result):
        self.events.append(("graded", result.action))


def test_per_query_llm_and_query_state_hooks():
    default_llm = ScriptedLLM(BRANCHES["correct"])
    per_query = ScriptedLLM(BRANCHES["incorrect"])
    graph = CRAGGraph(Retriever(DOCS), default_llm, Retriever(DOCS))
    spy = StateSpy()
    res = graph.run(EX, llm=per_query, query_state=spy)
    assert res.corrective_action == "incorrect"
    assert default_llm.prompts == []
    assert spy.events == [("retrieved", 5), ("graded", "incorrect")]


def test_separate_grader_llm():
    gen, grader = ScriptedLLM(BRANCHES["correct"]), ScriptedLLM(BRANCHES["correct"])
    CRAGGraph(Retriever(DOCS), gen, Retriever(DOCS), grader_llm=grader).run(EX)
    assert {infer_call_kind(p) for p in grader.prompts} == {"grade"}
    assert [infer_call_kind(p) for p in gen.prompts] == ["generate"]


def test_prompts_unchanged_from_reference():
    """기존 실험과의 호환: 공통 prompt 를 그대로 사용한다."""
    llm = ScriptedLLM(BRANCHES["correct"])
    CRAGGraph(Retriever(DOCS), llm, Retriever(DOCS)).run(EX)
    assert llm.prompts[0] == prompts.GRADE.format(question=Q, document="doc0")
