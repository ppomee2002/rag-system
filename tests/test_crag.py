from rag.pipelines.crag import CRAG
from rag.types import Document, QAExample


class FakeRetriever:
    def retrieve(self, query, top_k):
        return [Document(id=str(i), text=f"doc {i}") for i in range(top_k)]


class FakeWeb:
    def search(self, query, top_k):
        return [Document(id="web", text="web doc")]


class FakeLLM:
    def __init__(self, grade: str):
        self.grade = grade

    def generate(self, prompt, **kwargs):
        return self.grade if "Score:" in prompt else "answer"


def _run(grade: str):
    crag = CRAG(FakeRetriever(), FakeLLM(grade), FakeWeb(), top_k=3)
    return crag.run(QAExample(id="q1", question="q?", answers=["a"]))


def test_correct_skips_corrective_stages():
    res = _run("0.9")
    assert res.corrective_action == "correct"
    assert res.n_corrective_steps == 0
    assert res.stage_latencies["web_search"] == 0.0


def test_incorrect_uses_web_only():
    res = _run("0.1")
    assert res.corrective_action == "incorrect"
    assert res.n_corrective_steps == 1
    assert [d.id for d in res.contexts] == ["web"]


def test_records_initial_retrieval_scores():
    class ScoredRetriever(FakeRetriever):
        def retrieve(self, query, top_k):
            return [Document(id=str(i), text="d", score=1.0 - i / 10) for i in range(top_k)]

    crag = CRAG(ScoredRetriever(), FakeLLM("0.1"), FakeWeb(), top_k=3)
    res = crag.run(QAExample(id="q1", question="q?", answers=["a"]))
    # incorrect 로 contexts 가 web 문서로 바뀌어도 최초 retrieval score 는 남는다
    assert res.extra["retrieval_scores"] == [1.0, 0.9, 0.8]


def test_parallel_grading_same_scores_and_faster():
    import time

    class SlowLLM(FakeLLM):
        def generate(self, prompt, **kwargs):
            if "Score:" in prompt:
                time.sleep(0.05)
                return "0.9" if "doc 0" in prompt else "0.1"
            return "answer"

    ex = QAExample(id="q1", question="q?", answers=["a"])
    seq = CRAG(FakeRetriever(), SlowLLM("0"), FakeWeb(), top_k=5).run(ex)
    par = CRAG(FakeRetriever(), SlowLLM("0"), FakeWeb(), top_k=5, parallel_grading=True).run(ex)
    assert par.grader_scores == seq.grader_scores == [0.9, 0.1, 0.1, 0.1, 0.1]
    assert par.stage_latencies["grading"] < seq.stage_latencies["grading"] / 2
