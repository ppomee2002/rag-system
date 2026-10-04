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
