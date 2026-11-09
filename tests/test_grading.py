"""Grading 전략 단위 테스트: Lazy 중단, Batch 파싱·fallback, Adaptive 분기 (기능 검증용)."""

import pytest

from rag.grading import (
    AdaptiveGrading,
    BatchGrading,
    FullGrading,
    LazyGrading,
    build_grader,
    decide_action,
    parse_batch_scores,
)
from rag.telemetry.calls import infer_call_kind
from rag.types import Document

U, L = 0.7, 0.3


def docs(retrieval=(0.9, 0.8, 0.7, 0.6, 0.5)):
    return [Document(id=f"d{i}", text=f"doc{i}", score=s) for i, s in enumerate(retrieval)]


class LLM:
    def __init__(self, scores, batch_replies=()):
        self.scores = scores
        self.batch_replies = list(batch_replies)
        self.calls = []

    def generate(self, prompt, **kwargs):
        kind = kwargs.get("kind") or infer_call_kind(prompt)
        self.calls.append(kind)
        if kind == "grade":
            i = int(prompt.rsplit("Document: doc", 1)[1].split("\n", 1)[0])
            return str(self.scores[i])
        assert kind == "grade_batch"
        return self.batch_replies.pop(0)


def test_decide_action_matches_crag_rules():
    assert decide_action([0.1, 0.7], U, L) == "correct"
    assert decide_action([0.1, 0.29], U, L) == "incorrect"
    assert decide_action([0.3, 0.1], U, L) == "ambiguous"
    assert decide_action([None, 0.1, None], U, L) == "incorrect"
    assert decide_action([], U, L) == "incorrect"


def test_lazy_hit_and_rest_stop_at_first_hit():
    llm = LLM([0.1, 0.8, 0.9, 0.0, 0.0])
    hit = LazyGrading("hit").grade("q", docs(), llm, U, L)
    assert llm.calls == ["grade", "grade"]
    assert hit.scores == [0.1, 0.8, None, None, None]
    assert [d.id for d in hit.contexts] == ["d1"] and hit.action == "correct"
    rest = LazyGrading("rest").grade("q", docs(), LLM([0.1, 0.8, 0.9, 0, 0]), U, L)
    assert [d.id for d in rest.contexts] == ["d1", "d2", "d3", "d4"]


def test_lazy_without_hit_equals_full():
    scores = [0.4, 0.1, 0.2, 0.5, 0.0]
    lazy = LazyGrading().grade("q", docs(), LLM(scores), U, L)
    full = FullGrading().grade("q", docs(), LLM(scores), U, L)
    assert (lazy.scores, lazy.action, lazy.contexts) == (full.scores, full.action, full.contexts)


@pytest.mark.parametrize(
    "text,expected,status",
    [
        ('{"scores": [0.1, 0.9, 0.3]}', [0.1, 0.9, 0.3], "ok"),
        ('Sure! ```json\n{"scores":[0,1,0.5]}\n```', [0.0, 1.0, 0.5], "ok"),
        ("[0.2, 0.4, 0.6]", [0.2, 0.4, 0.6], "ok_list"),
        ("[1]: 0.9\n[2]: 0.1\n[3]: 0.3", [0.9, 0.1, 0.3], "indexed"),
        ("Document 1 = 0.9, Document 3 = 0.2", [0.9, None, 0.2], "partial"),
        ('{"scores": [0.1, 0.9]}', [None, None, None], "length_mismatch:2"),
        ('{"scores": [0.1, 7, "x"]}', [0.1, None, None], "partial"),
        ("I cannot judge these documents.", [None, None, None], "unparsed"),
    ],
)
def test_parse_batch_scores(text, expected, status):
    assert parse_batch_scores(text, 3) == (expected, status)


def test_batch_single_call_for_five_docs():
    llm = LLM([0] * 5, ['{"scores": [0.1, 0.2, 0.9, 0.1, 0.4]}'])
    r = BatchGrading().grade("q", docs(), llm, U, L)
    assert llm.calls == ["grade_batch"]
    assert r.n_llm_calls == 1 and r.action == "correct"
    assert [d.id for d in r.contexts] == ["d2"]
    assert r.info["fallback_indices"] == []


def test_batch_fallback_individual_regrades_only_invalid():
    llm = LLM([0.0, 0.8, 0.0, 0.0, 0.5], ['{"scores": [0.1, 1.5, 0.2, 0.1, "?"]}'])
    r = BatchGrading().grade("q", docs(), llm, U, L)
    assert llm.calls == ["grade_batch", "grade", "grade"]
    assert r.scores == [0.1, 0.8, 0.2, 0.1, 0.5]
    assert r.info["fallback_indices"] == [1, 4] and r.n_llm_calls == 3


def test_batch_fallback_zero_and_unparsed():
    llm = LLM([0.9] * 5, ["no idea"])
    r = BatchGrading(fallback="zero").grade("q", docs(), llm, U, L)
    assert llm.calls == ["grade_batch"]
    assert r.scores == [0.0] * 5 and r.action == "incorrect"
    assert r.info["batch_status"] == ["unparsed"]


def test_batch_size_splits_calls():
    llm = LLM([0] * 5, ['{"scores": [0.1, 0.1]}', '{"scores": [0.1, 0.1]}', '{"scores": [0.9]}'])
    r = BatchGrading(batch_size=2).grade("q", docs(), llm, U, L)
    assert llm.calls == ["grade_batch"] * 3 and r.action == "correct"


def test_adaptive_probe_hit_stops():
    llm = LLM([0.9, 0, 0, 0, 0])
    r = AdaptiveGrading().grade("q", docs(), llm, U, L)
    assert llm.calls == ["grade"] and r.info["stop_reason"] == "probe_hit"
    assert [d.id for d in r.contexts] == ["d0"]


def test_adaptive_probe_then_batch_rest():
    llm = LLM([0.1, 0, 0, 0, 0], ['{"scores": [0.2, 0.5, 0.1, 0.0]}'])
    r = AdaptiveGrading().grade("q", docs(), llm, U, L)
    assert llm.calls == ["grade", "grade_batch"]
    assert r.action == "ambiguous" and r.info["stop_reason"] == "graded_rest"
    assert [d.id for d in r.contexts] == ["d2"]


def test_adaptive_skip_low_retrieval_and_early_incorrect():
    llm = LLM([0.1, 0.0, 0.0, 0.0, 0.0])
    r = AdaptiveGrading(skip_below_retrieval=0.65, early_incorrect_retrieval=0.85).grade(
        "q", docs(), llm, U, L
    )
    assert r.info["skipped_low_retrieval"] == [3, 4]
    assert llm.calls == ["grade"] and r.info["stop_reason"] == "early_incorrect"
    assert r.action == "incorrect" and r.scores == [0.1, None, None, None, None]


def test_adaptive_early_incorrect_not_triggered_when_rest_retrieval_high():
    llm = LLM([0.1, 0.2, 0.1, 0.0, 0.0])
    r = AdaptiveGrading(rest_mode="sequential", early_incorrect_retrieval=0.75).grade(
        "q", docs(), llm, U, L
    )
    assert llm.calls == ["grade"] * 5 and r.info["stop_reason"] == "graded_rest"


def test_adaptive_all_skipped():
    r = AdaptiveGrading(skip_below_retrieval=0.95).grade("q", docs(), LLM([0] * 5), U, L)
    assert r.n_llm_calls == 0 and r.action == "incorrect"


def test_build_grader():
    assert build_grader(None).name == "full"
    assert build_grader({"strategy": "full", "parallel": True}).name == "full_parallel"
    assert build_grader({"strategy": "lazy_hit"}).name == "lazy_hit"
    assert build_grader({"strategy": "lazy_rest"}).name == "lazy_rest"
    assert build_grader({"strategy": "batch", "fallback": "zero"}).fallback == "zero"
    assert build_grader({"strategy": "adaptive", "probe_k": 2}).probe_k == 2
    with pytest.raises(ValueError):
        build_grader({"strategy": "nope"})
