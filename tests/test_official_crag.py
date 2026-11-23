"""공식 CRAG 재구현 (rag.crag_official) 과 C1/C2/C3 인터페이스 (rag.evidence) 단위 테스트.

모두 mock evaluator / 가짜 엔진으로 동작만 확인한다. 실제 T5 가중치 추론 결과가 아니며
정확도·성능 결과로 쓰지 않는다 (실제 추론 검증은 서버의 smoke_evaluator.py).
"""

import json
import random

import pytest

from rag.crag_official import generation as offgen
from rag.crag_official import source
from rag.crag_official.evaluator import (
    CachedScoreBackend,
    EvaluatorWeightsError,
    MockScoreBackend,
    OfficialRetrievalEvaluator,
    PairScore,
    check_weights_dir,
    official_action,
    official_doc_flag,
    pair_key,
    process_flag,
    resolve_model_path,
)
from rag.crag_official.official_data import (
    attach_answers,
    load_official_queries,
    load_ref,
    parse_pair_line,
    retrieved_psgs_line,
)
from rag.crag_official.refinement import (
    ambiguous_knowledge,
    external_knowledge,
    extract_strips,
    internal_knowledge,
    official_passages_from_line,
)
from rag.grading.base import decide_action
from rag.types import Document

UP, LO = 0.592, -0.995


def ev(fn, **kw):
    return OfficialRetrievalEvaluator(MockScoreBackend(fn), **kw)


# ---------------------------------------------------------------- decision
def test_official_thresholds_are_negated_lower():
    assert source.OFFICIAL_THRESHOLDS["popqa"] == {"upper": 0.592, "lower": -0.995}


@pytest.mark.parametrize(
    "scores,action",
    [
        ([0.6, -2, -2], "correct"),  # 하나라도 >= upper
        ([0.592, -5], "correct"),  # 경계 포함
        ([0.5, -0.995], "ambiguous"),  # lower 경계 포함
        ([-0.996, -3], "incorrect"),
        ([-1.0] * 10, "incorrect"),  # 빈 문서 (-1.0) 는 lower 미만
        ([], "incorrect"),
    ],
)
def test_official_action(scores, action):
    assert official_action(official_doc_flag(s, UP, LO) for s in scores) == action


def test_process_flag_groups_by_ndocs():
    scores = [0.7] + [-2] * 9 + [0.0] * 10 + [-2] * 10
    assert process_flag(scores, 10, UP, LO) == [2, 1, 0]


def test_official_rule_equals_legacy_rule():
    """flag any() 규칙 == 기존 decide_action (any>=upper / all<lower / 그 외)."""
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(0, 6)
        s = [rng.choice([-1.0, -0.995, 0.0, 0.3, 0.592, 0.7, 1.2]) for _ in range(n)]
        for up, lo in ((UP, LO), (0.7, 0.3)):
            assert official_action(official_doc_flag(x, up, lo) for x in s) == decide_action(
                s, up, lo
            )


# ---------------------------------------------------------------- evaluator
def test_empty_passage_scored_minus_one_without_model():
    e = ev(lambda q, p: 5.0)
    out = e.score_documents("q?", ["", "  ", "some text"])
    assert [p.score for p in out] == [-1.0, -1.0, 5.0]
    assert [p.source for p in out[:2]] == ["empty_rule", "empty_rule"]
    assert e.backend.calls == [("q?", "some text", False)]  # 판정 점수는 truncation 없음


def test_passage_normalized_like_official():
    e = ev(lambda q, p: 0.0)
    e.score_documents("q", [" a\nb\tc "])
    assert e.backend.calls[0][1] == "a b c"


def test_strip_scoring_short_rule_and_truncation():
    e = ev(lambda q, p: 1.0)
    out = e.score_strips("q", ["too short", "this strip has five words"])
    assert out[0].score == -1.0 and out[0].source == "short_strip_rule"
    assert out[1].score == 1.0
    assert e.backend.calls == [("q", "this strip has five words", True)]


def test_strip_backend_error_becomes_minus_one():
    def boom(q, p):
        raise RuntimeError("cuda")

    out = ev(boom).score_strips("q", ["one two three four five"])
    assert out[0].score == -1.0 and out[0].source == "error"


def test_evaluate_verdict():
    e = ev(lambda q, p: {"good": 0.9, "meh": 0.0}.get(p.split()[0], -3.0))
    v = e.evaluate("q", ["meh x", "bad x", ""])
    assert v.flags == [1, 0, 0] and v.action == "ambiguous"
    assert e.evaluate("q", ["good x", "bad"]).action == "correct"


def test_cached_backend(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [
        {"type": "meta", "model": "x"},
        {"key": pair_key("q", "doc a", False), "score": 0.8, "n_tokens": 7, "compute_seconds": 0.1},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    b = CachedScoreBackend([p])
    assert b.score_pairs([("q", "doc a")], truncation=False)[0].score == 0.8
    with pytest.raises(KeyError):
        b.score_pairs([("q", "doc a")], truncation=True)  # 다른 모드는 다른 key
    b2 = CachedScoreBackend([p], fallback=MockScoreBackend(lambda q, d: -0.5), on_miss="delegate")
    assert b2.score_pairs([("q", "zz")], truncation=False)[0].score == -0.5
    # cache miss 는 strip 점수에서도 -1.0 으로 숨기지 않는다
    with pytest.raises(KeyError):
        OfficialRetrievalEvaluator(b).score_strips("q", ["one two three four five"])


def test_weights_resolution_and_check(tmp_path, monkeypatch):
    monkeypatch.delenv("CRAG_EVALUATOR_PATH", raising=False)
    with pytest.raises(EvaluatorWeightsError):
        resolve_model_path({})
    monkeypatch.setenv("CRAG_EVALUATOR_PATH", str(tmp_path))
    assert resolve_model_path({}) == str(tmp_path)
    other = tmp_path / "other"
    assert resolve_model_path({"model_path": str(other)}) == str(other)  # config 가 우선
    with pytest.raises(EvaluatorWeightsError, match="누락"):
        check_weights_dir(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "t5", "architectures": ["T5ForSequenceClassification"]})
    )
    (tmp_path / "model.safetensors").write_bytes(b"0")
    (tmp_path / "spiece.model").write_bytes(b"0")
    rep = check_weights_dir(tmp_path)
    assert rep["model_type"] == "t5" and "warning" not in rep
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))
    with pytest.raises(EvaluatorWeightsError, match="t5"):
        check_weights_dir(tmp_path)


# ---------------------------------------------------------------- refinement
def test_extract_strips_modes():
    assert extract_strips("whole passage", "selection") == ["whole passage"]
    words = " ".join(f"w{i}" for i in range(105))
    assert [len(x.split()) for x in extract_strips(words, "fixed_num")] == [
        50,
        55,
    ]  # 10 미만은 앞에 붙임
    assert [len(s.split()) for s in extract_strips(" ".join(["x"] * 104), "fixed_num")] == [50, 54]
    assert [len(s.split()) for s in extract_strips(" ".join(["x"] * 115), "fixed_num")] == [
        50,
        50,
        15,
    ]
    assert extract_strips("tiny one", "fixed_num") == ["tiny one"]  # 원본 IndexError -> deviation
    s = "A is one two three four five. B short. C is one two three four five? D is one two three four"  # noqa: E501
    ex = extract_strips(s, "excerption")
    # '. ' / '?' 로 자르고, 5 단어 이하 조각은 앞 조각에 붙이고, 3개씩 묶는다
    assert ex == [
        "A is one two three four fiveB short C is one two three four five  D is one two three four"
    ]


def test_internal_knowledge_selection_top3_no_threshold():
    psgs = [
        "alpha doc text here",
        "beta doc text here",
        "gamma doc text here",
        "delta doc text here",
    ]
    scores = {"alpha": -3.0, "beta": 0.1, "gamma": 2.0, "delta": 0.1}
    e = ev(lambda q, p: scores[p.split()[0]])
    r = internal_knowledge("q", psgs, e, "selection", origins=["a", "b", "c", "d"])
    # 점수순 상위 3, 동점은 원래 순서 (안정 정렬). threshold 로 거르지 않음
    assert [s.origin for s in r.selected] == ["c", "b", "d"]
    assert r.text == "gamma doc text here; beta doc text here; delta doc text here"
    assert r.ranking == [2, 1, 3, 0]
    assert source.INTERNAL_TOP_N["excerption"] == 6


def test_official_passages_keep_spaces():
    line = retrieved_psgs_line(["p one", "", "p three"])
    assert line == "p one [sep]  [sep] p three"
    assert official_passages_from_line(line) == ["p one ", "  ", " p three"]


def test_external_knowledge_paragraphs_and_fallback():
    docs = [
        Document("u1", "short\nthis is a long paragraph one", {"title": "T1"}),
        Document("u2", "another long paragraph two here", {"title": ""}),
    ]
    e = ev(lambda q, p: float(len(p)))
    r = external_knowledge("q", docs, e, top_n=5)
    assert [s.text for s in r.selected] == [
        "T1: this is a long paragraph one",
        "another long paragraph two here",
    ]
    snip = external_knowledge("q", [Document("u", "tiny")], e)
    assert snip.text == "tiny" and snip.info["fallback"] == "snippets"
    assert external_knowledge("q", [], e).text == ""
    assert ambiguous_knowledge("A", "B") == "Knowledge1: A [sep] Knowledge2: B"


# ---------------------------------------------------------------- data / generation
def test_load_official_queries(tmp_path):
    lines = []
    for q in ("Q1?", "Q2?"):
        lines += [f"{q} [SEP] doc {i}" for i in range(9)] + [f"{q} [SEP]"]
    (tmp_path / "t.txt").write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "s").write_text("Q1?\nQ2?", encoding="utf-8")
    qs = load_official_queries(tmp_path / "t.txt", 10, tmp_path / "s")
    assert len(qs) == 2 and qs[1].passages[9] == "" and qs[0].passages[0] == "doc 0"
    assert parse_pair_line("Q [SEP] a [SEP] b".split(" [SEP] b")[0]) == ("Q", "a")
    ans = tmp_path / "a.jsonl"
    ans.write_text(
        json.dumps({"question": "Q1?", "answers": ["x"], "id": 5})
        + "\n"
        + json.dumps({"question": "Q2?", "possible_answers": '["y", "z"]'}),
        encoding="utf-8",
    )
    attach_answers(qs, ans)
    assert qs[0].answers == ["x"] and qs[0].ext_id == "5" and qs[1].answers == ["y", "z"]
    ans.write_text(
        json.dumps({"question": "Q2?", "answers": []})
        + "\n"
        + json.dumps({"question": "Q1?", "answers": []})
    )
    with pytest.raises(ValueError, match="불일치"):
        attach_answers(qs, ans)
    (tmp_path / "r").write_text("#k1 \n#k2", encoding="utf-8")
    assert load_ref(tmp_path / "r") == ["#k1", "#k2"]


def test_generation_prompt_and_metrics():
    p = offgen.format_popqa_prompt("Who {x}?", "doc {with} braces")
    assert p == (
        "Refer to the following documents, follow the instruction and answer the question."
        "\n\nDocuments: doc {with} braces\n\nInstruction: Answer the question: Who {x}?"
    )
    assert offgen.postprocess_answer("[Relevant]Paris\n</s>") == "Paris"
    assert offgen.official_match("He was a Politician", ["politician"]) == 0  # 대소문자 구분
    assert offgen.official_match("politician.", ["politician"]) == 1
    assert offgen.rag_paragraph(["a", "b", "c"], 2) == "a [sep] b"


def test_keyword_prompt_inserts_question():
    """공식 utils.py 는 '{{question}}' 때문에 질문이 빠지는 결함이 있다. 재구현은 질문을 넣는다."""
    assert "{{" not in source.POPQA_KEYWORD_PROMPT
    out = source.POPQA_KEYWORD_PROMPT.replace("{question}", "Who is X?")
    assert out.endswith("question: Who is X?\nquery: ")


def test_completion_engine_payload():
    from rag.crag_official.completion_engine import VLLMCompletionEngine
    from rag.llm.engine import LLMRequest

    class T:
        def __init__(self):
            self.sent = None

        def post_json(self, url, payload, headers):
            self.sent = (url, payload)
            return {
                "choices": [{"text": " Paris\n", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }, {}

    t = T()
    e = VLLMCompletionEngine("m", stream=False, transport=t, top_p=1.0, max_tokens=100)
    r = e.complete(LLMRequest("raw prompt"))
    url, payload = t.sent
    assert url.endswith("/v1/completions") and payload["prompt"] == "raw prompt"
    assert "messages" not in payload and payload["top_p"] == 1.0 and payload["max_tokens"] == 100
    assert r.text == " Paris\n" and r.prompt_tokens == 3  # strip 하지 않음 (공식 후처리에 맡김)


def test_pairscore_defaults():
    assert PairScore(1.0).source == "model"
