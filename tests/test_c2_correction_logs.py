"""C2 지원: 문서별 grading 로그, experiment manifest, correction 검색 캐시 (v2 / query_strict),
Correction 발생 질문 분석 + grading 중단 재구성.

가짜 엔진·mock evaluator 로 동작만 검증한다 (정확도 결과 아님).
"""

import csv
import json
import sys
from pathlib import Path

import pytest

from rag.crag_official.evaluator import (
    CachedScoreBackend,
    MockScoreBackend,
    OfficialRetrievalEvaluator,
    pair_key,
)
from rag.evidence.budget import build_budget
from rag.evidence.doc_log import FIELDS, LLAMA_LOGPROB_STATUS, doc_rows
from rag.evidence.frozen import FrozenItem, file_sha256, manifest_path_for
from rag.evidence.graders import LlamaPromptGrader, OfficialT5Grader
from rag.evidence.manifest import compare_manifests
from rag.evidence.pipeline import ControlledCRAG, PipelineConfig
from rag.evidence.policies import get_policy
from rag.retrieval.tavily_search import TavilyWebSearch, tavily_params
from rag.retrieval.web_cache import (
    CACHE_VERSION,
    RecordingWebSearch,
    ReplayWebSearch,
    read_jsonl,
)
from rag.types import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "official_crag"))
sys.path.insert(0, str(ROOT / "tests"))
import analyze_correction  # noqa: E402
import prefetch_web  # noqa: E402
from mode_b import Components, ModeBRunner, build_web_search  # noqa: E402
from test_evidence_pipeline import FakeEngine, _cfg, level, llm  # noqa: E402


def mock_score(q, p):
    return {"GOOD": 1.0, "MID": 0.0, "BAD": -2.0}[level(p)]


def t5():
    return OfficialRetrievalEvaluator(MockScoreBackend(mock_score))


# 질문 설계 (mock T5: GOOD 1.0 / MID 0.0 / BAD -2.0, popqa threshold 0.592 / -0.995)
#   c1 correct (rank0)          vanilla O, official O
#   c2 incorrect, web 에 정답    vanilla X, official O  -> improved
#   c3 incorrect, BAD 문서에 정답, web 에 없음  vanilla O, official X -> degraded
#   c4 ambiguous, web 에 없음    vanilla X, official X  -> both wrong
#   c5 correct (rank1)          조기 중단 max_docs:1 이면 판정 incorrect 로 바뀜
#   c6 correct (rank2)          max_docs:1,2 면 ambiguous
QUESTIONS = {
    "c1": [
        "GOOD Paris is the capital city here",
        "BAD filler text one two",
        "BAD filler three four",
    ],
    "c2": ["BAD filler text one two", "BAD another filler text here", "BAD more filler words now"],
    "c3": ["BAD but mentions Paris in passing", "BAD filler text one two", "BAD more filler words"],
    "c4": ["MID partially relevant text words", "BAD filler text here now", "BAD x y z w"],
    "c5": ["BAD filler words a b", "GOOD Paris capital text found", "BAD q r s t"],
    "c6": ["MID text text text text", "MID other words words here", "GOOD Paris text a b c"],
}
WEB = {
    "c2": [Document("w2", "web page says Paris is the answer")],
    "c3": [Document("w3", "web page without the answer string")],
    "c4": [Document("w4", "web page without the answer string")],
}


def make_frozen(tmp_path: Path) -> Path:
    path = tmp_path / "inputs" / "frozen.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, texts in QUESTIONS.items():
            docs = [Document(f"{qid}d{i}", t, score=1.0 - i / 10) for i, t in enumerate(texts)]
            item = FrozenItem(qid, f"What is the capital ({qid})?", ["Paris"], docs)
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    manifest_path_for(path).write_text(
        json.dumps(
            {
                "sha256": file_sha256(path),
                "n": len(QUESTIONS),
                "top_k": 3,
                "dataset": {"path": "qa.jsonl", "sha256": "d" * 64},
                "caveat": "공통 문항 집합 (일반화 결과 아님)",
            }
        ),
        encoding="utf-8",
    )
    return path


class QWeb:
    """질문별로 다른 web 결과 (runner 가 bind(question_id))."""

    def bind(self, qid):
        return _Bound(WEB.get(qid, []))


class _Bound:
    def __init__(self, docs):
        self.docs = docs
        self.last = {}

    def search(self, query, top_k):
        return self.docs[:top_k]


def run_mode_b(tmp_path, run_id, policy, grader="official_t5", budget=None, **over):
    cfg = _cfg(
        tmp_path,
        make_frozen(tmp_path)
        if not (tmp_path / "inputs" / "frozen.jsonl").exists()
        else tmp_path / "inputs" / "frozen.jsonl",
        evidence_policy=policy,
        grader={"type": grader, "upper": 0.7, "lower": 0.3},
        budget=budget or {"preset": "full_grading"},
    )
    cfg["run"]["warmup_queries"] = 0
    for k, v in over.items():
        cfg[k] = v
    comp = Components(engine=FakeEngine(), web_search=QWeb(), evaluator=t5())
    return ModeBRunner(cfg, components=comp, run_id=run_id).run()


def jl(path):
    return read_jsonl(path)


# ---------------------------------------------------------------- 1. 문서별 grading 로그
def _row_and_docs(policy, grader, pattern_texts, budget=None, evaluator=None):
    g = OfficialT5Grader(evaluator or t5()) if grader == "official_t5" else LlamaPromptGrader()
    p = ControlledCRAG(
        get_policy(policy), g, build_budget(budget), QWeb().bind("c2"), PipelineConfig()
    )
    docs = [Document(f"d{i}", t, score=0.5 - i / 10) for i, t in enumerate(pattern_texts)]
    item = FrozenItem("q1", "What is the capital?", ["Paris"], docs)
    row = p.run(item, llm())
    row.update(run_id="r", request_id="r:0:q1", question_id="q1", warmup=False)
    return row, docs


def test_doc_log_official_t5_fields():
    texts = ["MID partially relevant words", "", "GOOD Paris capital words here"]
    row, docs = _row_and_docs("official", "official_t5", texts)
    rows = doc_rows(row, docs, doc_tokens=lambda t: len(t.split()) + 1, tokenizer_name="ws")
    assert [r["doc_rank"] for r in rows] == [0, 1, 2]
    assert all(set(r) == set(FIELDS) for r in rows)
    r0, r1, r2 = rows
    assert r0["grader_type"] == "official_t5" and r0["score_scale"] == "t5_regression_logit"
    assert r0["grader_score"] == 0.0 and r0["t5_raw_logit"] == 0.0
    assert r0["grader_pass"] is False and r0["grader_decision"] == "partial"
    assert r2["grader_pass"] is True and r2["grader_flag"] == 2
    # 빈 문서: 공식 규칙 -1.0, 모델 raw 점수 없음, 시간 종류 구분
    assert r1["grader_score"] == -1.0 and r1["t5_raw_logit"] is None
    assert r1["timing_kind"] == "rule_no_model" and r1["grading_input_tokens"] == 0
    # 회귀 출력: 생성 토큰 없음 (0 이 아니라 null), logprob 은 T5 에 해당 없음
    assert r0["grading_output_tokens"] is None
    assert r0["grading_output_kind"] == "regression_scalar_no_generated_tokens"
    assert r0["llama_yes_logprob"] is None and r0["llama_yes_logprob_status"] == "not_applicable"
    assert r0["document_tokens"] == 5 and r0["retrieval_score"] == 0.5
    # grading_events / snapshot 과 연결
    assert r2["eval_order"] == 2 and r2["snapshot_index"] == 2
    assert r2["provisional_action_after"] == "correct" and r2["action_final_after"] is True
    assert row["grading_snapshots"][r0["snapshot_index"]]["evaluated_ids"] == ["d0"]
    assert r0["grading_latency_ms"] >= 0 and r0["timing_kind"] == "mock"


def test_doc_log_llama_logprob_is_null_not_score():
    texts = ["MID text words here", "GOOD Paris text words", "BAD filler words here"]
    row, docs = _row_and_docs("crag_rule", "llama_prompt", texts)
    rows = doc_rows(row, docs, doc_tokens=lambda t: 99, tokenizer_name="x")
    r = rows[1]
    assert r["grader_type"] == "legacy_llama" and r["grader_impl"] == "llama_prompt"
    assert r["llama_parsed_score"] == 0.9 and r["grader_score"] == 0.9
    assert r["llama_yes_logprob"] is None and r["llama_yes_logprob_status"] == LLAMA_LOGPROB_STATUS
    assert r["t5_raw_logit"] is None
    assert r["grading_output_tokens"] == 1 and r["grading_input_tokens"] > 0
    assert r["timing_kind"] == "live_llm" and r["is_cache_replay"] is False
    assert r["document_tokens"] is None  # Llama 토크나이저는 로컬에 없음 -> 세지 않음


def test_doc_log_unevaluated_docs_have_no_values():
    texts = ["GOOD Paris text words", "MID text words here", "BAD filler words here"]
    row, docs = _row_and_docs("crag_rule", "official_t5", texts, budget={"preset": "lazy_hit"})
    rows = doc_rows(row, docs)
    assert [r["evaluated"] for r in rows] == [True, False, False]
    for r in rows[1:]:
        for k in (
            "grader_score",
            "grader_pass",
            "grader_flag",
            "grading_latency_ms",
            "grading_input_tokens",
            "t5_raw_logit",
            "eval_order",
            "timing_kind",
        ):
            assert r[k] is None, k
        assert r["llama_yes_logprob_status"] == "not_applicable"


def test_doc_log_cache_replay_is_marked():
    q, texts = "What is the capital?", ["GOOD Paris text words", "BAD filler words here"]
    cache = [
        {
            "key": pair_key(q, t, False),
            "score": mock_score(q, t),
            "n_tokens": 7,
            "compute_seconds": 0.25,
        }
        for t in texts
    ]
    ev = OfficialRetrievalEvaluator(CachedScoreBackend([]))
    for c in cache:
        ev.backend.cache[c["key"]] = c
    row, docs = _row_and_docs("crag_rule", "official_t5", texts, evaluator=ev)
    r = doc_rows(row, docs)[0]
    assert r["is_cache_replay"] is True and r["timing_kind"] == "cache_replay"
    assert r["grading_compute_ms"] == 250.0  # 기록 당시 계산 시간 (이번 wall 과 구분)
    assert r["grading_input_tokens"] == 7
    assert r["grading_input_tokens_kind"] == "t5_attention_mask_sum_recorded"


def test_doc_log_skips_vanilla_and_error_rows():
    assert doc_rows({"grader_type": None}, []) == []
    assert doc_rows({"error": "x", "grader_type": "official_t5"}, []) == []


# ---------------------------------------------------------------- 2. manifest + runner 출력
def test_runner_writes_doc_log_and_manifest(tmp_path):
    off = run_mode_b(tmp_path, "off", "official")
    van = run_mode_b(tmp_path, "van", "vanilla")
    docs = jl(off / "grading_docs.jsonl")
    assert len(docs) == sum(len(v) for v in QUESTIONS.values())
    with open(off / "grading_docs.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == len(docs)
    assert not (van / "grading_docs.jsonl").exists() or not jl(van / "grading_docs.jsonl")
    summary = json.loads((off / "summary.json").read_text(encoding="utf-8"))
    assert summary["grading_docs"]["n_evaluated"] == len(docs)

    m = json.loads((off / "manifest.json").read_text(encoding="utf-8"))
    mv = json.loads((van / "manifest.json").read_text(encoding="utf-8"))
    assert m["experiment_mode"] == "B_controlled" and "A_official" in m["not_comparable_with"]
    assert m["question_ids"] == list(QUESTIONS)
    assert m["dataset"]["sha256"] == "d" * 64
    assert m["retrieval_inputs"]["sha256"] == file_sha256(tmp_path / "inputs" / "frozen.jsonl")
    assert m["prompts"]["used"]["generate"]["name"] == "legacy_generate"
    assert len(m["prompts"]["used"]["generate"]["sha256"]) == 64
    assert m["generator"]["temperature"] is None or "temperature" in m["generator"]
    assert (
        m["grader"]["upper"] == pytest.approx(0.592) and "any pass" in m["grader"]["decision_rule"]
    )
    assert m["evidence_policy"] == "official" and mv["evidence_policy"] == "vanilla"
    assert m["evaluation"]["files"]["src/rag/eval/metrics.py"]
    assert "commit" in m["git"] and m["analysis_scripts"]["mode_b.py"]
    assert m["subset_caveat"]
    # 정책만 다르면 비교 조건은 같다
    assert m["comparison_sha256"] == mv["comparison_sha256"]
    assert compare_manifests(m, mv) == []
    assert compare_manifests(m, {**mv, "comparison": {**mv["comparison"], "experiment_mode": "A"}})


# ---------------------------------------------------------------- 3. 검색 캐시
class FakeTavilyClient:
    def __init__(self):
        self.calls = []

    def search(self, query, **kw):
        self.calls.append((query, kw))
        return {
            "query": query,
            "results": [
                {
                    "url": f"https://x/{query}",
                    "content": "snippet Paris",
                    "title": "T",
                    "score": 0.5,
                    "raw_content": "full page",
                }
            ],
            "api_key": "sk-should-not-be-stored",
        }


def test_tavily_search_raw_keeps_existing_call_args():
    c = FakeTavilyClient()
    ws = TavilyWebSearch(client=c)
    docs = ws.search("q", 3)
    assert docs[0].text == "snippet Paris"
    assert c.calls[-1][1] == {"max_results": 3, "search_depth": "basic"}  # 기존 인자 그대로
    TavilyWebSearch(client=c, include_raw_content=True).search_raw("q", 3)
    assert c.calls[-1][1]["include_raw_content"] is True
    assert ws.params() == tavily_params("basic", False)


def test_recording_v2_fields_and_no_secrets(tmp_path):
    path = tmp_path / "web_cache.jsonl"
    rec = RecordingWebSearch(
        TavilyWebSearch(client=FakeTavilyClient()),
        path,
        "tavily",
        {"search_depth": "basic", "api_key": "sk-x"},
        "legacy_rewrite",
    )
    b = rec.bind("q7")
    b.search(" kw query ", 5)
    row = jl(path)[0]
    assert row["cache_version"] == CACHE_VERSION and row["query_id"] == "q7"
    assert row["query"] == "kw query" and row["provider"] == "tavily"
    assert row["params"] == {"search_depth": "basic", "api_key": "<redacted>", "top_k": 5}
    assert row["query_mode"] == "legacy_rewrite" and row["searched_at_utc"]
    assert row["raw_response"]["results"][0]["raw_content"] == "full page"
    assert "sk-" not in path.read_text(encoding="utf-8")
    assert b.last["source"] == "live" and b.last["recorded"] is True


def _strict(paths, query_mode="legacy_rewrite", **kw):
    return ReplayWebSearch(
        paths,
        "zero",
        key_mode="query_strict",
        expect={"provider": "tavily", "params": tavily_params(), "query_mode": query_mode},
        **kw,
    )


def test_replay_strict_does_not_conflate_queries(tmp_path):
    path = tmp_path / "web_cache.jsonl"
    rec = RecordingWebSearch(
        TavilyWebSearch(client=FakeTavilyClient()),
        path,
        "tavily",
        tavily_params(),
        "legacy_rewrite",
    )
    rec.bind("q1").search("alpha", 5)
    rec.bind("q2").search("beta", 5)
    r = _strict([path])
    b = r.bind("q1")
    assert b.search("alpha", 5)[0].id == "https://x/alpha"
    assert b.last["compatibility"] == "exact" and b.last["hit"] is True
    with pytest.raises(KeyError):  # 같은 query_id, 다른 검색어 -> 재사용하지 않음
        b.search("alpha rewritten differently", 5)
    assert b.last["miss_reason"] == "query_mismatch" and b.last["recorded_queries"] == ["alpha"]
    with pytest.raises(KeyError):  # 다른 질문의 검색어를 빌려오지 않음
        r.bind("q1").search("beta", 5)
    with pytest.raises(KeyError):  # top_k 가 다르면 다른 검색
        r.bind("q1").search("alpha", 3)
    # 검색 조건 (query_mode) 이 다르면 호환되지 않음
    r2 = _strict([path], query_mode="official_keywords")
    with pytest.raises(KeyError):
        r2.bind("q1").search("alpha", 5)
    assert r2.stats()["n_incompatible_records"] == 2
    r3 = _strict([path], on_miss="empty")
    assert r3.bind("q1").search("zzz", 5) == []


def test_replay_strict_legacy_records(tmp_path):
    path = tmp_path / "v1.jsonl"
    path.write_text(
        json.dumps(
            {"query": "alpha", "top_k": 5, "latency": 0.1, "docs": [{"id": "u", "text": "t"}]}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError):  # 기본: v1 (조건 정보 없음) 사용 안 함
        _strict([path]).bind("q1").search("alpha", 5)
    r = _strict([path], accept_legacy=True)
    b = r.bind("q1")
    assert b.search("alpha", 5)[0].id == "u" and b.last["compatibility"] == "legacy_assumed"
    with pytest.raises(KeyError):  # 검색어가 다르면 v1 도 miss
        b.search("beta", 5)
    with pytest.raises(KeyError):  # 공식 keyword 방식에는 v1 (legacy rewrite) 사용 불가
        _strict([path], query_mode="official_keywords", accept_legacy=True).bind("q1").search(
            "alpha", 5
        )


def test_replay_live_fallback_records_query_id(tmp_path):
    path = tmp_path / "web_cache.jsonl"
    path.write_text("", encoding="utf-8")
    out = tmp_path / "new.jsonl"
    fb = RecordingWebSearch(
        TavilyWebSearch(client=FakeTavilyClient()),
        out,
        "tavily",
        tavily_params(),
        "legacy_rewrite",
    )
    r = ReplayWebSearch(
        [path],
        "zero",
        fallback=fb,
        key_mode="query_strict",
        expect={"provider": "tavily", "params": tavily_params(), "query_mode": "legacy_rewrite"},
    )
    r.bind("q9").search("gamma", 5)
    assert jl(out)[0]["query_id"] == "q9"
    assert _strict([out]).bind("q9").search("gamma", 5)


def test_prefetch_then_strict_replay_in_mode_b(tmp_path):
    frozen = make_frozen(tmp_path)
    cfg = _cfg(tmp_path, frozen, evidence_policy="official")
    cfg["run"]["warmup_queries"] = 0
    cache_dir = prefetch_web.prefetch(
        cfg,
        "shared",
        engine=FakeEngine(),
        web=TavilyWebSearch(client=FakeTavilyClient()),
        output_dir=tmp_path / "results",
    )
    rows = jl(cache_dir / "web_cache.jsonl")
    assert [r["query_id"] for r in rows] == list(QUESTIONS)
    assert {r["query"] for r in rows} == {"rewritten"}  # FakeEngine rewrite
    man = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["n_recorded"] == 6 and man["correction"]["query_mode"] == "legacy_rewrite"
    assert man["inputs"]["sha256"] == file_sha256(frozen)

    cfg["web_search"] = {
        "provider": "replay",
        "replay_key": "query_strict",
        "latency_mode": "zero",
        "replay_paths": [str(cache_dir / "web_cache.jsonl")],
        "top_k": 5,
    }
    rd = tmp_path / "wsd"
    rd.mkdir()
    ws = build_web_search(cfg, rd)
    comp = Components(engine=FakeEngine(), web_search=ws, evaluator=t5())
    out = ModeBRunner(cfg, components=comp, run_id="strict").run()
    qrows = jl(out / "queries.jsonl")
    assert all("error" not in r for r in qrows)
    perf = [r for r in qrows if r["correction"]["performed"]]
    assert perf and all(r["correction"]["replay"]["compatibility"] == "exact" for r in perf)
    assert ws.stats()["misses"] == 0
    # official_keywords 로 바꾸면 같은 캐시를 쓰지 않는다 (검색 조건 다름)
    cfg2 = {**cfg, "correction": {**cfg["correction"], "query_mode": "official_keywords"}}
    ws2 = build_web_search(cfg2, rd)
    with pytest.raises(KeyError):
        ws2.bind("c2").search("rewritten", 5)


# ---------------------------------------------------------------- 4/5. Correction 분석
@pytest.fixture
def runs(tmp_path):
    return {
        "official": run_mode_b(tmp_path, "off", "official"),
        "vanilla": run_mode_b(tmp_path, "van", "vanilla"),
        "keep_all": run_mode_b(tmp_path, "keep", "keep_all"),
        "legacy": run_mode_b(tmp_path, "leg", "crag_rule", grader="llama_prompt"),
    }


def _analyze(tmp_path, monkeypatch, runs, name, *extra):
    monkeypatch.setattr(analyze_correction, "ROOT", tmp_path)
    argv = [
        "--official",
        str(runs["official"]),
        "--vanilla",
        str(runs["vanilla"]),
        "--name",
        name,
        *extra,
    ]
    return analyze_correction.main(argv)


def test_correction_analysis(tmp_path, monkeypatch, runs):
    out = _analyze(
        tmp_path,
        monkeypatch,
        runs,
        "c2",
        "--keep-all",
        str(runs["keep_all"]),
        "--legacy",
        str(runs["legacy"]),
    )
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["mock_evaluator"] is True and s["excluded_runs"] == {}
    assert s["n_total"] == 6 and s["n_correction"] == 3 and s["n_correction_strict"] == 2
    assert s["path_counts"] == {"correct": 3, "ambiguous": 1, "incorrect": 2}
    p = s["metrics"]["acc"]["correction"]["official_vs_vanilla"]
    assert (p["improved"], p["degraded"], p["both_correct"], p["both_wrong"]) == (1, 1, 0, 1)
    assert p["base_acc"] == pytest.approx(1 / 3) and p["compare_acc"] == pytest.approx(1 / 3)
    assert s["c2_categories_acc"] == {
        "correction_improved": 1,
        "correction_no_change_correct": 0,
        "correction_no_change_wrong": 1,
        "correction_degraded": 1,
    }
    # keep_all 은 내부 문서를 유지하므로 c3 (BAD 문서에 정답) 를 지킨다
    k = s["metrics"]["acc"]["correction"]["keep_all_vs_vanilla"]
    assert k["degraded"] == 0 and k["improved"] == 1
    assert s["path_crosstab_official_vs_legacy"]["correct->correct"] == 3

    pq = {
        r["question_id"]: r
        for r in csv.DictReader(open(out / "per_question.csv", encoding="utf-8"))
    }
    assert pq["c2"]["c2_category"] == "correction_improved"
    assert pq["c3"]["c2_category"] == "correction_degraded"
    assert pq["c1"]["c2_category"] == "no_correction"
    assert pq["c2"]["correction_query"] == "rewritten" and pq["c2"]["web_valid"] == "True"

    pb = list(csv.DictReader(open(out / "path_breakdown.csv", encoding="utf-8")))
    row = next(r for r in pb if r["t5_path"] == "incorrect" and r["policy"] == "official")
    assert row["n"] == "2" and float(row["web_search_rate"]) == 1.0

    es = {
        (r["run"], r["rule"]): r
        for r in csv.DictReader(open(out / "early_stop_summary.csv", encoding="utf-8"))
    }
    # 공식 판정: 첫 통과 문서에서 멈추면 판정은 절대 바뀌지 않는다
    assert es[("official", "first_pass")]["n_decision_changed"] == "0"
    assert es[("official", "max_docs:1")]["n_decision_changed"] == "2"  # c5, c6
    assert es[("official", "max_docs:1")]["n_evidence_changed"] == "2"
    assert es[("official", "max_docs:1")]["potential_loss_upper_bound"] == "2"
    assert es[("official", "max_docs:2")]["n_decision_changed"] == "1"  # c6
    assert float(es[("official", "max_docs:1")]["saved_grading_latency_frac"]) >= 0
    # crag_rule: 판정이 같아도 남길 문서가 바뀌면 evidence 변화 (c1 rank0 GOOD 은 같음)
    assert ("legacy", "first_pass") in es
    cands = list(csv.DictReader(open(out / "verification_candidates.csv", encoding="utf-8")))
    assert {c["question_id"] for c in cands if c["run"] == "official"} == {"c5", "c6"}
    assert all("미구현" in c["how_to_verify"] for c in cands if c["run"] == "official")
    rep = (out / "report.md").read_text(encoding="utf-8")
    assert "MOCK" in rep and "PopQA" in rep and "미확정" in rep


def test_correction_analysis_question_ids_and_checks(tmp_path, monkeypatch, runs):
    ids = tmp_path / "ids.txt"
    ids.write_text("c2\nc3\nc1\n", encoding="utf-8")
    out = _analyze(tmp_path, monkeypatch, runs, "sub", "--question-ids", str(ids))
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["n_total"] == 3 and s["n_correction"] == 2
    bad = tmp_path / "bad.txt"
    bad.write_text("c1\nzz\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="zz"):
        _analyze(tmp_path, monkeypatch, runs, "bad", "--question-ids", str(bad))
    # official 자리에 다른 정책 -> 거부
    with pytest.raises(SystemExit, match="official"):
        monkeypatch.setattr(analyze_correction, "ROOT", tmp_path)
        analyze_correction.main(
            ["--official", str(runs["keep_all"]), "--vanilla", str(runs["vanilla"]), "--name", "x"]
        )
    with pytest.raises(FileExistsError):
        _analyze(tmp_path, monkeypatch, runs, "sub")


def test_correction_analysis_rejects_different_generator(tmp_path, monkeypatch, runs):
    other = run_mode_b(
        tmp_path,
        "van_t05",
        "vanilla",
        llm={"provider": "fake", "model": "fake", "temperature": 0.5},
    )
    with pytest.raises(SystemExit, match="temperature"):
        monkeypatch.setattr(analyze_correction, "ROOT", tmp_path)
        analyze_correction.main(
            ["--official", str(runs["official"]), "--vanilla", str(other), "--name", "mm"]
        )
    # 비교 대상 정책 run 이 조건이 다르면 제외 (분석은 계속)
    keep = run_mode_b(
        tmp_path,
        "keep_t05",
        "keep_all",
        llm={"provider": "fake", "model": "fake", "temperature": 0.5},
    )
    out = _analyze(tmp_path, monkeypatch, runs, "ex", "--keep-all", str(keep))
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (
        s["excluded_runs"]["keep_all"].startswith("조건 다름")
        and "temperature" in (s["excluded_runs"]["keep_all"])
    )
    assert "keep_all_vs_vanilla" not in s["metrics"]["acc"]["correction"]
