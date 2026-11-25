"""rag.evidence (C1 정책 / C2 상태 / C3 budget) + MODE B runner 테스트.

가짜 엔진과 mock evaluator 로 동작만 검증한다 (정확도·성능 결과 아님).
기존 구현과의 동등성: crag_rule / no_correct_filter / no_incorrect_discard 가 기존 CRAGGraph /
AblationCRAGGraph 와 같은 context 를, lazy_hit / lazy_rest 가 기존 LazyGrading 과 같은 context 를
만드는지.
"""

import json
import sys
import threading
from pathlib import Path

import pytest

from rag.crag_official.evaluator import MockScoreBackend, OfficialRetrievalEvaluator
from rag.evidence.budget import build_budget
from rag.evidence.frozen import (
    FrozenItem,
    build_frozen_items,
    file_sha256,
    load_frozen,
    manifest_path_for,
)
from rag.evidence.graders import LlamaPromptGrader, OfficialT5Grader
from rag.evidence.pipeline import ControlledCRAG, PipelineConfig, WebSearchFailed
from rag.evidence.policies import POLICIES, get_policy
from rag.evidence.schema import validate_record
from rag.grading.strategies import FullGrading, LazyGrading
from rag.graph import CRAGGraph
from rag.llm.engine import LLMResponse
from rag.llm.engine_llm import EngineLLM
from rag.telemetry.calls import CallRecorder, infer_call_kind
from rag.types import Document, QAExample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "crag_system"))
sys.path.insert(0, str(ROOT / "experiments" / "official_crag"))
from ablation import AblationCRAGGraph  # noqa: E402
from mode_b import Components, ModeBRunner  # noqa: E402


def level(text: str) -> str:
    return "GOOD" if "GOOD" in text else "MID" if "MID" in text else "BAD"


class FakeEngine:
    name, model = "fake", "fake-model"

    def __init__(self):
        self.lock = threading.Lock()
        self.prompts = []

    def complete(self, request):
        kind = request.kind if request.kind != "other" else infer_call_kind(request.prompt)
        with self.lock:
            self.prompts.append((kind, request.prompt))
        doc = request.prompt.split("Document:")[-1]
        if kind == "grade":
            text = {"GOOD": "0.9", "MID": "0.5", "BAD": "0.1"}[level(doc)]
        elif kind == "rewrite":
            text = "rewritten"
        else:
            text = "Paris" if "Paris" in request.prompt else "unknown"
        return LLMResponse(
            text,
            "fake",
            "fake-model",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=1,
        )

    def describe(self):
        return {"engine": "fake"}


class FakeWeb:
    def __init__(self, docs=None, fail=None):
        self.docs = (
            docs
            if docs is not None
            else [
                Document("w1", "web says Paris here\nsecond para line"),
                Document("w2", "other web text body"),
            ]
        )
        self.fail = fail
        self.queries = []

    def search(self, query, top_k):
        self.queries.append(query)
        if self.fail:
            raise self.fail
        return self.docs[:top_k]


def t5_eval():
    return OfficialRetrievalEvaluator(
        MockScoreBackend(lambda q, p: {"GOOD": 1.0, "MID": 0.0, "BAD": -2.0}[level(p)])
    )


def docs_for(pattern: str) -> list[Document]:
    names = {"G": "GOOD", "M": "MID", "B": "BAD"}
    return [
        Document(
            f"d{i}",
            f"{names[c]} doc number {i} text Paris"
            if c == "G"
            else f"{names[c]} doc number {i} text",
        )
        for i, c in enumerate(pattern)
    ]


def llm():
    return EngineLLM(FakeEngine(), recorder=CallRecorder())


def run_policy(policy, pattern, grader="official_t5", budget=None, web=None, **pcfg):
    grader_obj = OfficialT5Grader(t5_eval()) if grader == "official_t5" else LlamaPromptGrader()
    p = ControlledCRAG(
        get_policy(policy),
        grader_obj,
        build_budget(budget),
        web or FakeWeb(),
        PipelineConfig(**pcfg),
    )
    item = FrozenItem("q1", "Capital of France?", ["Paris"], docs_for(pattern))
    return p.run(item, llm())


# ---------------------------------------------------------------- C1 정책
@pytest.mark.parametrize(
    "policy,pattern,action,kept,web",
    [
        ("crag_rule", "MGB", "correct", ["d1"], False),
        ("no_correct_filter", "MGB", "correct", ["d0", "d1", "d2"], False),
        ("keep_all", "MGB", "correct", ["d0", "d1", "d2"], False),
        ("crag_rule", "MBB", "ambiguous", ["d0"], True),
        ("keep_all", "MBB", "ambiguous", ["d0", "d1", "d2"], True),
        ("crag_rule", "BBB", "incorrect", [], True),
        ("no_incorrect_discard", "BBB", "incorrect", ["d0", "d1", "d2"], True),
        ("vanilla", "BBB", "none", ["d0", "d1", "d2"], False),
    ],
)
def test_policy_matrix(policy, pattern, action, kept, web):
    r = run_policy(policy, pattern)
    assert r["correction_path"] == action
    assert r["evidence"]["kept_retrieved_ids"] == kept
    assert r["correction"]["performed"] is web
    assert set(r["evidence"]["removed_retrieved_ids"]) == {f"d{i}" for i in range(3)} - set(kept)
    if web:
        assert r["evidence"]["added_ids"] == ["w1", "w2"]
    assert validate_record({**r, **_ids()}) == []


def _ids():
    return {
        k: None
        for k in (
            "schema",
            "run_id",
            "request_id",
            "request_index",
            "question_id",
            "question",
            "experiment_mode",
            "warmup",
            "generation_input_tokens",
            "generation_output_tokens",
            "llm_calls",
            "queue_wait",
        )
    }


def test_official_correct_refines_all_docs_without_threshold():
    r = run_policy("official", "BMGBB", prompt="official")
    assert r["correction_path"] == "correct"
    # selection: 문서 전부 점수순 상위 3 (GOOD 1.0, MID 0.0, BAD -2.0 중 첫 번째)
    assert r["evidence"]["kept_retrieved_ids"] == ["d2", "d1", "d0"]
    assert r["evidence"]["removed_retrieved_ids"] == ["d3", "d4"]
    assert r["refinement_events"][0]["ranking"][:3] == [2, 1, 0]
    assert r["stage_timings"]["refinement"] is not None
    assert r["stage_timings"]["rewrite"] is None  # 실행 안 한 stage 는 None
    assert r["official_policy_semantics"] is True


def test_official_ambiguous_and_incorrect_paths():
    amb = run_policy("official", "MBB", prompt="official", record_texts=True)
    assert amb["correction_path"] == "ambiguous"
    p = amb["generation_prompt"]
    assert "Documents: Knowledge1: " in p and " [sep] Knowledge2: " in p
    assert amb["refinement_events"][1]["target"] == "external"
    inc = run_policy("official", "BBB", prompt="official")
    assert inc["evidence"]["kept_retrieved_ids"] == []
    assert all(u["kind"] == "web_strip" for u in inc["evidence"]["units"])


def test_official_with_llama_is_marked_nonofficial_and_reuses_scores():
    r = run_policy("official", "MGB", grader="llama_prompt")
    assert r["official_policy_semantics"] is False and r["policy_id"] == "official@llama_prompt"
    assert r["grader_type"] == "llama_prompt" and r["thresholds"] == {"upper": 0.7, "lower": 0.3}


def test_crag_rule_and_ablations_equal_legacy_graph():
    """같은 Llama grader 응답에서 기존 CRAGGraph / AblationCRAGGraph 와 같은 context."""

    class R:
        def __init__(self, d):
            self.d = d

        def retrieve(self, q, k):
            return self.d[:k]

    cases = {
        "crag_rule": {},
        "no_correct_filter": {"correct_keep_topk": True},
        "no_incorrect_discard": {"incorrect_keep_faiss": True},
    }
    for pattern in ("MGB", "MBB", "BBB", "GGM"):
        docs = docs_for(pattern)
        for policy, abl in cases.items():
            web = FakeWeb()
            g = CRAGGraph(R(docs), None, web, grader=FullGrading(), top_k=3)
            if abl:
                g = AblationCRAGGraph.from_graph(g, abl)
            legacy = g.run(QAExample("q1", "Capital of France?", ["Paris"]), llm=llm())
            new = run_policy(policy, pattern, grader="llama_prompt")
            assert new["correction_path"] == legacy.corrective_action
            assert new["evidence"]["selected_evidence_ids"] == [c.id for c in legacy.contexts], (
                policy,
                pattern,
            )


# ---------------------------------------------------------------- C3 budget
@pytest.mark.parametrize("mode", ["hit", "rest"])
def test_lazy_presets_equal_legacy_lazy(mode):
    for pattern in ("MGBG", "BBMB", "GBBB"):
        docs = docs_for(pattern)
        legacy = LazyGrading(mode).grade("q", docs, llm(), 0.7, 0.3)
        r = run_policy(
            "crag_rule", pattern, grader="llama_prompt", budget={"preset": f"lazy_{mode}"}
        )
        assert len(r["grading_events"]) == legacy.n_graded
        if legacy.action == "correct":
            assert r["evidence"]["selected_evidence_ids"] == [d.id for d in legacy.contexts]
        assert r["correction_path"] == legacy.action


def test_budget_separates_stop_and_retention():
    b = build_budget({"stop_rule": "first_pass", "retention": "ratio", "ratio": 0.5})
    assert b.stop.name == "first_pass" and b.retention.name == "ratio:0.5:rest"
    r = run_policy(
        "crag_rule", "BGBBB", budget={"stop_rule": "first_pass", "retention": "ratio", "ratio": 0.5}
    )
    assert len(r["grading_events"]) == 2 and r["decision"]["stopped_early"]
    assert r["evidence"]["kept_retrieved_ids"] == ["d1", "d2"]  # 후보 d1..d4 의 앞 ceil(0.5x4)
    assert build_budget({"preset": "lazy_rest"}).name == "lazy_rest"
    assert build_budget(None).is_full_default
    with pytest.raises(ValueError):
        build_budget({"stop_rule": "sometimes"})


def test_official_policy_rejects_lazy():
    with pytest.raises(ValueError, match="official"):
        run_policy("official", "MGB", budget={"preset": "lazy_rest"})
    with pytest.raises(ValueError):
        run_policy("vanilla", "MGB", budget={"preset": "lazy_hit"})


# ---------------------------------------------------------------- C2 state
def test_snapshots_track_partial_state():
    r = run_policy("crag_rule", "MBGB", grader="llama_prompt")
    snaps = r["grading_snapshots"]
    assert [s["n_evaluated"] for s in snaps] == [1, 2, 3, 4]
    assert [s["provisional_action"] for s in snaps] == [
        "ambiguous",
        "ambiguous",
        "correct",
        "correct",
    ]
    assert [s["action_final"] for s in snaps] == [False, False, True, True]
    assert snaps[2]["passed_ids"] == ["d2"] and snaps[2]["first_pass_rank"] == 2
    # Llama: 토큰은 엔진 보고값 (가짜 엔진: 단어 수)
    assert snaps[-1]["cum_input_tokens"] == sum(e["input_tokens"] for e in r["grading_events"])
    ev = r["grading_events"][0]
    assert {"rank", "doc_id", "score", "flag", "label", "latency", "eval_order"} <= set(ev)


# ---------------------------------------------------------------- correction 실패
def test_web_failure_is_not_a_normal_correction():
    with pytest.raises(WebSearchFailed):
        run_policy("crag_rule", "BBB", web=FakeWeb(fail=KeyError("miss")))
    r = run_policy(
        "crag_rule", "BBB", web=FakeWeb(fail=RuntimeError("no key")), on_web_error="continue"
    )
    assert r["correction"]["status"] == "error" and r["correction"]["valid"] is False
    r = run_policy("crag_rule", "BBB", web=FakeWeb(docs=[]))
    assert r["correction"]["status"] == "empty" and r["correction"]["valid"] is False


def test_official_keyword_query_mode():
    web = FakeWeb()
    run_policy("crag_rule", "BBB", web=web, query_mode="official_keywords", wiki_prefix=True)
    assert web.queries == ["Wikipedia, rewritten"]  # kind=rewrite 호출 출력 + 공식 접두


def test_policy_registry_documents_sources():
    assert set(POLICIES) == {
        "vanilla",
        "crag_rule",
        "no_correct_filter",
        "no_incorrect_discard",
        "keep_all",
        "official",
        "official_fb_combined",
        "official_fb_internal",
        "official_fb_web_empty",
    }
    with pytest.raises(ValueError):
        get_policy("nope")


# ---------------------------------------------------------------- Incorrect fallback ablation
@pytest.mark.parametrize("pattern", ["MGB", "MBB"])  # correct / ambiguous 는 official 과 같아야 함
def test_fallback_policies_match_official_outside_incorrect(pattern):
    base = run_policy("official", pattern, prompt="official", record_texts=True)
    for pol in ("official_fb_combined", "official_fb_internal", "official_fb_web_empty"):
        r = run_policy(pol, pattern, prompt="official", record_texts=True)
        assert r["correction_path"] == base["correction_path"]
        assert r["generation_prompt"] == base["generation_prompt"], pol
        assert r["official_policy_semantics"] is False


def test_fallback_policies_incorrect_path():
    off = run_policy("official", "BBB", prompt="official", record_texts=True)
    assert off["correction_path"] == "incorrect"
    assert off["evidence"]["kept_retrieved_ids"] == []  # 공식: 검색 evidence 전부 제거
    assert off["evidence"]["added_ids"]  # web 만
    comb = run_policy("official_fb_combined", "BBB", prompt="official", record_texts=True)
    assert comb["evidence"]["kept_retrieved_ids"] and comb["evidence"]["added_ids"]
    assert (
        "Knowledge1: " in comb["generation_prompt"] and "Knowledge2: " in comb["generation_prompt"]
    )
    web = FakeWeb()
    intl = run_policy("official_fb_internal", "BBB", web=web, prompt="official", record_texts=True)
    assert web.queries == [] and intl["correction"]["performed"] is False
    assert intl["evidence"]["kept_retrieved_ids"] and not intl["evidence"]["added_ids"]
    # web_empty: web 결과가 있으면 공식과 같고, 비면 internal 로 대체
    w_ok = run_policy("official_fb_web_empty", "BBB", prompt="official", record_texts=True)
    assert w_ok["generation_prompt"] == off["generation_prompt"] and w_ok["fallback"] is None
    w_empty = run_policy(
        "official_fb_web_empty", "BBB", web=FakeWeb(docs=[]), prompt="official", record_texts=True
    )
    off_empty = run_policy(
        "official", "BBB", web=FakeWeb(docs=[]), prompt="official", record_texts=True
    )
    assert w_empty["fallback"] == {"applied": True, "reason": "empty"}
    assert w_empty["evidence"]["kept_retrieved_ids"] == ["d0", "d1", "d2"]
    assert off_empty["evidence"]["n_units"] == 0  # 공식: 빈 문단으로 생성
    with pytest.raises(ValueError, match="official"):
        run_policy("official_fb_combined", "BBB", budget={"preset": "lazy_rest"})


def test_vanilla_official_prompt_uses_same_evidence():
    leg = run_policy("vanilla", "MGB", record_texts=True)
    op = run_policy("vanilla", "MGB", prompt="official", record_texts=True)
    assert leg["evidence"]["selected_evidence_ids"] == op["evidence"]["selected_evidence_ids"]
    assert op["generation_prompt"].startswith("Refer to the following documents")
    assert " [sep] " in op["generation_prompt"]  # 공식 rag 처럼 문서 결합


# ---------------------------------------------------------------- frozen inputs
def _frozen(tmp_path, n=5):
    examples = {
        f"{i}": {"question": f"Q{i}? Paris", "answers": ["Paris"], "s_pop": 3} for i in range(n)
    }
    rows = {
        f"{i}": {
            "question": f"Q{i}? Paris",
            "gold_answers": ["Paris"],
            "retrieval_doc_ids": ["c0", "c1", "c2"],
            "extra": {"retrieval_scores": [0.9, 0.8, 0.7]},
        }
        for i in range(n)
    }
    chunks = {
        "c0": {"text": "GOOD chunk Paris"},
        "c1": {"text": "MID chunk", "title": "t"},
        "c2": {"text": "BAD"},
    }
    items, _ = build_frozen_items(list(examples), examples, [rows], chunks, 3)
    path = tmp_path / "inputs.jsonl"
    path.write_text("".join(json.dumps(it.to_dict()) + "\n" for it in items), encoding="utf-8")
    manifest_path_for(path).write_text(json.dumps({"sha256": file_sha256(path)}), encoding="utf-8")
    return path, examples, rows, chunks


def test_frozen_roundtrip_and_checks(tmp_path):
    path, examples, rows, chunks = _frozen(tmp_path)
    items, info = load_frozen(path)
    assert len(items) == 5 and [d.id for d in items[0].docs] == ["c0", "c1", "c2"]
    assert items[0].docs[1].metadata == {"title": "t"} and items[0].docs[0].score == 0.9
    other = {k: {**v, "retrieval_doc_ids": ["c1", "c0", "c2"]} for k, v in rows.items()}
    with pytest.raises(ValueError, match="Top-3 가 다름"):
        build_frozen_items(["0"], examples, [rows, other], chunks, 3)
    with pytest.raises(ValueError, match="텍스트 없음"):
        build_frozen_items(["0"], examples, [rows], {"c0": {"text": "x"}}, 3)
    with pytest.raises(ValueError, match="정답 불일치"):
        build_frozen_items(
            ["0"], examples, [{"0": {**rows["0"], "gold_answers": ["Lyon"]}}], chunks, 3
        )
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_frozen(path)


# ---------------------------------------------------------------- MODE B runner
def _cfg(tmp_path, frozen, **over):
    cfg = {
        "experiment": "official_crag",
        "exp_id": "SMOKE-test",
        "seed": 42,
        "output_dir": str(tmp_path / "results"),
        "inputs": {"frozen_path": str(frozen), "limit": None},
        "grader": {"type": "official_t5", "upper": 0.7, "lower": 0.3},
        "evaluator": {"task": "popqa"},
        "evidence_policy": "official",
        "budget": {"preset": "full_grading"},
        "correction": {"query_mode": "legacy_rewrite", "on_web_error": "fail"},
        "web_search": {"top_k": 5},
        "generation": {"prompt": "legacy"},
        "llm": {"provider": "fake", "model": "fake"},
        "scheduler": {"policy": "fcfs_query", "slots": 1},
        "run": {"purpose": "smoke", "mode": "sequential", "warmup_queries": 2},
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


@pytest.mark.parametrize(
    "grader,policy",
    [("official_t5", "official"), ("llama_prompt", "crag_rule"), ("official_t5", "vanilla")],
)
def test_mode_b_runner_end_to_end(tmp_path, grader, policy):
    frozen, *_ = _frozen(tmp_path)
    cfg = _cfg(
        tmp_path,
        frozen,
        evidence_policy=policy,
        grader={"type": grader, "upper": 0.7, "lower": 0.3},
    )
    comp = Components(engine=FakeEngine(), web_search=FakeWeb(), evaluator=t5_eval())
    run_dir = ModeBRunner(cfg, components=comp, run_id="r1").run()
    rows = [
        json.loads(x) for x in (run_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 7 and sum(r["warmup"] for r in rows) == 2
    for r in rows:
        assert "error" not in r, r.get("error")
        assert "schema_missing" not in r
        assert r["generation_input_tokens"] is not None and r["queue_wait"]["engine_queue"] is None
        assert r["request_id"].startswith("r1:")
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert s["n_measured"] == 5 and s["n_errors"] == 0 and s["accuracy"] == 1.0
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["inputs"]["sha256"] == file_sha256(frozen)
    assert meta["official_source"]["commit"].startswith("de7c296")
    assert meta["experiment_mode"] == "B_controlled"
    if grader == "llama_prompt":
        assert rows[-1]["llm_calls"]["grade"]["n"] == 3
    if policy == "vanilla":
        assert rows[-1]["grading_events"] == [] and rows[-1]["correction_path"] == "none"


def test_mode_b_paper_requires_approval(tmp_path):
    frozen, *_ = _frozen(tmp_path)
    cfg = _cfg(tmp_path, frozen, exp_id="C1-x", run={"purpose": "paper"})
    with pytest.raises(ValueError, match="paper-approved"):
        ModeBRunner(cfg, components=Components(engine=FakeEngine()), run_id="r")
    cfg = _cfg(tmp_path, frozen, exp_id="SYS-1")
    with pytest.raises(ValueError, match="SMOKE-"):
        ModeBRunner(cfg, components=Components(engine=FakeEngine()), run_id="r")


def test_mode_b_live_mode_with_workload(tmp_path):
    frozen, *_ = _frozen(tmp_path)
    cfg = _cfg(tmp_path, frozen, run={"purpose": "smoke", "mode": "live", "warmup_queries": 0})
    cfg["scheduler"] = {"policy": "fcfs_query", "slots": 2}
    wl = {
        "sha256": "test",
        "requests": [{"i": i, "query_id": str(i), "arrival": 0.01 * i} for i in range(5)],
    }
    comp = Components(engine=FakeEngine(), web_search=FakeWeb(), evaluator=t5_eval())
    run_dir = ModeBRunner(cfg, components=comp, run_id="live", workload=wl).run()
    rows = [
        json.loads(x) for x in (run_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(r["request_index"] for r in rows) == list(range(5))
    assert all(r["arrival"] is not None and "error" not in r for r in rows)
    with pytest.raises(ValueError, match="workload"):
        ModeBRunner(cfg, components=comp, run_id="live2")
