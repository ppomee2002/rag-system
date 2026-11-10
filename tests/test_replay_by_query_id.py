"""query_id 기준 web search 재생 (정책 간 같은 검색 조건) 기능 검증. 실제 Tavily 호출 없음."""

import json
import re
import shutil

import pytest
from crag_fixtures import FakeEngine, Retriever, pool_ids, read_jsonl  # noqa: F401
from runner import Components, Runner, build_web_search
from workload import make_workload

from rag.graph import CRAGGraph
from rag.retrieval.web_cache import (
    RecordingWebSearch,
    ReplayWebSearch,
    load_replay_index,
)
from rag.types import Document, QAExample


class QueryWeb:
    """검색어마다 다른 문서를 돌려주는 가짜 검색 (어떤 검색어로 찾았는지 문서 id 로 확인)."""

    def __init__(self):
        self.n = 0

    def search(self, query, top_k):
        self.n += 1
        tag = re.sub(r"\W+", "_", query)[:40]
        return [Document(id=f"w:{tag}:{i}", text=f"web {i}   for {query}") for i in range(top_k)]


class RewriteEngine(FakeEngine):
    """rewrite 출력 형식을 바꿀 수 있는 가짜 엔진 (동시 실행 시 rewrite 비결정성 흉내)."""

    def __init__(self, prefix, delay=0.002):
        super().__init__(delay=delay)
        self.prefix = prefix

    def complete(self, request):
        r = super().complete(request)
        if request.kind == "rewrite":
            q = request.prompt.split("Question: ", 1)[1].split("\n", 1)[0]
            r.text = f"{self.prefix} {q}"
        return r


# ---------------------------------------------------------------- unit
def write_cache(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def cache_row(query, ids):
    return {
        "query": query,
        "top_k": 2,
        "latency": 1.5,
        "docs": [{"id": i, "text": f"t {i}   x", "metadata": {}, "score": None} for i in ids],
    }


def test_replay_key_modes_and_miss_policies(tmp_path):
    cache = tmp_path / "web_cache.jsonl"
    write_cache(cache, [cache_row("author of museum", ["a1", "a2"])])
    index = {"310944": {"query": "author of museum", "run": "A"}}
    slept = []

    by_query = ReplayWebSearch([cache], "zero")
    with pytest.raises(KeyError, match="web cache miss"):  # 이번 실험의 실패 상황
        by_query.bind("310944").search("Author of The Museum", 2)

    by_id = ReplayWebSearch(
        [cache], "recorded", key_mode="query_id", query_index=index, sleep=slept.append
    )
    bound = by_id.bind("310944")
    docs = bound.search("Author of The Museum", 2)  # rewrite 가 달라도 같은 질문의 기록
    assert [d.id for d in docs] == ["a1", "a2"] and slept == [1.5]
    assert bound.last["rewrite_matched"] is False and bound.last["hit"] is True
    assert by_id.stats()["rewrite_mismatch"] == 1
    with pytest.raises(KeyError, match="query_id=999"):  # 기준 run 에 correction 기록 없음
        by_id.bind("999").search("x", 2)

    empty = ReplayWebSearch(
        [cache], "zero", key_mode="query_id", query_index=index, on_miss="empty"
    )
    b = empty.bind("999")
    assert b.search("x", 2) == [] and b.last["source"] == "empty"
    assert empty.stats()["empty_returned"] == 1 and empty.stats()["misses"] == 1

    with pytest.raises(ValueError):
        ReplayWebSearch([cache], key_mode="query_id")  # index 없음
    with pytest.raises(ValueError):
        ReplayWebSearch([cache], on_miss="ignore")


def test_load_replay_index_uses_recorded_rewrites(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_cache(
        a / "queries.jsonl",
        [
            {"query_id": "1", "extra": {"rewritten_query": " q one "}},
            {"query_id": "2", "extra": {"rewritten_query": None}},  # correct: 검색 없음
            {"query_id": "3", "warmup": True, "extra": {"rewritten_query": "w"}},
            {"query_id": "4", "error": "x", "extra": {"rewritten_query": "e"}},
        ],
    )
    write_cache(
        b / "queries.jsonl",
        [
            {"query_id": "1", "extra": {"rewritten_query": "other"}},
            {"query_id": "5", "extra": {"rewritten_query": "q5"}},
        ],
    )
    idx = load_replay_index([a, b])
    assert idx == {
        "1": {"query": "q one", "run": str(a)},
        "3": {"query": "w", "run": str(a)},
        "5": {"query": "q5", "run": str(b)},
    }  # warmup 포함, 오류·검색 없음 제외


def test_graph_uses_per_request_web_search():
    class Fixed:
        def search(self, q, k):
            return [Document(id="override", text="web 0")]

    class Ret:
        def retrieve(self, q, k):
            return [Document(id="d0", text="doc0", score=0.5)]

    llm = RewriteEngine("rw", delay=0)
    from rag.llm.engine_llm import EngineLLM

    graph = CRAGGraph(Ret(), EngineLLM(llm), QueryWeb(), top_k=1)
    res = graph.run(QAExample("q", "Question 3?", ["ans"]), web_search=Fixed())
    assert res.extra["web_doc_ids"] == ["override"]


# ---------------------------------------------------------------- runner: 정책 간 같은 검색 결과
@pytest.fixture
def reference_run(setup, tmp_path):
    tmp, make = setup
    rec = tmp_path / "rec.jsonl"
    cfg, _ = make(["exp_id=SMOKE-A1-FULL", "grading.strategy=full"])
    run_a = Runner(
        cfg,
        Components(Retriever(), RecordingWebSearch(QueryWeb(), rec), RewriteEngine("A:")),
        run_id="a",
    ).run()
    shutil.copy(rec, run_a / "web_cache.jsonl")  # runner 의 record=true 와 같은 위치
    return tmp, make, run_a


def live_cfg(make, run_a, key, pol="fcfs", extra=()):
    return make(
        [
            "run.mode=live",
            "exp_id=PRE-B",
            "run.purpose=pre",
            f"scheduler.policy={pol}",
            "scheduler.slots=2",
            "crag.web_search.provider=replay",
            f"crag.web_search.replay_from_runs=['{run_a.as_posix()}']",
            "crag.web_search.latency_mode=zero",
            f"crag.web_search.replay_key={key}",
            *extra,
        ]
    )


def test_policies_get_identical_web_results_despite_rewrite_drift(reference_run):
    tmp, make, run_a = reference_run
    a_rows = {r["query_id"]: r for r in read_jsonl(run_a / "queries.jsonl") if not r["warmup"]}

    # 기존 방식 (query 문자열 일치): rewrite 가 달라지면 miss -> 오류 (이번 실패 재현)
    cfg, _ = live_cfg(make, run_a, "query")
    wl = make_workload(pool_ids(cfg)[1], 10, seed=0, rate=500.0)
    web = build_web_search(cfg, tmp)
    run_q = Runner(
        cfg, Components(Retriever(), web, RewriteEngine("B:")), workload=wl, run_id="q"
    ).run()
    errs = [r for r in read_jsonl(run_q / "queries.jsonl") if r.get("error")]
    assert errs and all("web cache miss" in r["error"] for r in errs)

    # query_id 기준: 모든 정책이 기준 run 과 같은 web 문서를 받는다
    for pol in ("fcfs", "srpt_est"):
        extra = []
        if pol == "srpt_est":  # 같은 질문 trace 로 학습하므로 OOF
            cm = "{trace: '" + run_a.as_posix() + "', service_key: wall, oof_folds: 3}"
            extra = [f"scheduler.cost_model={cm}"]
        cfg, _ = live_cfg(make, run_a, "query_id", pol, extra)
        web = build_web_search(cfg, tmp)
        run_b = Runner(
            cfg, Components(Retriever(), web, RewriteEngine("B:")), workload=wl, run_id=f"id-{pol}"
        ).run()
        rows = [r for r in read_jsonl(run_b / "queries.jsonl") if not r["warmup"]]
        assert not any(r.get("error") for r in rows)
        corrected = [r for r in rows if r["corrective_action"] != "correct"]
        assert corrected
        for r in corrected:
            assert r["extra"]["web_doc_ids"] == a_rows[r["query_id"]]["extra"]["web_doc_ids"]
            assert r["web_replay"]["rewrite_matched"] is False and r["web_replay"]["hit"]
        end = json.loads((run_b / "run_end.json").read_text(encoding="utf-8"))
        assert end["web_search"]["key_mode"] == "query_id" and end["web_search"]["misses"] == 0
        all_rows = read_jsonl(run_b / "queries.jsonl")  # warmup 도 같은 기록으로 재생
        n_corr_all = sum(r.get("corrective_action") not in (None, "correct") for r in all_rows)
        assert end["web_search"]["rewrite_mismatch"] == n_corr_all >= len(corrected)


def test_query_id_key_requires_reference_runs(setup, tmp_path):
    _, make = setup
    cache = tmp_path / "c.jsonl"
    write_cache(cache, [cache_row("x", ["a"])])
    cfg, _ = make(
        [
            "crag.web_search.provider=replay",
            f"crag.web_search.replay_paths=['{cache.as_posix()}']",
            "crag.web_search.replay_key=query_id",
        ]
    )
    with pytest.raises(ValueError, match="replay_from_runs"):
        build_web_search(cfg, tmp_path)
