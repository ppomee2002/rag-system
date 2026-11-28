"""C2 gating (MODE A 위의 B / C / Cbn) — T5Service, 공식 규칙 동등성, 조건 간 정합성 (mock).

mock 결과는 정확도·latency 로 쓰지 않는다. 여기서는 제어 흐름 (취소, submit 시점, 누수, 동일성) 만
  본다.
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from rag.crag_official.evaluator import MockScoreBackend, OfficialRetrievalEvaluator, pair_text
from rag.crag_official.refinement import internal_knowledge, official_passages_from_line
from rag.crag_official.t5_service import (
    JobCancelled,
    JobScorer,
    MockPairRunner,
    OfficialRules,
    T5Service,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "c2_gating"))
from test_t5_backend_codepath import tiny_t5 as _tiny_t5_fixture  # noqa: E402

tiny_t5 = _tiny_t5_fixture


def score_fn(text: str) -> float:
    p = text.split(" [SEP] ", 1)[-1]
    return 1.0 if "GOOD" in p else (0.0 if "MID" in p else -2.0)


# ---------------------------------------------------------------- official rules
def test_rules_match_official_evaluator():
    passages = ["GOOD doc one two", "", "  MID  doc\twith tab ", "bad doc here now", "a b"]
    ev = OfficialRetrievalEvaluator(MockScoreBackend(lambda q, p: score_fn(pair_text(q, p))))
    want = [p.score for p in ev.score_documents("q?", passages)]
    svc = T5Service(MockPairRunner(score_fn))
    rules = OfficialRules(svc)
    got = [p.score for p in rules.submit_documents("r", "q?", passages).result(5)]
    assert got == want
    strips = official_passages_from_line(" [sep] ".join(passages))
    want_s = [p.score for p in ev.score_strips("q?", strips)]
    got_s = [p.score for p in rules.submit_strips("r", "q?", strips, "internal_refine").result(5)]
    assert got_s == want_s
    # refinement 결과 (순위·텍스트) 도 같다
    a = internal_knowledge("q?", strips, ev, "selection")
    b = internal_knowledge("q?", strips, JobScorer(rules, "r", "internal_refine"), "selection")
    assert (a.text, a.ranking) == (b.text, b.ranking)
    svc.close()


# ---------------------------------------------------------------- service
def test_fifo_cancel_queued_and_running():
    gate = threading.Event()

    def slow_sleep(s):
        gate.wait(5)

    runner = MockPairRunner(score_fn, sleep=slow_sleep)
    svc = T5Service(runner)
    j1 = svc.submit("a", "grading", ["q [SEP] x", "q [SEP] y", "q [SEP] z"], False)
    time.sleep(0.05)
    j2 = svc.submit("b", "external_refine", ["q [SEP] w"], True)
    assert svc.cancel(j2) == "removed_from_queue"
    assert j2.status == "cancelled" and j2.busy_s == 0 and j2.n_forward == 0
    assert svc.cancel(j1) == "stop_requested"
    gate.set()
    with pytest.raises(JobCancelled):
        j1.wait(5)
    assert j1.cancel_effect == "stopped_running" and j1.n_forward == 1  # 첫 batch 만
    assert svc.idle(5) and svc.snapshot()["queue_len"] == 0
    svc.close()


def test_memo_tensor_key_and_verify():
    runner = MockPairRunner(score_fn, max_length=8)
    svc = T5Service(runner, memo=True, memo_verify=True)
    g = svc.submit("r", "grading", ["q [SEP] GOOD a b", "q [SEP] " + "w " * 20], False)
    g.wait(5)
    # 앞뒤 공백만 다른 strip -> 같은 텐서 -> hit. 512(여기선 8) 초과 + truncation -> 다른 텐서 ->
    # miss
    s = svc.submit("r", "internal_refine", ["q [SEP]  GOOD a b ", "q [SEP] " + "w " * 20], True)
    s.wait(5)
    assert s.n_memo_hits == 1 and s.memo_mismatch == 0
    assert s.n_forward == 2  # verify 모드: hit 도 다시 계산
    other = svc.submit("x", "internal_refine", ["q [SEP] GOOD a b"], True)
    other.wait(5)
    assert other.n_memo_hits == 0  # 다른 요청의 memo 는 쓰지 않음
    svc.release_request("r")
    svc.release_request("x")
    assert svc.snapshot()["memo_requests"] == 0
    svc.close()


def test_memo_not_recreated_after_release():
    gate = threading.Event()
    svc = T5Service(MockPairRunner(score_fn, sleep=lambda s: gate.wait(5)), memo=True)
    j1 = svc.submit("a", "grading", ["q [SEP] x y z w"], False)
    j2 = svc.submit("b", "external_refine", ["q [SEP] x y z w"], True)
    svc.release_request("b")  # 요청 b 가 끝난 뒤 job 이 실행되는 경우 (C 의 늦은 취소)
    gate.set()
    j1.wait(5), j2.wait(5)
    svc.release_request("a")
    assert svc.snapshot()["memo_requests"] == 0
    svc.close()


def test_batching_groups_same_length_only():
    runner = MockPairRunner(score_fn, max_length=8)
    svc = T5Service(runner, batch_size=4)
    texts = ["q [SEP] a b", "q [SEP] c d", "q [SEP] " + "w " * 20, "q [SEP] e f"]
    j = svc.submit("r", "grading", texts, False)
    assert j.wait(5) == [score_fn(t) for t in texts]
    assert j.n_batches == 3  # [0,1] [2 (긴 입력 혼자)] [3]
    svc.close()


# ---------------------------------------------------------------- real (tiny) T5
def test_t5_pair_runner_matches_backend(tiny_t5):
    from rag.crag_official.evaluator import T5EvaluatorBackend
    from rag.crag_official.t5_service import T5PairRunner

    be = T5EvaluatorBackend(tiny_t5, device="cpu")
    pairs = [("the question", "the city of the politician born"), ("q", "")]
    long = ("the question", " ".join(["occupation"] * 700))
    for trunc in (False, True):
        want = [p.score for p in be.score_pairs([*pairs, long], truncation=trunc)]
        svc = T5Service(T5PairRunner(be))
        got = svc.submit("r", "k", [pair_text(q, p) for q, p in [*pairs, long]], trunc).wait(60)
        svc.close()
        assert got == want
    # 공식 grading (normalize) vs refinement (앞뒤 공백 유지) 텍스트의 텐서가 같은지
    # = memo 성립 조건
    r = T5PairRunner(be)
    a = r.encode(pair_text("q", "the city of"), False)
    b = r.encode(pair_text("q", " the city of "), True)
    assert a.ids == b.ids  # sentencepiece 공백 정규화 (공식 가중치에서는 서버 smoke 로 다시 확인)


# ---------------------------------------------------------------- end-to-end (mock)
def make_repo(tmp: Path, n: int = 12) -> tuple[Path, Path]:
    d = tmp / "repo" / "data" / "popqa"
    (d / "ref").mkdir(parents=True)
    lines, srcs, refs = [], [], {"correct": [], "incorrect": [], "ambiguous": []}
    web_rows = []
    for i in range(n):
        q = f"What is item {i}?"
        kind = ("correct", "ambiguous", "incorrect")[i % 3]
        tag = {"correct": "GOOD", "ambiguous": "MID", "incorrect": "bad"}[kind]
        docs = [
            f"{tag} passage {i} number {k} text" if k == 3 else f"bad doc {k} for {i} words"
            for k in range(10)
        ]
        docs[7] = ""  # 빈 문서
        lines += [f"{q} [SEP] {p}" for p in docs]
        srcs.append(q)
        refs["correct"].append(f"#correct knowledge {i}")
        refs["incorrect"].append(f"#incorrect knowledge {i}")
        refs["ambiguous"].append(f"Knowledge1: c{i} [sep] Knowledge2: e{i}")
        web_rows.append(
            {
                "query_id": str(i),
                "query": f"Wikipedia, {q}, keyword",
                "top_k": 5,
                "latency": 0.03,
                "cache_version": "web_cache.v2",
                "docs": [],
                "raw_response": {
                    "results": [
                        {
                            "url": f"https://en.wikipedia.org/wiki/I{i}_{k}",
                            "title": f"I{i}",
                            "content": "snippet text here",
                            "raw_content": "\n".join(
                                f"paragraph {m} about item {i} with words" for m in range(6)
                            ),
                        }
                        for k in range(3)
                    ]
                    + [{"url": "https://example.com", "title": "x", "content": "other snippet"}]
                },
            }
        )
    (d / "test_popqa.txt").write_text("\n".join(lines), encoding="utf-8")
    (d / "sources").write_text("\n".join(srcs), encoding="utf-8")
    for k, v in refs.items():
        (d / "ref" / k).write_text("\n".join(v), encoding="utf-8")
    web = tmp / "web_cache.jsonl"
    web.write_text("\n".join(json.dumps(r) for r in web_rows), encoding="utf-8")
    return tmp / "repo", web


def _load(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"c2_gating_{name}", ROOT / "experiments" / "c2_gating" / f"{name}.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    c2run = _load("run")

    tmp = tmp_path_factory.mktemp("c2")
    repo, web = make_repo(tmp)
    out = c2run.main(
        [
            "--mode-a-config",
            str(ROOT / "experiments/official_crag/configs/mode_a_llama2_7b_tp2.yaml"),
            "--repo",
            str(repo),
            "--web-cache",
            str(web),
            "--limit",
            "9",
            "--mock",
            "--modes",
            "B,C,Cbn",
            "--concurrency",
            "1,4",
            "--repeats",
            "1",
            "--purpose",
            "smoke",
            "--output-dir",
            str(tmp / "results"),
            "--set",
            "run.warmup_queries=2",
            "--set",
            "run.gpu_sample_ms=0",
            "--set",
            "t5_service.memo=true",
            "--set",
            "t5_service.memo_verify=true",
        ]
    )
    return json.loads((out / "summary.json").read_text(encoding="utf-8")), out


def test_identity_across_conditions(mock_run):
    s, _ = mock_run
    assert all(s["identity"]["all_identical"].values()), s["identity"]
    for c in s["conditions"].values():
        assert c["n_errors"] == 0, c["errors"]
        assert c["action_counts"] == {"ambiguous": 3, "correct": 3, "incorrect": 3}


def test_correct_external_refine_and_cbn_gate(mock_run):
    s, _ = mock_run
    for cid, c in s["conditions"].items():
        mode = c["cond"]["mode"]
        if mode in ("B", "Cbn"):
            assert c["correct_external_refine"]["n_jobs_submitted"] == 0, cid
            assert c["t5"]["busy_per_request_mean_s"]["external_wasted"] == 0, cid
        if mode in ("B", "Cbn"):
            assert c["external_submit_before_decision"] == 0, cid
        if mode in ("C", "Cbn"):
            assert c["overlap"]["stage_overlap_frac"] == 1.0, cid
            assert c["waste"]["n_rewrite"] == 3, cid  # correct 3문항의 speculative rewrite
        if mode == "B":
            assert c["overlap"]["n_speculative"] == 0 and c["waste"]["n_rewrite"] == 0


def test_no_leaks_and_memo(mock_run):
    s, _ = mock_run
    for cid, c in s["conditions"].items():
        assert c["leak"]["ok"], (cid, c["leak"])
        assert c["t5"]["memo_mismatch"] == 0
        assert c["t5"]["memo_hits"] > 0  # internal refinement 가 grading 결과 재사용


def test_c_wastes_external_refine_on_correct(mock_run):
    """C 는 web 이후 바로 submit -> grading 보다 늦게 끝나는 경우 낭비가 생길 수 있다 (mock 시간상
    grading 50ms > rewrite 20ms + web 30ms 라 일부). 0 이어도 실패는 아니지만 기록은 있어야 한다."""
    s, _ = mock_run
    for c in s["conditions"].values():
        if c["cond"]["mode"] == "C":
            ce = c["correct_external_refine"]
            assert ce["n_jobs_submitted"] >= ce["n_jobs_executed"] >= 0


def test_smoke_check_passes_on_mock(mock_run):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "c2_smoke_check", ROOT / "experiments" / "c2_gating" / "smoke_check.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _, run_dir = mock_run
    rep = m.check(run_dir)
    assert rep["all_ok"], [r for r in rep["checks"] if not r["ok"]]
    assert any("memo 결과 비트 단위 동일" in r["check"] for r in rep["checks"])


def test_controlled_replay_mode_and_precondition(mock_run):
    s, run_dir = mock_run
    assert s["experiment_mode"] == "controlled_replay"
    assert "controlled_replay" in str(run_dir)
    assert "End-to-End" in s["claim_scope"] and s["deviations"]["web_search"]
    for c in s["conditions"].values():
        gp = c["generation_precondition"]
        assert gp["n_checked"] == c["n_ok"] and gp["n_violations"] == 0, gp


def test_runtime_e2e_mock(tmp_path):
    c2run = _load("run")
    repo, web = make_repo(tmp_path)
    out = c2run.main(
        [
            "--mode-a-config",
            str(ROOT / "experiments/official_crag/configs/mode_a_llama2_7b_tp2.yaml"),
            "--repo",
            str(repo),
            "--web-cache",
            str(web),
            "--limit",
            "6",
            "--mock",
            "--modes",
            "B,C,Cbn",
            "--concurrency",
            "2",
            "--repeats",
            "1",
            "--purpose",
            "smoke",
            "--output-dir",
            str(tmp_path / "results"),
            "--set",
            "run.warmup_queries=2",
            "--set",
            "run.gpu_sample_ms=0",
            "--set",
            "gating.experiment_mode=runtime_e2e",
        ]
    )
    assert "runtime_e2e" in str(out)
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert all(s["identity"]["all_identical"].values()), s["identity"]
    rows = [
        json.loads(x)
        for x in (next((out / "conds").iterdir()) / "queries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    ids = {r["evidence_id"].split("#")[0] for r in rows if not r.get("warmup")}
    assert ids == {"runtime/internal", "runtime/external", "runtime/ambiguous"}
    for c in s["conditions"].values():
        assert c["generation_precondition"]["n_violations"] == 0


def test_workload_sets_disjoint_and_composition():
    m = _load("make_workload_sets")
    rows = (
        [{"index": i, "action": "correct"} for i in range(700)]
        + [{"index": 700 + i, "action": "ambiguous"} for i in range(400)]
        + [{"index": 1100 + i, "action": "incorrect"} for i in range(299)]
    )
    sets, info = m.build(rows, 200, None, 0)
    assert info["ratios"]["mid"] == 0.5 and info["disjoint"]
    assert info["ratios"]["low"] < 0.5 < info["ratios"]["high"]
    act = {r["index"]: r["action"] for r in rows}
    for k, idx in sets.items():
        assert len(idx) == 200 == len(set(idx))
        nc = sum(act[i] == "correct" for i in idx)
        assert nc == round(200 * info["ratios"][k])
    assert not any(info["overlap"].values())


def test_recommend_util():
    m = _load("probe_serving")
    assert m.recommend_util(14.6, 2.0, 0.85) == (True, 0.85)
    fits, rec = m.recommend_util(14.6, 4.5, 0.85)
    assert not fits and rec == 0.69 and 14.6 * (1 - rec) >= 4.5


def test_resolve_mode_a_synthetic(tmp_path):
    import yaml

    m = _load("resolve_mode_a")
    repo, _ = make_repo(tmp_path, n=6)
    qs = (repo / "data/popqa/sources").read_text(encoding="utf-8").split("\n")
    ans = tmp_path / "answers.jsonl"
    ans.write_text(
        "\n".join(
            json.dumps({"id": i, "question": q, "answers": [f"A{i}"]}) for i, q in enumerate(qs)
        ),
        encoding="utf-8",
    )
    cfg = str(ROOT / "experiments/official_crag/configs/mode_a_llama2_7b_tp2.yaml")
    llm = m.load_layered_config(cfg, [], [])["llm"]
    acts = ["correct", "ambiguous", "incorrect"] * 2
    sc = tmp_path / "score"
    sc.mkdir()
    (sc / "scores.jsonl").write_text(
        "\n".join(json.dumps({"index": i, "action": a, "scores": []}) for i, a in enumerate(acts)),
        encoding="utf-8",
    )
    for name, method, hits in (("rag", "rag", {0}), ("crag", "crag", {0, 1})):
        d = tmp_path / name
        d.mkdir()
        conf = {"method": method, "llm": llm, "prompt_version": "v3_2024_10"}
        if method == "crag":
            conf["scores_run"] = str(sc)
        (d / "config.yaml").write_text(yaml.safe_dump(conf), encoding="utf-8")
        (d / "preds.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "index": i,
                        "action": acts[i] if method == "crag" else None,
                        "prediction": f"A{i}" if i in hits else "x",
                    }
                )
                for i in range(6)
            ),
            encoding="utf-8",
        )
    args = m.argparse.Namespace(
        mode_a_config=cfg,
        scores=str(sc),
        rag_preds=str(tmp_path / "rag"),
        crag_preds=str(tmp_path / "crag"),
        prompt_version="v3_2024_10",
        expect_rag=1,
        expect_crag=2,
        answers=str(ans),
        repo=str(repo),
    )
    m.DEFAULTS["n"] = 6
    res = m.cmd_resolve(args)
    assert res["ok"], [c for c in res["checks"] if not c["ok"]]
    assert res["answers"] == str(ans) and res["official_match"] == {"rag": (1, 6), "crag": (2, 6)}
    args.expect_crag = 3
    assert not m.cmd_resolve(args)["ok"]
