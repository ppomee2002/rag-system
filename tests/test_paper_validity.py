"""논문 타당성에 직접 영향을 주는 구현 검증 (기능 검증용, 성능·정확도 결과 아님).

1. QueryState 실행 상태 (stage / correction / 잔여 작업량 / 대기) 와 정책 입력 분리
2. 정적 SJF (sjf_pred) vs 상태 갱신형 SRPT-est
3. priority 설정 일치 검증 (client / engine / 서버)
4. cost model 데이터 누수 방지 (학습·평가 분리, OOF)
5. Full / Lazy / Batch 공정 비교 (같은 검색 결과, correction 검색 replay, 비교 가능성 검사)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rag.graph import CRAGGraph
from rag.llm.engine_llm import EngineLLM
from rag.llm.factory import build_engine
from rag.llm.ollama_engine import OllamaEngine
from rag.llm.vllm_engine import VLLMEngine
from rag.pipelines.crag import CRAG
from rag.retrieval.web_cache import RecordingWebSearch
from rag.scheduling import (
    POLICY_FIELDS,
    CostModel,
    OOFCostModel,
    QueryState,
    SchedulingGate,
    SJFPred,
    SRPTEst,
    build_policy,
    cost_model_from_dict,
    fit_cost_model,
    fit_oof_cost_model,
    job_from_row,
)
from rag.scheduling.policies import Aging, CallTicket
from rag.scheduling.preflight import (
    PreflightError,
    check_cost_model_leakage,
    check_launch_meta,
    check_priority_config,
    probe_server_priority,
)
from rag.telemetry.calls import CallRecorder
from rag.types import QAExample

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "crag_system"))
from comparability import compare_runs  # noqa: E402
from crag_fixtures import FakeEngine, Retriever, Web, pool_ids, read_jsonl  # noqa: E402
from runner import (  # noqa: E402
    Components,
    QueryRetriever,
    Runner,
    apply_split,
    build_web_search,
    load_examples,
    split_warmup,
)
from workload import make_workload as make_wl  # noqa: E402


def model(**kw):
    d = dict(
        grade_coef=np.array([1.0, 0.0]),
        gen_coef=np.array([2.0, 0.0]),
        rewrite_mean=0.5,
        gen_corrected_mean=3.0,
        total_beta=np.array([10.0, 0, 0, 0, 0, 0]),
        post_beta=np.array([2.0, 0, 0, 0, 0, 0]),
        meta={"train_query_ids": ["t1", "t2"]},
    )
    d.update(kw)
    return CostModel(**d)


def graph_run(question, gate=None, policy_state=None):
    """가짜 엔진 + gate 로 LangGraph CRAG 를 한 번 실행하고 QueryState 를 돌려준다."""
    q = policy_state or QueryState("q1")
    kw = {"recorder": CallRecorder(), "query_state": q}
    if gate is not None:
        kw["scheduler"] = gate
    llm = EngineLLM(FakeEngine(delay=0), **kw)
    graph = CRAGGraph(Retriever(), None, Web(), top_k=5, web_top_k=2)
    res = graph.run(QAExample("q1", question, ["ans"]), llm=llm, query_state=q)
    q.finish(res.corrective_action)
    return q, res


def stages(q):
    return [s for s, _ in q.stage_history]


# ================================================================ 1. QueryState
def test_policy_view_exposes_only_policy_fields():
    q = QueryState("q", s_pop=10)
    v = q.view()
    assert v.stage == "arrived" and v.corrected is None and v.s_pop == 10
    for hidden in ("action", "est_history", "stage_history", "retrieval_doc_ids", "lock"):
        with pytest.raises(AttributeError):
            getattr(v, hidden)
    with pytest.raises(AttributeError):
        v.stage = "done"
    # 정책 입력에 실제 실행시간 / 서비스 시간 필드가 없다 (누적 대기만 관측값)
    assert not any(w in f for f in POLICY_FIELDS for w in ("latency", "service", "wall", "server"))


def test_graph_reports_stage_and_correction_state():
    q, res = graph_run("Question 2?")  # even -> correct
    assert res.corrective_action == "correct" and q.corrected is False
    assert stages(q) == ["arrived", "retrieval", "grading", "decided", "generation", "done"]
    q, res = graph_run("Question 3?")  # odd -> incorrect
    assert q.corrected is True and q.action == "incorrect"
    assert stages(q) == [
        "arrived",
        "retrieval",
        "grading",
        "decided",
        "rewrite",
        "web_search",
        "generation",
        "done",
    ]
    assert q.n_graded == 5 and q.n_docs == 5 and len(q.retrieval_doc_ids) == 5


def test_reference_crag_pipeline_stage_via_llm_calls():
    q = QueryState("q2")
    llm = EngineLLM(FakeEngine(delay=0), query_state=q)
    res = CRAG(QueryRetriever(Retriever(), q), llm, Web(), top_k=3).run(
        QAExample("q2", "Question 3?", ["ans"])
    )
    q.finish(res.corrective_action)
    assert stages(q) == ["arrived", "grading", "rewrite", "generation", "done"]
    assert q.corrected is True and q.n_docs == 3  # QueryRetriever 가 검색 특징 기록


def test_gate_records_cumulative_wait_and_estimates():
    gate = SchedulingGate(SRPTEst(model()), slots=None)
    q, _ = graph_run("Question 3?", gate=gate)
    snap = q.snapshot()
    assert snap["n_dispatched"] == 7 and snap["cum_wait"] >= 0  # grade 5 + rewrite + generate
    assert [e["kind"] for e in snap["est_history"]] == ["grade"] * 5 + ["rewrite", "generate"]
    assert snap["est_remaining"] == snap["est_history"][-1]["value"]


# ================================================================ 2. SJF vs SRPT-est
def test_srpt_recomputes_after_grading_but_sjf_is_frozen():
    m = model()
    srpt_q, _ = graph_run("Question 3?", gate=SchedulingGate(SRPTEst(m), slots=None))
    vals = [e["value"] for e in srpt_q.est_history]
    assert vals[:5] == pytest.approx([5 + 2, 4 + 2, 3 + 2, 2 + 2, 1 + 2])  # 남은 grade + post
    assert vals[5] == pytest.approx(m.rewrite_mean + m.gen_corrected_mean)  # correction 반영
    assert vals[6] == pytest.approx(2.0)  # generation 만 남음

    sjf_q, _ = graph_run("Question 3?", gate=SchedulingGate(SJFPred(m), slots=None))
    sjf_vals = {e["value"] for e in sjf_q.est_history}
    assert len(sjf_vals) == 1  # 첫 호출 예측 (pred_total) 을 끝까지 유지
    assert sjf_vals.pop() == pytest.approx(10.0)
    assert len(sjf_q.est_history) == 7


def test_sjf_frozen_per_request_even_if_features_change():
    p = SJFPred(model(total_beta=np.array([1.0, 5.0, 0, 0, 0, 0])))
    q = QueryState("same", retrieval_scores=[0.2], retrieval_chars=[10], n_docs=1)
    first = p.base(CallTicket(q, "grade", 0, 10, 0.0, 0))
    q.retrieval_scores = [0.9]  # 이후 상태가 바뀌어도
    assert p.base(CallTicket(q, "generate", 0, 10, 0.0, 1)) == first
    other = QueryState("same", retrieval_scores=[0.9], retrieval_chars=[10], n_docs=1)
    assert p.base(CallTicket(other, "grade", 0, 10, 0.0, 2)) != first  # 같은 query 다른 요청
    assert q.request_key != other.request_key


def test_build_policy_sjf_and_aging_wrapper():
    assert build_policy({"policy": "sjf_pred"}, model()).name == "sjf_pred"
    aged = build_policy({"policy": "sjf_pred", "aging_alpha": 0.5}, model())
    assert isinstance(aged, Aging) and aged.base_is_estimate
    with pytest.raises(ValueError):
        build_policy({"policy": "sjf_pred"})


def test_live_runner_sjf_vs_srpt_same_workload(setup):
    _, make = setup
    trace = Runner(*make(["dataset.split.part=train"]), run_id="trace").run()
    results = {}
    for pol in ("sjf_pred", "srpt_est"):
        cfg, comp = make(
            [
                "run.mode=live",
                "dataset.split.part=eval",
                f"scheduler.policy={pol}",
                "scheduler.slots=1",
                f"scheduler.cost_model={{trace: '{trace.as_posix()}', service_key: wall}}",
            ]
        )
        wl = make_wl(pool_ids_split(cfg), 8, seed=4, rate=300.0, name="shared")
        run_dir = Runner(cfg, comp, workload=wl, run_id=pol).run()
        rows = [r for r in read_jsonl(run_dir / "queries.jsonl") if not r["warmup"]]
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["preflight"]["cost_model_leakage"]["status"] == "disjoint"
        results[pol] = (rows, meta["workload"]["sha256"])
    assert results["sjf_pred"][1] == results["srpt_est"][1]  # 같은 도착 패턴
    for r in results["sjf_pred"][0]:
        assert len({e["value"] for e in r["query_state"]["est_history"]}) == 1
    corrected = [r for r in results["srpt_est"][0] if r["query_state"]["corrected"]]
    assert corrected and all(
        len({e["value"] for e in r["query_state"]["est_history"]}) > 1 for r in corrected
    )


def pool_ids_split(cfg):
    _, pool = split_warmup(cfg, load_examples(cfg))
    return [ex.id for ex in apply_split(cfg, pool)]


# ================================================================ 3. priority 설정
@pytest.mark.parametrize(
    "sched,llm,ok",
    [
        ({}, {"provider": "vllm"}, True),
        ({"policy": "srpt_est", "engine_priority": "none"}, {"provider": "vllm"}, True),
        (
            {"policy": "srpt_est", "engine_priority": "key"},
            {"provider": "vllm", "send_priority": True, "server_scheduling_policy": "priority"},
            True,
        ),
        (
            {"policy": "srpt_est", "engine_priority": "key"},
            {"provider": "vllm", "send_priority": False, "server_scheduling_policy": "priority"},
            False,
        ),
        (
            {"policy": "srpt_est", "engine_priority": "none"},
            {"provider": "vllm", "send_priority": True},
            False,
        ),
        (
            {"policy": "srpt_est", "engine_priority": "key"},
            {"provider": "vllm", "send_priority": True, "server_scheduling_policy": "fcfs"},
            False,
        ),
        (
            {"policy": "srpt_est", "engine_priority": "key"},
            {"provider": "ollama", "send_priority": True, "server_scheduling_policy": "priority"},
            False,
        ),
        (
            {"policy": "none", "engine_priority": "key"},
            {"provider": "vllm", "send_priority": True, "server_scheduling_policy": "priority"},
            False,
        ),
        ({}, {"provider": "vllm", "server_scheduling_policy": "lifo"}, False),
    ],
)
def test_priority_config_matrix(sched, llm, ok):
    if ok:
        assert check_priority_config(sched, llm)["send_priority"] == llm.get("send_priority", False)
    else:
        with pytest.raises(PreflightError):
            check_priority_config(sched, llm)


def launch_file(tmp_path, *args, model_name="m"):
    p = tmp_path / "vllm_launch.json"
    p.write_text(json.dumps({"argv": ["vllm", "serve", model_name, *args]}), encoding="utf-8")
    return p


def test_launch_meta_checked_against_declared_policy(tmp_path):
    p = launch_file(tmp_path, "--scheduling-policy", "priority", "--max-num-seqs", "8")
    info = check_launch_meta(p, {"model": "m", "server_scheduling_policy": "priority"})
    assert info["actual_scheduling_policy"] == "priority" and info["max_num_seqs"] == "8"
    with pytest.raises(PreflightError, match="불일치"):
        check_launch_meta(p, {"model": "m", "server_scheduling_policy": "fcfs"})
    with pytest.raises(PreflightError, match="모델"):
        check_launch_meta(p, {"model": "other", "server_scheduling_policy": "priority"})
    default = launch_file(tmp_path, model_name="m")  # 인자 없으면 vLLM 기본 fcfs
    assert check_launch_meta(default, {"model": "m"})["actual_scheduling_policy"] == "fcfs"


def test_server_probe_against_mock(mock_server):
    fcfs = mock_server("vllm", sched_policy="fcfs")
    eng = VLLMEngine("mock-model", base_url=fcfs.url + "/v1", api_key_env=None)
    with pytest.raises(PreflightError, match="거부"):
        probe_server_priority(eng, "priority")
    assert probe_server_priority(eng, "fcfs")["nonzero_priority_accepted"] is False
    prio = mock_server("vllm", sched_policy="priority")
    eng2 = VLLMEngine("mock-model", base_url=prio.url + "/v1", api_key_env=None)
    assert probe_server_priority(eng2, "priority")["nonzero_priority_accepted"] is True
    assert "note" in probe_server_priority(eng2, "fcfs")  # 0.33.0: 수락만으로 판별 불가
    assert "skipped" in probe_server_priority(OllamaEngine("m"), "priority")


def test_runner_rejects_priority_mismatch_before_run_dir(setup):
    tmp, make = setup
    with pytest.raises(PreflightError):
        Runner(*make(["scheduler.policy=fcfs", "scheduler.engine_priority=key"]), run_id="bad")
    assert not (tmp / "results" / "crag_system" / "smoke" / "SMOKE-dev" / "bad").exists()


def test_runner_server_probe_failure_is_recorded(setup, mock_server):
    tmp, make = setup
    srv = mock_server("vllm", sched_policy="fcfs")
    cfg, _ = make(
        [
            "scheduler.policy=fcfs",
            "scheduler.engine_priority=key",
            "llm.send_priority=true",
            "llm.server_scheduling_policy=priority",
            "run.probe_server_policy=true",
            f"llm.base_url={srv.url}/v1",
            "llm.api_key_env=null",
        ]
    )
    comp = Components(Retriever(), Web(), build_engine(cfg["llm"]))
    runner = Runner(cfg, comp, run_id="probe")
    with pytest.raises(PreflightError):
        runner.run()
    failed = json.loads((runner.run_dir / "preflight_failed.json").read_text(encoding="utf-8"))
    assert "거부" in failed["error"]
    assert not (runner.run_dir / "queries.jsonl").exists()


# ================================================================ 4. cost model 누수
def trace_rows(n=10, slow=None):
    rows = []
    for i in range(n):
        svc = 50.0 if slow == f"q{i}" else 0.5 + 0.01 * i
        rows.append(
            {
                "query_id": f"q{i}",
                "corrective_action": "correct" if i % 2 else "incorrect",
                "calls": [
                    {"kind": "grade", "idx": k, "prompt_chars": 500, "t_start": k, "wall": svc}
                    for k in range(5)
                ]
                + [
                    {
                        "kind": "generate",
                        "idx": 0,
                        "prompt_chars": 2000,
                        "t_start": 9,
                        "wall": 2 * svc,
                    }
                ],
                "stage_latencies": {"retrieval": 0.01, "web_search": 0.0},
                "extra": {"retrieval_scores": [0.9 - 0.05 * i, 0.5, 0.4, 0.3, 0.2]},
                "retrieval_chars": [400 + 10 * i] * 5,
                "s_pop": 10 * i,
            }
        )
    return rows


def test_fit_records_train_ids_and_leakage_check():
    jobs = [job_from_row(r, "wall") for r in trace_rows()]
    m = fit_cost_model(jobs)
    assert m.train_query_ids == [f"q{i}" for i in range(10)]
    assert check_cost_model_leakage(m, {"x1", "x2"})["status"] == "disjoint"
    with pytest.raises(PreflightError, match="겹침"):
        check_cost_model_leakage(m, {"q1", "x1"})
    assert check_cost_model_leakage(m, {"q1"}, allow_overlap=True)["status"] == "overlap_allowed"
    legacy = model(meta={})
    with pytest.raises(PreflightError, match="검증 불가"):
        check_cost_model_leakage(legacy, {"q1"})
    assert check_cost_model_leakage(None, {"q1"})["status"] == "no_cost_model"


def test_oof_prediction_never_uses_own_runtime():
    """q3 의 실제 실행시간을 크게 바꿔도 OOF 에서 q3 의 예측은 그대로여야 한다."""
    normal = [job_from_row(r, "wall") for r in trace_rows()]
    slow = [job_from_row(r, "wall") for r in trace_rows(slow="q3")]
    feats = normal[3].feats
    oof_a = fit_oof_cost_model(normal, k=5, seed=1)
    oof_b = fit_oof_cost_model(slow, k=5, seed=1)
    assert oof_a.for_query("q3").pred_total(feats) == pytest.approx(
        oof_b.for_query("q3").pred_total(feats)
    )
    # 단일 모델은 자기 실행시간이 예측에 들어간다 (누수) -> 값이 바뀐다
    assert fit_cost_model(normal).pred_total(feats) != pytest.approx(
        fit_cost_model(slow).pred_total(feats)
    )
    # 다른 fold 의 query 예측에는 q3 데이터가 쓰인다 (정상)
    other = next(q for q, f in oof_a.fold_of.items() if f != oof_a.fold_of["q3"])
    assert oof_a.for_query(other) is not oof_a.for_query("q3")
    assert oof_a.for_query("unseen") is oof_a.full
    assert check_cost_model_leakage(oof_a, {"q3"})["status"] == "oof"
    back = cost_model_from_dict(json.loads(json.dumps(oof_a.to_dict())))
    assert isinstance(back, OOFCostModel) and back.fold_of == oof_a.fold_of


def test_split_is_disjoint_and_deterministic(setup):
    _, make = setup
    cfg_tr, _ = make(["dataset.split.part=train"])
    cfg_ev, _ = make(["dataset.split.part=eval"])
    cfg_all, _ = make()
    tr, ev, al = pool_ids_split(cfg_tr), pool_ids_split(cfg_ev), pool_ids_split(cfg_all)
    assert set(tr).isdisjoint(ev) and set(tr) | set(ev) == set(al) and len(tr) == 3
    assert pool_ids_split(make(["dataset.split.part=train"])[0]) == tr


def test_runner_rejects_cost_model_trained_on_eval_queries(setup):
    tmp, make = setup
    trace = Runner(*make(), run_id="trace_all").run()
    cfg, comp = make(
        [
            "run.mode=live",
            "scheduler.policy=srpt_est",
            f"scheduler.cost_model={{trace: '{trace.as_posix()}', service_key: wall}}",
        ]
    )
    wl = make_wl(pool_ids(cfg)[1], 5, seed=0, rate=100.0)
    with pytest.raises(PreflightError, match="겹침"):
        Runner(cfg, comp, workload=wl, run_id="leak")
    assert not (tmp / "results" / "crag_system" / "smoke" / "SMOKE-dev" / "leak").exists()


# ================================================================ 5. 공정 비교
def test_grading_comparison_with_replayed_correction_search(setup, tmp_path):
    tmp, make = setup
    rec_dir = tmp_path / "record"
    rec_dir.mkdir()
    live_web = Web()
    cfg, _ = make(["grading.strategy=full"])
    full = Runner(
        cfg,
        Components(
            Retriever(), RecordingWebSearch(live_web, rec_dir / "web_cache.jsonl"), FakeEngine()
        ),
        run_id="full",
    ).run()
    assert live_web.n > 0
    runs = [full]
    for strategy in ("lazy_hit", "batch"):
        cfg, _ = make(
            [
                f"grading.strategy={strategy}",
                "crag.web_search.provider=replay",
                f"crag.web_search.replay_from_runs=['{rec_dir.as_posix()}']",
                "crag.web_search.latency_mode=zero",
            ]
        )
        web = build_web_search(cfg, tmp_path)  # 실제 Tavily 없음 (fallback 없음)
        runs.append(Runner(cfg, Components(Retriever(), web, FakeEngine()), run_id=strategy).run())
        end = json.loads((runs[-1] / "run_end.json").read_text(encoding="utf-8"))
        assert end["web_search"]["misses"] == 0 and end["web_search"]["hits"] > 0
    report = compare_runs(runs)
    critical = [c for c in report["critical"] if c["check"] != "web_replay_sources"]
    assert critical == []  # 같은 질문·검색 결과·모델 설정
    assert report["varied"]["grading"] == ["full", "lazy_hit", "batch"]
    assert not any(w["check"] == "web_doc_ids_on_corrected" for w in report["warnings"])
    meta = json.loads((runs[1] / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["comparability"]["web_search"]["sources"][0]["sha256"]


class OtherRetriever(Retriever):
    def retrieve(self, query, top_k):
        docs = super().retrieve(query, top_k)
        return list(reversed(docs))


def test_comparability_flags_retrieval_and_model_differences(setup):
    _, make = setup
    a = Runner(*make(["run.session={id: s1, order_index: 0}"]), run_id="a").run()
    cfg, _ = make(["run.session={id: s1, order_index: 1}"])
    b = Runner(cfg, Components(OtherRetriever(), Web(), FakeEngine()), run_id="b").run()
    c = Runner(*make(["llm.model=another-model"]), run_id="c").run()
    rep_ab = compare_runs([a, b])
    assert not rep_ab["ok"]
    assert [x["check"] for x in rep_ab["critical"]] == ["retrieval_doc_ids"]
    assert [x["session"]["order_index"] for x in rep_ab["confounders"]] == [0, 1]
    rep_ac = compare_runs([a, c])
    assert "llm" in [x["check"] for x in rep_ac["critical"]]
