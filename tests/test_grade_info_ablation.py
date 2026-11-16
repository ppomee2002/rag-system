"""grade_info_ablation.py 기능 검증 (가짜 엔진 run / 손으로 만든 trace, 실제 성능 결과 아님)."""

import csv
import json

import fit_cost_model
import grade_info_ablation as gia
import numpy as np
import pytest
from crag_fixtures import FakeEngine, Retriever, Web
from runner import Components, Runner
from workload import make_workload, save_workload

from rag.scheduling.cost_model import CostModel


def _model():
    return CostModel(
        grade_coef=np.array([0.1, 0.0]), gen_coef=np.array([0.2, 0.0]), rewrite_mean=0.1,
        gen_corrected_mean=0.3, total_beta=np.zeros(6), post_beta=np.zeros(6),
    )  # fmt: skip


def _tq(qid, calls, scores=(), corrected=False, retrieval=0.0, web=0.0):
    return gia.TQ(
        key=f"0:{qid}", qid=qid, corrected=corrected, feats=np.zeros(6), n_docs=len(scores),
        scores=list(scores),
        calls=[{"kind": k, "idx": i, "chars": 10, "wall": w} for k, i, w in calls],
        retrieval=retrieval, web=web,
    )  # fmt: skip


def _train():
    """correct (첫 문서 hit) 2개, corrected 2개."""
    g = [("grade", 0, 1.0), ("grade", 1, 1.0), ("grade", 2, 1.0)]
    return [
        _tq("t1", [*g, ("generate", 0, 2.0)], [0.9, 0.1, 0.1]),
        _tq("t2", [*g, ("generate", 0, 2.0)], [0.1, 0.9, 0.1]),
        _tq("t3", [*g, ("rewrite", 0, 1.0), ("generate", 0, 4.0)], [0.1, 0.1, 0.1], True),
        _tq("t4", [*g, ("rewrite", 0, 1.0), ("generate", 0, 4.0)], [0.5, 0.1, 0.1], True),
    ]


def test_tables_and_grade_aware_uses_only_prefix():
    T = gia.Tables(_train(), 0.7, 0.3, min_cell=1)
    assert T.post_not == 2.0 and T.post_corr == 5.0
    assert T.stage["grade0"] == pytest.approx((5 + 5 + 8 + 8) / 4)
    assert T.p_corrected("c1", 1, [0.9]) == (0.0, "hit")
    p, src = T.p_corrected("c1", 1, [0.1])  # t2, t3, t4 중 t3, t4 corrected
    assert p == pytest.approx(2 / 3) and src == "1:nohit"
    m = _model()
    q = _tq("e", [("grade", 0, 1), ("grade", 1, 1), ("grade", 2, 1), ("generate", 0, 2)],
            [0.9, 0.1, 0.1])  # fmt: skip
    base = [gia.point_estimate(q, j, m, T) for j in range(len(q.calls))]
    # idx 0 은 관측 없음 -> C == A, hit 이후는 post = E[post | not corrected]
    assert base[0]["C1_grade_hit"] == base[0]["A_srpt_est"]
    assert base[1]["C1_grade_hit"] == pytest.approx(2 * 0.1 + T.post_not)
    # 최종 경로·마지막 점수를 바꿔도 C 는 그대로 (D 정답만 미래 정보)
    q2 = _tq("e", [(c["kind"], c["idx"], c["wall"]) for c in q.calls], [0.9, 0.1, 0.95], True)
    alt = [gia.point_estimate(q2, j, m, T) for j in range(len(q2.calls))]
    for x, y in zip(base, alt, strict=True):
        for k in ("A_srpt_est", "B_stage", "C1_grade_hit", "C2_grade_score"):
            assert x[k] == y[k]
    with pytest.raises(ValueError):  # 이후 점수를 넘기면 거부
        gia.estimates(m, T, "e", "grade", 1, 10, 3, np.zeros(6), [0.9, 0.1])
    e = gia.estimates(m, T, "e", "grade", 2, 10, 3, np.zeros(6), None)
    assert e["C1_grade_hit"] is None  # 관측 불가 표시


def test_simulate_k1_order_and_ps():
    T = gia.Tables(_train(), 0.7, 0.3, min_cell=1)
    m = _model()
    tq = {
        "a": _tq("a", [("generate", 0, 5.0)]),
        "b": _tq("b", [("generate", 0, 5.0)]),
        "c": _tq("c", [("generate", 0, 1.0)]),
    }
    reqs = [{"i": i, "query_id": q, "arrival": t} for i, (q, t) in enumerate(
        [("a", 0.0), ("b", 1.0), ("c", 2.0)])]  # fmt: skip
    one = gia.make_g({1: 1.0}, "ps")

    def lat(pol, K=1, g=one):
        return {r["query_id"]: r["latency"] for r in gia.simulate(reqs, tq, pol, K, g, m, T)}

    assert lat("fcfs") == pytest.approx({"a": 5, "b": 9, "c": 9})
    assert lat("D_oracle") == pytest.approx({"a": 5, "b": 10, "c": 4})
    ind = gia.make_g(None, "independent")
    assert lat("fcfs", 2, ind) == pytest.approx({"a": 5, "b": 5, "c": 4})
    # 처리율 모델: 2개 동시 실행 시 합계 처리율 1 (= 각 0.5)
    flat = gia.make_g({1: 1.0, 2: 1.0}, "ps")
    two = [{"i": 0, "query_id": "a", "arrival": 0.0}, {"i": 1, "query_id": "b", "arrival": 0.0}]
    res = gia.simulate(two, tq, "fcfs", 2, flat, m, T)
    assert [r["latency"] for r in res] == pytest.approx([10.0, 10.0])
    assert gia.make_g({1: 1.0, 2: 1.6, 4: 2.4}, "ps")(3) == pytest.approx(2.0)


def test_leakage_check():
    tr = _train()
    info, errs = gia.leakage_checks(tr, [_tq("t1", [("generate", 0, 1.0)])], _model())
    assert info["train_eval_overlap"] == 1 and errs


@pytest.fixture
def sv(setup):
    tmp, make = setup
    train = Runner(*make(["dataset.split.part=train"]), run_id="train").run()
    seq = Runner(*make(["dataset.split.part=eval"]), run_id="eval_seq").run()
    cm = tmp / "cm.json"
    fit_cost_model.main([str(train), "--service-key", "wall", "--out", str(cm)])
    wdir = tmp / "wl"
    for s in (0, 1):
        cfg, _ = make(["dataset.split.part=eval"])
        wl = make_workload(_eval_ids(cfg), 8, seed=s, rate=2000.0)
        wl = save_workload(wl, wdir / f"SV1_rho0.8_s{s}.json")
        wl = json.loads(wl.read_text(encoding="utf-8"))
        for pol in ("fcfs", "fcfs_query", "srpt_nopath", "srpt_est"):
            cfg, _ = make(
                ["run.mode=live", "exp_id=PRE-SV", "run.purpose=pre", "dataset.split.part=eval",
                 f"scheduler.policy={pol}", "scheduler.slots=1",
                 "scheduler.cost_model={path: '" + cm.as_posix() + "'}"]
            )  # fmt: skip
            Runner(
                cfg, Components(Retriever(), Web(), FakeEngine(delay=0.01)), workload=wl,
                run_id=f"rho0.8_s{s}_{pol}_K1",
            ).run()  # fmt: skip
    return tmp, train, seq, cm, wdir


def _eval_ids(cfg):
    from runner import apply_split, load_examples, split_warmup

    _, pool = split_warmup(cfg, load_examples(cfg))
    return [ex.id for ex in apply_split(cfg, pool)]


def test_end_to_end(sv, tmp_path):
    tmp, train, seq, cm, wdir = sv
    root = tmp / "results" / "crag_system" / "pre" / "PRE-SV"
    out = gia.main(
        ["--train", str(train), "--seq", str(seq), "--cost-model", str(cm),
         "--live-root", str(root), "--k", "1", "--rhos", "0.8", "--seeds", "0", "1",
         "--workloads", str(wdir), "--tag", "SV1", "--speedup", "1:1", "--min-cell", "1",
         "--n-boot", "200", "--name", "g", "--out-root", str(tmp_path), "--quiet"]
    )  # fmt: skip
    res = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert res["leakage"]["train_eval_overlap"] == 0
    assert res["grading_check"]["eval"]["not_sequential"] == 0
    assert res["grading_check"]["eval"]["score_count_mismatch"] == 0
    with open(out / "point_estimates.csv", encoding="utf-8-sig") as f:
        pts = list(csv.DictReader(f))
    d = [r for r in pts if r["estimator"] == "D_oracle" and r["stage"] == "all"]
    assert d and all(float(r["mae"]) == 0.0 for r in d)
    q = res["queue"]
    assert {r["run"] for r in q} == {"fcfs_query", "srpt_est"}
    for r in q:  # 자기 정책 규칙 = 실제 선택
        if r["run"] == "srpt_est" and r["rule"] == "A_srpt_est" and r["n_available"]:
            assert r["agree_actual"] == pytest.approx(1.0)
        if r["run"] == "fcfs_query" and r["rule"] == "fcfs_query" and r["n_available"]:
            assert r["agree_actual"] == pytest.approx(1.0)
    sim = {(r["cond"], r["policy"]): r for r in res["sim"]}
    assert len(sim) == 2 * len(gia.SIM_POLICIES)
    assert sim[("rho0.8_s0", "A_srpt_est")]["live_mean"] is not None
    assert {r["group"] for r in res["sim_paired"]} >= {"rho0.8_s0", "all_pooled"}
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "누수" in md and "시뮬레이션" in md
