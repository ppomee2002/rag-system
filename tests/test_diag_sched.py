"""diag_sched.py 기능 검증 (가짜 엔진 run, 실제 성능 결과 아님)."""

import csv
import json

import diag_sched
import pytest
from crag_fixtures import FakeEngine, Retriever, Web, pool_ids
from runner import Components, Runner
from workload import make_workload


def log(seqs, n_waiting, key="0:q"):
    """모두 t=0 에 enqueue 되고 1초 간격으로 dispatch 된 gate 로그 (wait = t)."""
    return [
        {"seq": s, "n_waiting": w, "t": float(i + 1), "wait": float(i + 1), "request_key": key}
        for i, (s, w) in enumerate(zip(seqs, n_waiting, strict=True))
    ]


def test_reorder_vs_call_from_gate_sequence():
    # seq 3 이 seq 2 보다 먼저 나감 -> 호출 순서 기준 1 회 재정렬 (2 를 앞지름)
    m = diag_sched.dispatch_metrics(log([0, 1, 3, 2, 4], [4, 3, 2, 1, 0]), {"0:q"})
    assert m["reorder_vs_call_dispatches"] == 1 and m["reorder_vs_call_skipped_mean"] == 1.0
    assert m["queue_reconstruction_mismatch"] == 0
    assert m["choice_ge2"] == pytest.approx(4 / 5)
    # 순서대로면 0 (fcfs 검증 기준)
    m = diag_sched.dispatch_metrics(log([0, 1, 2, 3], [3, 2, 1, 0]), {"0:q"})
    assert m["reorder_vs_call_dispatches"] == 0
    # 여러 개를 앞지르는 경우
    m = diag_sched.dispatch_metrics(log([3, 0, 1, 2], [3, 2, 1, 0]), {"0:q"})
    assert m["reorder_vs_call_dispatches"] == 1 and m["reorder_vs_call_skipped_mean"] == 3.0
    assert diag_sched.dispatch_metrics([], set()) == {"available": False}


def fcfs_query_log():
    """fcfs_query 동작: A 가 먼저 도착, B 가 나중. A 의 다음 호출 (seq 2) 이 B 의 호출 (seq 1) 보다
    늦게 enqueue 되지만, A 가 먼저 도착했으므로 먼저 나간다.

        t=0.0 A grade  enqueue (seq 0)   t=0.1 dispatch seq 0
        t=0.5 B grade  enqueue (seq 1)
        t=1.0 A gen    enqueue (seq 2)   t=1.2 dispatch seq 2 (대기: seq 1, 2)
                                         t=2.0 dispatch seq 1
    """
    return [
        {"seq": 0, "t": 0.1, "wait": 0.1, "n_waiting": 0, "request_key": "0:A"},
        {"seq": 2, "t": 1.2, "wait": 0.2, "n_waiting": 1, "request_key": "0:A"},
        {"seq": 1, "t": 2.0, "wait": 1.5, "n_waiting": 0, "request_key": "1:B"},
    ]


def test_fcfs_query_reorders_calls_but_keeps_arrival_order():
    m = diag_sched.dispatch_metrics(fcfs_query_log(), {"0:A", "1:B"}, {"0:A": 0.0, "1:B": 0.5})
    assert m["reorder_vs_call_dispatches"] == 1  # 호출 단위로는 순서가 바뀜 (정상)
    assert m["reorder_vs_arrival_dispatches"] == 0  # 요청 도착 순서는 유지
    assert m["queue_reconstruction_mismatch"] == 0


def test_not_yet_enqueued_calls_are_not_counted():
    """dispatch 시점에 아직 enqueue 되지 않은 (먼저 도착한 요청의) 호출은 재정렬로 세지 않는다."""
    entries = [
        {"seq": 1, "t": 0.6, "wait": 0.1, "n_waiting": 0, "request_key": "1:B"},  # B 먼저 나감
        {"seq": 2, "t": 1.1, "wait": 0.1, "n_waiting": 0, "request_key": "0:A"},  # A 는 1.0 에 도착
    ]
    m = diag_sched.dispatch_metrics(entries, {"0:A", "1:B"}, {"0:A": 0.0, "1:B": 0.5})
    assert m["reorder_vs_arrival_dispatches"] == 0 and m["reorder_vs_call_dispatches"] == 0


def test_warmup_dispatches_excluded_but_tracked():
    entries = log([0, 1], [3, 2], key="-1:w") + log([3, 2], [1, 0], key="0:q")[:2]
    for i, e in enumerate(entries):
        e["t"] = e["wait"] = float(i + 1)
    m = diag_sched.dispatch_metrics(entries, {"0:q"})
    assert m["n_dispatch"] == 2 and m["reorder_vs_call_dispatches"] == 1


def test_estimate_metrics_rank_correlation_and_post_grading():
    rows, calls = [], {}
    for i, (walls, corrected) in enumerate(
        [([1, 1, 4], True), ([1, 1, 1], False), ([2, 2, 2], False)]
    ):
        kinds = [("grade", 0), ("grade", 1), ("rewrite" if corrected else "generate", 0)]
        calls[i] = [
            {"kind": k, "idx": x, "t_start": j, "wall": w}
            for j, ((k, x), w) in enumerate(zip(kinds, walls, strict=True))
        ]
        remaining = [sum(walls[j:]) for j in range(3)]
        hist = [
            {"kind": k, "idx": x, "value": r * 1.1}
            for (k, x), r in zip(kinds, remaining, strict=True)
        ]
        rows.append(
            {
                "request_index": i,
                "query_state": {"est_history": hist},
                "corrective_action": "incorrect" if corrected else "correct",
            }
        )
    m = diag_sched.estimate_metrics(rows, calls)
    assert m["available"] and m["n_points"] == 9
    assert m["spearman_vs_actual"] == pytest.approx(1.0, abs=0.05)
    assert m["post_grading_ratio"]["corrected"]["mean"] == pytest.approx(4 / 5)
    assert not diag_sched.estimate_metrics([{"request_index": 0, "query_state": {}}], {})[
        "available"
    ]


@pytest.fixture
def policy_runs(setup):
    _, make = setup
    trace = Runner(*make(), run_id="trace").run()
    cm = "{trace: '" + trace.as_posix() + "', service_key: wall, oof_folds: 3}"
    out = {}
    for name, pol, k in (
        ("fcfs", "fcfs", 1),
        ("fcfsq", "fcfs_query", 1),
        ("srpt", "srpt_est", 1),
        ("srpt6", "srpt_est", 6),
        ("none", "none", 1),
    ):
        sets = [
            "run.mode=live",
            "exp_id=PRE-B",
            "run.purpose=pre",
            f"scheduler.policy={pol}",
            f"scheduler.slots={k}",
            f"scheduler.cost_model={cm}",
            "run.sample_engine_metrics_sec=0.05",
        ]
        cfg, _ = make(sets)
        wl = make_workload(pool_ids(cfg)[1], 10, seed=0, rate=2000.0, name="burst")
        out[name] = Runner(
            cfg, Components(Retriever(), Web(), FakeEngine(delay=0.01)), workload=wl, run_id=name
        ).run()
    return out


def test_diag_sched_end_to_end(policy_runs, tmp_path):
    runs = [policy_runs[k] for k in ("fcfs", "srpt", "srpt6")]
    out = diag_sched.main([*map(str, runs), "--name", "t", "--out-root", str(tmp_path), "--quiet"])
    rep = json.loads((out / "diag.json").read_text(encoding="utf-8"))
    by = {r["run"]: r for r in rep["runs"]}
    fcfs, srpt, srpt6 = (by[str(p)] for p in runs)
    assert fcfs["policy"] == "fcfs" and fcfs["dispatch"]["reorder_vs_call_dispatches"] == 0
    assert fcfs["dispatch"]["queue_reconstruction_mismatch"] == 0
    assert fcfs["dispatch"]["choice_ge2"] > 0.5  # burst + K=1 -> 대기열 생김
    assert not fcfs["estimates"]["available"]
    assert srpt["dispatch"]["reorder_vs_call_dispatches"] > 0
    assert srpt["estimates"]["available"] and srpt["estimates"]["spearman_vs_actual"] is not None
    assert srpt6["dispatch"]["choice_ge2"] < srpt["dispatch"]["choice_ge2"]  # K 가 크면 기회 감소
    assert srpt["conditions"]["cost_model_type"] == "oof"
    assert srpt["conditions"]["leakage_check"] == "oof"
    assert any("K 불일치" in w for w in rep["warnings"])
    assert not any("재정렬" in w for w in rep["warnings"])
    assert not any("복원 불일치" in w for w in rep["warnings"])
    with open(out / "per_run.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["policy"] for r in rows] == ["fcfs", "srpt_est", "srpt_est"]
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "choice>=2" in md and "조건 확인" in md


def test_diag_sched_without_scheduler(policy_runs, tmp_path):
    out = diag_sched.main(
        [
            str(policy_runs["none"]),
            str(policy_runs["fcfs"]),
            "--name",
            "n",
            "--out-root",
            str(tmp_path),
            "--quiet",
        ]
    )
    rep = json.loads((out / "diag.json").read_text(encoding="utf-8"))
    none = next(r for r in rep["runs"] if r["policy"] == "none")
    assert none["dispatch"] == {"available": False} and none["K"] is None
    assert none["estimates"]["available"] is False


def test_fcfs_query_run_keeps_arrival_order(policy_runs, tmp_path):
    out = diag_sched.main(
        [
            str(policy_runs["fcfsq"]),
            str(policy_runs["fcfs"]),
            "--name",
            "q",
            "--out-root",
            str(tmp_path),
            "--quiet",
        ]
    )
    rep = json.loads((out / "diag.json").read_text(encoding="utf-8"))
    q = next(r for r in rep["runs"] if r["policy"] == "fcfs_query")
    f = next(r for r in rep["runs"] if r["policy"] == "fcfs")
    assert q["dispatch"]["reorder_vs_arrival_dispatches"] == 0  # 정책 정의대로
    assert q["dispatch"]["queue_reconstruction_mismatch"] == 0
    assert f["dispatch"]["reorder_vs_call_dispatches"] == 0
    assert not any("재정렬" in w for w in rep["warnings"])
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "reorder call" in md and "reorder arrival" in md
