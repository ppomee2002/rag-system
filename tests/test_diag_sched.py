"""diag_sched.py 기능 검증 (가짜 엔진 run, 실제 성능 결과 아님)."""

import csv
import json

import diag_sched
import pytest
from crag_fixtures import FakeEngine, Retriever, Web, pool_ids
from runner import Components, Runner
from workload import make_workload


def log(seqs, n_waiting, key="0:q"):
    return [
        {"seq": s, "n_waiting": w, "wait": 0.1 * w, "request_key": key}
        for s, w in zip(seqs, n_waiting, strict=True)
    ]


def test_reorder_counts_from_gate_sequence():
    # seq 3 이 seq 2 보다 먼저 나감 -> 1 회 재정렬 (2 를 앞지름)
    m = diag_sched.dispatch_metrics(log([0, 1, 3, 2, 4], [0, 2, 1, 0, 0]), {"0:q"})
    assert m["reorder_dispatches"] == 1 and m["skipped_mean"] == 1.0
    assert m["queue_len_hist"] == {"1": 3, "2": 1, "3": 1}
    assert m["choice_ge2"] == pytest.approx(2 / 5)
    assert m["reorder_rate_given_choice"] == pytest.approx(1 / 2)
    # 순서대로면 재정렬 0 (FCFS 검증 기준)
    assert (
        diag_sched.dispatch_metrics(log([0, 1, 2, 3], [3, 2, 1, 0]), {"0:q"})["reorder_dispatches"]
        == 0
    )
    # 여러 개를 앞지르는 경우
    m = diag_sched.dispatch_metrics(log([3, 0, 1, 2], [3, 2, 1, 0]), {"0:q"})
    assert m["reorder_dispatches"] == 1 and m["skipped_mean"] == 3.0
    assert diag_sched.dispatch_metrics([], set()) == {"available": False}


def test_warmup_dispatches_excluded_but_tracked():
    entries = log([0, 1], [0, 0], key="-1:w") + log([3, 2], [1, 0], key="0:q")
    m = diag_sched.dispatch_metrics(entries, {"0:q"})
    assert m["n_dispatch"] == 2 and m["reorder_dispatches"] == 1


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
    assert fcfs["policy"] == "fcfs" and fcfs["dispatch"]["reorder_dispatches"] == 0
    assert fcfs["dispatch"]["choice_ge2"] > 0.5  # burst + K=1 -> 대기열 생김
    assert not fcfs["estimates"]["available"]
    assert srpt["dispatch"]["reorder_dispatches"] > 0
    assert srpt["estimates"]["available"] and srpt["estimates"]["spearman_vs_actual"] is not None
    assert srpt6["dispatch"]["choice_ge2"] < srpt["dispatch"]["choice_ge2"]  # K 가 크면 기회 감소
    assert srpt["conditions"]["cost_model_type"] == "oof"
    assert srpt["conditions"]["leakage_check"] == "oof"
    assert any("K 불일치" in w for w in rep["warnings"])
    assert not any("FCFS 인데 재정렬" in w for w in rep["warnings"])
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
