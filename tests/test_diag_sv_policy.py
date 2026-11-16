"""diag_sv_policy.py / paired_bootstrap_sv.py 기능 검증 (가짜 엔진 run, 실제 성능 결과 아님)."""

import csv
import json

import diag_sv_policy as dsp
import numpy as np
import paired_bootstrap_sv as pb
import pytest
from crag_fixtures import FakeEngine, Retriever, Web, pool_ids
from runner import Components, Runner
from workload import make_workload

POLS = ("none", "fcfs", "fcfs_query", "srpt_nopath", "srpt_est")


@pytest.fixture
def grid(setup):
    """seed 2개 x 정책 5개 (K=1, burst) 를 run_sched_validation.sh 와 같은 폴더 이름으로."""
    tmp, make = setup
    trace = Runner(*make(), run_id="trace").run()
    cm = "{trace: '" + trace.as_posix() + "', service_key: wall, oof_folds: 3}"
    for s in (0, 1):
        wl = None
        for pol in POLS:
            cfg, _ = make(
                ["run.mode=live", "exp_id=PRE-SV", "run.purpose=pre",
                 f"scheduler.policy={pol}", "scheduler.slots=1", f"scheduler.cost_model={cm}"]
            )  # fmt: skip
            wl = wl or make_workload(pool_ids(cfg)[1], 10, seed=s, rate=2000.0, name=f"b{s}")
            Runner(
                cfg, Components(Retriever(), Web(), FakeEngine(delay=0.01)), workload=wl,
                run_id=f"rho0.8_s{s}_{pol}_K1",
            ).run()  # fmt: skip
    return tmp / "results" / "crag_system" / "pre" / "PRE-SV", trace


def test_diag_end_to_end(grid, tmp_path):
    root, trace = grid
    out = dsp.main(
        ["--live-root", str(root), "--seq", str(trace), "--k", "1", "--rhos", "0.8",
         "--seeds", "0", "1", "--name", "t", "--out-root", str(tmp_path / "ana"), "--quiet"]
    )  # fmt: skip
    res = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert [c["cond"] for c in res["conditions"]] == ["rho0.8_s0", "rho0.8_s1"]
    assert res["missing"] == [p for p in res["missing"] if "sjf_pred" in p]  # sjf_pred 만 없음
    for c in res["conditions"]:
        # 오프라인 재계산 key 가 기록된 dispatch key 와 같아야 함
        for lb in ("srpt_est", "srpt_nopath"):
            kc = c["key_check"][lb]
            assert kc["checked"] > 0 and kc["n_mismatch_gt_1e-6"] == 0
        ch = c["choices"]
        # 자기 정책 규칙과 실제 선택은 항상 일치
        for lb in ("fcfs", "fcfs_query", "srpt_est", "srpt_nopath"):
            if ch[lb]["n_with_choice"]:
                assert ch[lb]["agree_with_actual"][lb] == pytest.approx(1.0)
        assert c["dispatch"]["fcfs_query"]["reorder_vs_arrival_dispatches"] == 0
        assert "none" not in c["dispatch"]  # gate 없음
    with open(out / "request_breakdown.csv", encoding="utf-8-sig") as f:
        bd = list(csv.DictReader(f))
    assert {r["run"] for r in bd} == set(POLS)
    for r in bd:  # 구성 요소 합 = latency
        parts = sum(float(r[c]) for c in dsp.COMPONENTS)
        assert parts == pytest.approx(float(r["latency"]), abs=1e-6)
    fq = [r for r in bd if r["run"] == "fcfs_query"]
    assert all(r["overtaken_by_later"] == "0" for r in fq)
    with open(out / "decomposition.csv", encoding="utf-8-sig") as f:
        dec = list(csv.DictReader(f))
    assert {(r["base"], r["target"]) for r in dec} == {
        ("fcfs_query", "srpt_est"),
        ("srpt_nopath", "srpt_est"),
    }
    for r in dec:
        s = sum(float(r[f"d_{c}"]) for c in dsp.COMPONENTS)
        assert s == pytest.approx(float(r["d_latency"]), abs=1e-6)
    assert (out / "estimates.csv").exists()
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "로그 한계" in md and "악화 원인" in md
    with pytest.raises(FileExistsError):
        out.mkdir(parents=True, exist_ok=False)


def test_true_remaining_skips_path_mismatch():
    seq = {"q": {"corrected": False, "calls": [("grade", 0, 1.0), ("grade", 1, 2.0),
                                               ("generate", 0, 3.0)]}}  # fmt: skip
    assert dsp.true_remaining(seq, "q", "grade", 1, False) == 5.0
    assert dsp.true_remaining(seq, "q", "grade", 1, True) is None  # 경로 다르면 추정 안 함
    assert dsp.true_remaining(seq, "q", "rewrite", 0, False) is None
    assert dsp.true_remaining(seq, "zz", "grade", 0, False) is None
    assert dsp.decided_at([0.1, 0.8, 0.9], 0.7) == 1
    assert dsp.decided_at([0.1, None], 0.7) is None


def test_bootstrap(grid, tmp_path):
    root, _ = grid
    out = pb.main(
        ["--live-root", str(root), "--k", "1", "--rhos", "0.8", "--seeds", "0", "1",
         "--n-boot", "300", "--block", "3", "--name", "b", "--out-root", str(tmp_path), "--quiet"]
    )  # fmt: skip
    rows = json.loads((out / "bootstrap.json").read_text(encoding="utf-8"))["rows"]
    groups = {(r["pair"], r["group"], r["method"]) for r in rows if r["subset"] == "all"}
    for pair in ("fcfs_query:srpt_est", "srpt_nopath:srpt_est"):
        for g in ("rho0.8_s0", "rho0.8_s1", "rho0.8_pooled", "all_pooled"):
            for m in ("iid", "block"):
                assert (pair, g, m) in groups
    for r in rows:
        assert r["mean_lo"] <= r["mean_hi"] and r["p95_lo"] <= r["p95_hi"]
    pooled = [r for r in rows if r["group"] == "rho0.8_pooled" and r["subset"] == "all"]
    assert all(r["n_pairs"] == 20 for r in pooled)
    with pytest.raises(SystemExit):
        pb.main(["--live-root", str(root), "--pairs", "fcfs:bogus", "--out-root", str(tmp_path)])


def test_boot_known_shift():
    rng = np.random.default_rng(1)
    a = rng.exponential(10, 400)
    b = a + 2.0  # 모든 요청이 정확히 2초 느림
    r = pb.boot([(a, b)], 500, 0, "block", 20)
    assert r["mean_diff"] == pytest.approx(2.0)
    assert r["mean_lo"] == pytest.approx(2.0) and r["mean_hi"] == pytest.approx(2.0)
    idx = pb.resample_idx(10, 5, np.random.default_rng(0), "block", 4)
    assert idx.shape == (5, 10) and idx.max() < 10
