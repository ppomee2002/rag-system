"""equal_load.py 기능 검증 (가짜 엔진 run, 실제 성능 결과 아님)."""

import json

import equal_load as eql
import numpy as np
import pytest
import run_evidence_live as rel
from crag_fixtures import Retriever, Web
from runner import Components, Runner, apply_split, load_examples, split_warmup
from test_diag_evc_live import DocEngine
from workload import load_workload, make_workload, requests_sha, save_workload


@pytest.fixture
def src(setup, tmp_path):
    tmp, make = setup
    cfg, _ = make(["dataset.split.part=eval"])
    _, pool = split_warmup(cfg, load_examples(cfg))
    ids = [ex.id for ex in apply_split(cfg, pool)]
    wld = tmp_path / "wl"
    p = save_workload(make_workload(ids, None, seed=0, rho=0.8, capacity_rps=250.0,
                                    name="SV1_rho0.8_s0"), wld / "SV1_rho0.8_s0.json")  # fmt: skip
    return make, p, wld


def test_make_and_check(src, tmp_path):
    _, p, wld = src
    out = eql.main(["make", "--source", str(p), "--work-ratio", "1.5", "--target-rho-eff", "0.8",
                    "--tag", "EQT", "--out-dir", str(wld), "--k", "2"])  # fmt: skip
    new = load_workload(out[0])
    old = load_workload(p)
    assert out[0].name == "EQT_keep_all_rhoeff0.8_s0.json"
    assert [r["query_id"] for r in new["requests"]] == [r["query_id"] for r in old["requests"]]
    assert np.allclose([r["arrival"] for r in new["requests"]],
                       [1.5 * r["arrival"] for r in old["requests"]])  # fmt: skip
    assert new["params"]["rate"] == pytest.approx(200.0 / 1.5)
    assert new["params"]["rho"] == pytest.approx(0.8 / 1.5)
    assert new["equal_load"]["rho_eff_est"] == pytest.approx(0.8)
    assert eql.run_id(new, "keep_all", 2) == "rho0.533_s0_keep_all_K2"
    # 같은 내용이면 재사용, 다른 비율이면 거부 (덮어쓰기 금지)
    eql.main(["make", "--source", str(p), "--work-ratio", "1.5", "--tag", "EQT", "--out-dir",
              str(wld)])  # fmt: skip
    with pytest.raises(SystemExit):
        eql.main(["make", "--source", str(p), "--work-ratio", "1.4", "--tag", "EQT", "--out-dir",
                  str(wld)])  # fmt: skip
    live = tmp_path / "live"
    rc = eql.main(["check", "--workload", str(out[0]), "--live-root", str(live), "--k", "2"])
    assert rc == 0
    res = eql.check_one(out[0], type("A", (), {"live_root": None, "seq": None, "evf": None,
                                               "k": 2})())  # fmt: skip
    assert res["regenerated_same_arrivals"] and res["regenerated_same_order"]
    assert res["max_rel_dev"] < 1e-12
    # 도착 시각 하나를 바꾼 workload (sha 는 다시 맞춤) -> 실패
    bad = json.loads(out[0].read_text(encoding="utf-8"))
    bad["requests"][2]["arrival"] *= 1.01
    bad["sha256"] = requests_sha(bad["requests"])
    bp = wld / "bad.json"
    bp.write_text(json.dumps(bad), encoding="utf-8")
    assert eql.main(["check", "--workload", str(bp)]) == 1
    # 중단된 run 폴더 (summary 없음) 가 있으면 실패
    (live / "rho0.533_s0_keep_all_K2").mkdir(parents=True)
    assert eql.main(["check", "--workload", str(out[0]), "--live-root", str(live), "--k",
                     "2"]) == 1  # fmt: skip


def test_report(src, tmp_path):
    make, p, wld = src
    out = eql.main(["make", "--source", str(p), "--work-ratio", "1.5", "--tag", "EQT",
                    "--out-dir", str(wld)])  # fmt: skip
    cfg, _ = make(["dataset.split.part=eval"])
    seq = Runner(cfg, Components(Retriever(), Web(), DocEngine()), run_id="seq").run()
    for exp, wl_path, pols in (("PRE-REF", p, ("crag_rule", "keep_all")),
                               ("PRE-NEW", out[0], ("keep_all",))):  # fmt: skip
        wl = load_workload(wl_path)
        for pol in pols:
            rid = (eql.run_id(wl, pol, 2) if exp == "PRE-NEW"
                   else f"rho0.8_s0_{pol}_K2")  # fmt: skip
            cfg, _ = make(["run.mode=live", f"exp_id={exp}", "run.purpose=pre",
                           "dataset.split.part=eval", f"evidence_policy={pol}",
                           "scheduler.policy=fcfs_query", "scheduler.slots=2"])  # fmt: skip
            rel.EvidenceRunner(cfg, Components(Retriever(), Web(), DocEngine()), workload=wl,
                               run_id=rid).run()  # fmt: skip
    root = tmp_path / "results" / "crag_system" / "pre"
    o = eql.main(["report", "--new-root", str(root / "PRE-NEW"), "--ref-root",
                  str(root / "PRE-REF"), "--seq", str(seq), "--k", "2", "--n-boot", "50",
                  "--out-root", str(tmp_path / "ana"), "--quiet"])  # fmt: skip
    res = json.loads((o / "analysis.json").read_text(encoding="utf-8"))
    runs = {(r["label"], r["policy"]): r for r in res["runs"]}
    assert set(runs) == {("new", "keep_all"), ("ref", "crag_rule"), ("ref", "keep_all")}
    n = runs[("new", "keep_all")]
    assert n["workload_is_equal_load"] and n["n_errors"] == 0
    assert n["lambda_nominal"] == pytest.approx(200.0 / 1.5)
    assert n["rho_eff_obs_est"] is not None and n["work_ratio_vs_ref_crag_rule"] > 1
    assert {(x["new"], x["ref"]) for x in res["pairs"]} == {
        ("keep_all@rho0.533", "crag_rule@rho0.8"),
        ("keep_all@rho0.533", "keep_all@rho0.8"),
    }
    assert all(x["n_pairs"] == n["n"] for x in res["pairs"])
    assert "동일 추정 유효 부하" in (o / "report.md").read_text(encoding="utf-8")
