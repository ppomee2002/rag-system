"""analyze_sched_validation.py 기능 검증 (가짜 엔진 run, 실제 성능 결과 아님)."""

import csv
import json

import analyze_sched_validation as asv
import pytest
from crag_fixtures import FakeEngine, Retriever, Web, pool_ids
from runner import Components, Runner
from workload import make_workload


@pytest.fixture
def sv_runs(setup):
    """같은 burst workload 로 none / fcfs / srpt_nopath / srpt_est (K=1) 를 돌린 가짜 run."""
    _, make = setup
    trace = Runner(*make(), run_id="trace").run()
    cm = "{trace: '" + trace.as_posix() + "', service_key: wall, oof_folds: 3}"
    out = {"trace": trace}
    wl = None
    for name in ("none", "fcfs", "srpt_nopath", "srpt_est"):
        cfg, _ = make(
            [
                "run.mode=live",
                "exp_id=PRE-SV",
                "run.purpose=pre",
                f"scheduler.policy={name}",
                "scheduler.slots=1",
                f"scheduler.cost_model={cm}",
                "run.sample_engine_metrics_sec=0.05",
            ]
        )
        wl = wl or make_workload(pool_ids(cfg)[1], 10, seed=0, rate=2000.0, name="burst")
        out[name] = Runner(
            cfg, Components(Retriever(), Web(), FakeEngine(delay=0.01)), workload=wl, run_id=name
        ).run()
    return out


def test_gpu_util_parses_nvidia_smi_csv(tmp_path):
    p = tmp_path / "gpu.csv"
    p.write_text(
        "timestamp, index, utilization.gpu [%], memory.used [MiB]\n"
        "2026/11/14 10:00:00.000, 0, 80, 1000\n"
        "2026/11/14 10:00:00.500, 0, 100, 1000\n"
        "2026/11/14 10:00:00.500, 1, 0, 3\n",
        encoding="utf-8",
    )
    g = asv.gpu_util(p)
    assert g["0"]["mean"] == pytest.approx(90.0) and g["0"]["n"] == 2
    assert g["1"]["mean"] == 0.0
    assert asv.gpu_util(tmp_path / "missing.csv") is None


def test_end_to_end(sv_runs, tmp_path):
    gpu = tmp_path / "gpu.csv"
    gpu.write_text("t, 0, 50, 1\nt, 0, 70, 1\n", encoding="utf-8")
    args = []
    for k in ("none", "fcfs", "srpt_nopath", "srpt_est"):
        args += ["--run", f"{k}={sv_runs[k]}"]
    out = asv.main(
        [*args, "--ref", "fcfs", "--gpu-log", f"fcfs={gpu}", "--name", "t",
         "--out-root", str(tmp_path / "ana"), "--quiet"]
    )  # fmt: skip
    res = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    by = {r["run"]: r for r in res["per_run"]}
    assert by["none"]["policy"] == "none" and by["none"]["gate_wait_mean"] == 0.0
    assert by["fcfs"]["gate_wait_mean"] > 0  # burst + K=1
    assert by["fcfs"]["gpu_util"] == pytest.approx(60.0) and by["srpt_est"]["gpu_util"] is None
    assert by["srpt_nopath"]["policy"] == "srpt_nopath"
    assert all(r["n"] == 10 and r["errors"] == 0 for r in res["per_run"])
    pa = [r for r in res["paired"] if r["subset"] == "all"]
    assert {r["run"] for r in pa} == {"none", "srpt_nopath", "srpt_est"}
    for r in pa:
        assert r["n_pairs"] == 10 and r["mean_ci_lo"] <= r["mean_diff"] <= r["mean_ci_hi"]
    assert any("K 불일치" in w for w in res["warnings"])  # none 은 K 없음 (정상)
    paths = {(r["run"], r["path"]): r for r in res["paths"]}
    allp = paths[("fcfs", "all")]
    assert allp["n"] == 10 and allp["share"] == 1.0 and allp["llm_calls"] > 0
    with open(out / "paired.csv", encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == len(res["paired"])
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "실험 A" in md and "paired" in md
    with pytest.raises(FileExistsError):  # 같은 폴더 덮어쓰기 금지
        (out).mkdir(parents=True, exist_ok=False)


def test_paths_only_and_bad_ref(sv_runs, tmp_path):
    out = asv.main(
        ["--run", f"seq={sv_runs['trace']}", "--paths-only", "--name", "a",
         "--out-root", str(tmp_path), "--quiet"]
    )  # fmt: skip
    res = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert "per_run" not in res and res["paths"]
    assert not (out / "per_run.csv").exists()
    with pytest.raises(SystemExit):
        asv.analyze({"a": str(sv_runs["fcfs"])}, ref="zzz")
    with pytest.raises(SystemExit):
        asv.parse_pairs(["noequals"], "--run")
