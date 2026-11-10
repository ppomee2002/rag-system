"""공통 runner / workload / web 기록·재생 기능 검증 (가짜 엔진, 성능·정확도 결과 아님)."""

import json
import sys
from pathlib import Path

import pytest

from rag.retrieval.web_cache import RecordingWebSearch, ReplayWebSearch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "crag_system"))
import make_workload  # noqa: E402
import run as run_cli  # noqa: E402
from crag_fixtures import ROOT, Web, pool_ids, read_jsonl  # noqa: E402
from runner import (  # noqa: E402
    Runner,
    load_layered_config,
    load_trace,
)
from workload import load_workload, resolve_rate  # noqa: E402
from workload import make_workload as make_wl


def test_layered_config_and_set():
    cfg = load_layered_config(
        ROOT / "experiments/crag_system/config.yaml",
        [],
        ["grading.strategy=batch", "scheduler.slots=4", "new.key=[1, 2]"],
    )
    assert cfg["grading"]["strategy"] == "batch" and cfg["scheduler"]["slots"] == 4
    assert cfg["new"]["key"] == [1, 2] and cfg["seed"] == 42  # base.yaml 유지


@pytest.mark.parametrize(
    "pipeline,strategy",
    [
        ("crag_graph", "full"),
        ("crag_graph", "batch"),
        ("crag_graph", "lazy_hit"),
        ("crag_graph", "adaptive"),
        ("crag", "full"),
        ("vanilla", "full"),
    ],
)
def test_sequential_run_writes_results(setup, pipeline, strategy):
    tmp, make = setup
    cfg, comp = make([f"pipeline={pipeline}", f"grading.strategy={strategy}"])
    run_dir = Runner(cfg, comp, run_id="r").run()
    q = read_jsonl(run_dir / "queries.jsonl")
    assert len(q) == 12 and sum(r["warmup"] for r in q) == 2
    assert not any(r.get("error") for r in q)
    calls = read_jsonl(run_dir / "calls.jsonl")
    assert {c["request_index"] for c in calls} == {r["request_index"] for r in q}
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "sequential" and meta["n_warmup"] == 2 and meta["n_requests"] == 10
    assert meta["engines"]["main"] == {"engine": "fake"}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_ok"] == 10 and summary["accuracy"] == 1.0
    assert (run_dir / "config.yaml").exists() and (run_dir / "run_end.json").exists()
    if pipeline != "vanilla":
        measured = [r for r in q if not r["warmup"]]
        assert {r["corrective_action"] for r in measured} == {"correct", "incorrect"}
    with pytest.raises(FileExistsError):  # 같은 run_id 는 덮어쓰지 않음
        Runner(cfg, comp, run_id="r")


def test_batch_strategy_reduces_grade_calls(setup):
    _, make = setup
    full_dir = Runner(*make(["grading.strategy=full"]), run_id="full").run()
    batch_dir = Runner(*make(["grading.strategy=batch"]), run_id="batch").run()
    kinds = lambda d: [c["kind"] for c in read_jsonl(d / "calls.jsonl") if not c["warmup"]]  # noqa: E731
    assert kinds(full_dir).count("grade") == 30 and "grade_batch" not in kinds(full_dir)
    assert kinds(batch_dir).count("grade_batch") == 10 and "grade" not in kinds(batch_dir)


def test_workload_generation_and_reuse(setup):
    tmp, make = setup
    cfg, _ = make()
    out = make_workload.main(
        [
            "--config",
            str(ROOT / "experiments/crag_system/config.yaml"),
            "--overlay",
            str(tmp / "overlay.yaml"),
            "--name",
            "wl",
            "--n",
            "15",
            "--rate",
            "200",
            "--seed",
            "3",
            "--out-dir",
            str(tmp / "wl"),
        ]
    )
    wl = load_workload(out)
    assert len(wl["requests"]) == 15 and wl["params"]["rate_basis"] == "rate"
    pool = {r["query_id"] for r in wl["requests"]}
    assert pool == set(pool_ids(cfg)[1]) and len(pool) == 10  # warmup 2 제외
    again = make_wl(wl["query_pool"], 15, seed=3, rate=200)
    assert again["sha256"] == wl["sha256"]  # 같은 seed -> 같은 패턴
    with pytest.raises(FileExistsError):
        make_workload.main(
            [
                "--overlay",
                str(tmp / "overlay.yaml"),
                "--name",
                "wl",
                "--rate",
                "1",
                "--out-dir",
                str(tmp / "wl"),
            ]
        )
    data = json.loads(out.read_text(encoding="utf-8"))
    data["requests"][0]["arrival"] = 999
    out.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_workload(out)


def test_resolve_rate_bases():
    assert resolve_rate(rate=2.0) == (2.0, "rate")
    assert resolve_rate(rho=0.5, capacity_rps=4.0) == (2.0, "rho_capacity")
    assert resolve_rate(rho=0.8, mean_work=4.0) == (0.2, "rho_mean_work")
    with pytest.raises(ValueError):
        resolve_rate(rho=0.5)


@pytest.mark.parametrize(
    "sched",
    [
        ["scheduler.policy=fcfs", "scheduler.slots=2"],
        ["scheduler.policy=fcfs", "scheduler.slots=1", "scheduler.aging_alpha=0.5"],
        ["scheduler.policy=fcfs", "scheduler.slots=1", "scheduler.wait_threshold=0.01"],
    ],
)
def test_live_run_with_scheduler(setup, sched):
    tmp, make = setup
    cfg, comp = make(["run.mode=live", *sched])
    wl = make_wl(pool_ids(cfg)[1], 12, seed=0, rate=300.0, name="t")
    run_dir = Runner(cfg, comp, workload=wl, run_id="live").run()
    rows = read_jsonl(run_dir / "queries.jsonl")
    q = [r for r in rows if not r["warmup"]]
    assert len(q) == 12 and not any(r.get("error") for r in q)
    assert all(r["latency"] >= 0 and r["arrival"] is not None for r in q)
    assert comp.engine.max_active <= cfg["scheduler"]["slots"]
    log = read_jsonl(run_dir / "scheduler_log.jsonl")
    assert len(log) == sum(r["calls_summary"]["n_calls"] for r in rows)  # warmup 포함 모든 호출
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["workload"]["sha256"] == wl["sha256"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["throughput_rps"] > 0 and summary["latency_corrected"]["n"] > 0


def test_srpt_from_sequential_trace_and_engine_priority(setup):
    _, make = setup
    seq_dir = Runner(*make(), run_id="trace").run()
    rows = load_trace(seq_dir)
    assert len(rows) == 10 and all(r["calls"] for r in rows)
    cfg, comp = make(
        [
            "run.mode=live",
            "scheduler.policy=srpt_est",
            "scheduler.slots=1",
            "scheduler.engine_priority=key",
            "llm.send_priority=true",
            "llm.server_scheduling_policy=priority",
            # 같은 질문의 trace 로 학습하므로 OOF 필요 (아니면 누수 검사에서 거부)
            f"scheduler.cost_model={{trace: '{seq_dir.as_posix()}', service_key: wall, "
            "oof_folds: 5}",
        ]
    )
    wl = make_wl(pool_ids(cfg)[1], 10, seed=1, rate=300.0)
    run_dir = Runner(cfg, comp, workload=wl, run_id="srpt").run()
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["cost_model"]["type"] == "oof"
    assert meta["preflight"]["cost_model_leakage"]["status"] == "oof"
    assert meta["preflight"]["priority"]["send_priority"] is True
    assert meta["scheduler"]["policy"]["name"] == "srpt_est"
    assert all(isinstance(p, int) for p in comp.engine.priorities)


def test_live_requires_workload(setup):
    _, make = setup
    with pytest.raises(ValueError, match="workload"):
        Runner(*make(["run.mode=live"]), run_id="x")


def test_workload_must_match_pool(setup):
    _, make = setup
    cfg, comp = make()
    warm, pool = pool_ids(cfg)
    wl = make_wl([warm[0], pool[0]], 2, arrival="sequential")  # warmup query 는 pool 밖
    with pytest.raises(ValueError, match="pool"):
        Runner(cfg, comp, workload=wl, run_id="x")


def test_web_record_and_replay(tmp_path):
    web = Web()
    rec = RecordingWebSearch(web, tmp_path / "web.jsonl")
    docs = rec.search(" who wrote hamlet ", 2)
    slept = []
    replay = ReplayWebSearch([tmp_path / "web.jsonl"], "recorded", sleep=slept.append)
    again = replay.search("who wrote hamlet", 2)
    assert [d.id for d in again] == [d.id for d in docs] and len(slept) == 1
    zero = ReplayWebSearch([tmp_path / "web.jsonl"], "zero", sleep=slept.append)
    zero.search("who wrote hamlet", 2)
    assert len(slept) == 1 and zero.stats()["hits"] == 1
    with pytest.raises(KeyError):
        zero.search("unknown", 2)
    fb = ReplayWebSearch([tmp_path / "web.jsonl"], "zero", fallback=web)
    fb.search("unknown", 2)
    assert fb.stats()["misses"] == 1 and web.n == 2


def test_cli_run(setup, monkeypatch):
    tmp, make = setup
    cfg, comp = make()
    monkeypatch.setattr(
        run_cli,
        "Runner",
        lambda c, workload=None, run_id=None: __import__("runner").Runner(
            c, comp, workload=workload, run_id=run_id
        ),
    )
    run_dir = run_cli.main(
        [
            "--overlay",
            str(tmp / "overlay.yaml"),
            "--exp-id",
            "SYS-test",
            "--set",
            "grading.strategy=lazy_hit",
            "--limit",
            "6",
        ]
    )
    assert run_dir.parent.name == "SYS-test"
    assert len(read_jsonl(run_dir / "queries.jsonl")) == 6


def test_reference_pipelines_reject_non_full_grading(setup):
    _, make = setup
    with pytest.raises(ValueError, match="crag_graph"):
        Runner(*make(["pipeline=crag", "grading.strategy=batch"]), run_id="x")


def test_invalid_config_leaves_no_run_dir(setup):
    tmp, make = setup
    for sets in (
        ["pipeline=crag", "grading.strategy=batch"],
        ["run.mode=live"],
        ["scheduler.policy=srpt_est"],
    ):
        with pytest.raises(ValueError):
            Runner(*make(sets), run_id="bad")
    assert not (tmp / "results" / "crag_system" / "SYS-dev" / "bad").exists()
