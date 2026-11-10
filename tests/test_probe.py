"""probe_engine.py (SMOKE-P) 기능 검증. mock 서버 사용, 성능·정확도 결과 아님."""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "server"))
import probe_engine  # noqa: E402


def write_cfg(tmp_path, llm, extra=None, name="engine.yaml"):
    p = tmp_path / name
    p.write_text(
        yaml.safe_dump({"experiment": "crag_system", "llm": llm, **(extra or {})}), encoding="utf-8"
    )
    return str(p)


def vllm_llm(srv, **kw):
    return {
        "provider": "vllm",
        "model": "mock-model",
        "base_url": srv.url + "/v1",
        "api_key_env": None,
        **kw,
    }


def run_probe(cfg, tmp_path, *args):
    # run 폴더는 초 단위 timestamp 이고 덮어쓰지 않으므로 호출마다 다른 output 폴더 사용
    out = tmp_path / f"out{len(list(tmp_path.glob('out*')))}"
    run_dir = probe_engine.main(["--config", cfg, "--output-dir", str(out), *args])
    return run_dir, json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


def test_probe_vllm_basic_and_crag_sweep(mock_server, tmp_path, monkeypatch):
    srv = mock_server(
        "vllm",
        api_key="k",
        responder=lambda p: "0.8" if p.startswith("Is the following") else "Paris",
    )
    monkeypatch.setenv("PROBE_KEY", "k")
    cfg = write_cfg(tmp_path, vllm_llm(srv, api_key_env="PROBE_KEY"))
    run_dir, report = run_probe(cfg, tmp_path, "--sweep", "1,2", "--sweep-requests", "1")
    assert report["purpose"] == "smoke"
    assert report["smoke"]["text"] == "Paris"
    assert report["determinism"]["identical"] is True
    assert report["grade_format"]["parsed"] == 0.8
    assert report["auth"]["rejected_without_key"] is True
    assert report["metrics"]["expected_present"]["waiting"] is True
    rows = report["sweep"]
    assert [r["K"] for r in rows] == [1, 2] and rows[1]["n_units"] == 2
    assert rows[1]["by_kind"]["grade"]["n"] == 10 and rows[1]["by_kind"]["generate"]["n"] == 2
    assert rows[0]["by_kind"]["grade"]["mean_prompt_tokens"] > 250  # 실제 chunk 길이 수준
    assert rows[0]["vllm"]["n_samples"] >= 1
    assert run_dir.parent.name == "SMOKE-P" and run_dir.parent.parent.name == "smoke"
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["engines"]["main"]["engine"] == "vllm"


def test_crag_units_are_unique_and_sized():
    a = probe_engine.crag_unit(1, 2000, 5)
    b = probe_engine.crag_unit(2, 2000, 5)
    assert len(a["grade"]) == 5 and a["grade"][0] != b["grade"][0]
    assert 1900 < len(a["grade"][0]) < 2400 and len(a["generate"]) > 9500


@pytest.mark.parametrize(
    "server_policy,declared,verdict,consistent",
    [
        ("priority", "priority", "priority_order", True),
        ("fcfs", "fcfs", "fcfs_order", True),
        ("fcfs", "priority", "fcfs_order", False),  # 선언과 실제 서버가 다름 -> 검출
    ],
)
def test_priority_order_by_processing_order(
    mock_server, tmp_path, server_policy, declared, verdict, consistent
):
    # 0.33.0 처럼 FCFS 서버도 priority 를 수락 (fcfs_rejects_priority=False), slot 1개
    srv = mock_server(
        "vllm", sched_policy=server_policy, fcfs_rejects_priority=False, slots=1, service_time=0.1
    )
    cfg = write_cfg(tmp_path, vllm_llm(srv, server_scheduling_policy=declared))
    _, report = run_probe(cfg, tmp_path, "--priority-order", "5", "--priority-blockers", "1")
    po = report["priority_order"]
    assert po["slots_full_before_test"] is True
    assert po["verdict"] == verdict and po["consistent_with_declared"] is consistent
    assert po["arrival_order"] == [0, 1, 2, 3, 4] and po["priority_order"] == [4, 3, 2, 1, 0]
    assert len(po["requests"]) == 5 and all(r["error"] is None for r in po["requests"])
    if verdict == "priority_order":
        assert po["observed_start_order"] == [4, 3, 2, 1, 0]
        assert po["mean_wait_high_priority_s"] < po["mean_wait_low_priority_s"]
    assert report["priority_api"]["priority_1"] == "accepted"


def test_kendall_tau():
    assert probe_engine.kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert probe_engine.kendall_tau([3, 2, 1], [1, 2, 3]) == -1.0
    assert probe_engine.kendall_tau([1, 3, 2], [1, 2, 3]) == pytest.approx(1 / 3)


def test_probe_checks_launch_meta(mock_server, tmp_path):
    srv = mock_server("vllm")
    meta = tmp_path / "x_launch.json"
    meta.write_text(
        json.dumps(
            {
                "argv": [
                    "vllm",
                    "serve",
                    "mock-model",
                    "--tensor-parallel-size",
                    "1",
                    "--scheduling-policy",
                    "fcfs",
                    "--max-num-seqs",
                    "4",
                    "--max-model-len",
                    "6144",
                ],
                "cuda_visible_devices": "0",
                "profile": {"name": "p1"},
            }
        ),
        encoding="utf-8",
    )
    server = {"tensor_parallel_size": 1, "max_model_len": 6144}
    ok = write_cfg(
        tmp_path,
        vllm_llm(srv, server_scheduling_policy="fcfs", launch_meta=str(meta)),
        {"engine_profile": "p1", "server": server},
    )
    _, report = run_probe(ok, tmp_path)
    assert report["launch_meta"]["max_num_seqs"] == "4"
    bad = write_cfg(
        tmp_path,
        vllm_llm(srv, server_scheduling_policy="fcfs", launch_meta=str(meta)),
        {"engine_profile": "p1", "server": {**server, "tensor_parallel_size": 2}},
        name="bad.yaml",
    )
    _, report = run_probe(bad, tmp_path)
    assert "error" in report["launch_meta"]


def test_probe_ollama_against_mock(mock_server, tmp_path):
    srv = mock_server("ollama", model="llama3.1:8b")
    cfg = write_cfg(
        tmp_path, {"provider": "ollama", "model": "llama3.1:8b", "num_ctx": 6144, "host": srv.url}
    )
    run_dir, report = run_probe(cfg, tmp_path)
    assert report["smoke"]["server_total"] == 2.0
    assert "auth" not in report
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["engines"]["main"]["model_info"]["details"]["quantization_level"] == "Q4_K_M"
