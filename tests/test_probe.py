import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "server"))
import probe_engine  # noqa: E402


def write_cfg(tmp_path, llm):
    p = tmp_path / "engine.yaml"
    p.write_text(yaml.safe_dump({"experiment": "crag_system", "llm": llm}), encoding="utf-8")
    return str(p)


def test_probe_vllm_against_mock(mock_server, tmp_path, monkeypatch):
    srv = mock_server(
        "vllm",
        api_key="k",
        sched_policy="priority",
        responder=lambda p: "0.8" if p.startswith("Is the following") else "Paris",
    )
    monkeypatch.setenv("PROBE_KEY", "k")
    cfg = write_cfg(
        tmp_path,
        {
            "provider": "vllm",
            "model": "mock-model",
            "base_url": srv.url + "/v1",
            "api_key_env": "PROBE_KEY",
        },
    )
    run_dir = probe_engine.main(
        [
            "--config",
            cfg,
            "--output-dir",
            str(tmp_path / "out"),
            "--priority-load",
            "2",
            "--sweep",
            "1,2",
            "--sweep-requests",
            "1",
        ]
    )
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["smoke"]["text"] == "Paris"
    assert report["determinism"]["identical"] is True
    assert report["grade_format"]["parsed"] == 0.8
    assert report["auth"]["rejected_without_key"] is True
    assert report["priority"]["priority_scheduling_likely_enabled"] is True
    assert report["metrics"]["expected_present"]["preemptions"] is True
    assert set(report["priority_order"]) >= {"probe_high", "probe_same"}
    assert [r["K"] for r in report["sweep"]] == [1, 2]
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["engines"]["main"]["engine"] == "vllm"
    assert "k" != meta["env"].get("PROBE_KEY")
    assert (run_dir / "calls.jsonl").read_text(encoding="utf-8").count("\n") >= 6
    assert run_dir.parent.name == "SYS-P"


def test_probe_vllm_fcfs_server_skips_priority_order(mock_server, tmp_path):
    srv = mock_server("vllm", sched_policy="fcfs")
    cfg = write_cfg(
        tmp_path,
        {
            "provider": "vllm",
            "model": "mock-model",
            "base_url": srv.url + "/v1",
            "api_key_env": None,
        },
    )
    run_dir = probe_engine.main(
        ["--config", cfg, "--output-dir", str(tmp_path / "out"), "--priority-load", "2"]
    )
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["priority"]["priority_1"].startswith("rejected (HTTP 400)")
    assert "skipped" in report["priority_order"]
    assert report["auth"]["rejected_without_key"] is False


def test_probe_ollama_against_mock(mock_server, tmp_path):
    srv = mock_server("ollama", model="llama3.1:8b")
    cfg = write_cfg(
        tmp_path, {"provider": "ollama", "model": "llama3.1:8b", "num_ctx": 6144, "host": srv.url}
    )
    run_dir = probe_engine.main(["--config", cfg, "--output-dir", str(tmp_path / "out")])
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["smoke"]["server_total"] == 2.0
    assert "auth" not in report
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["engines"]["main"]["model_info"]["details"]["quantization_level"] == "Q4_K_M"
