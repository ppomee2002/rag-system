"""launch_vllm.py / verify_launch.py 기능 검증 (실제 vLLM·GPU 없이, 가짜 프로세스·로그 사용)."""

import json
import sys
from pathlib import Path

import pytest

from rag.scheduling.preflight import PreflightError, check_launch_meta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "server"))
import launch_vllm  # noqa: E402
import verify_launch  # noqa: E402

PROFILES = ROOT / "experiments" / "crag_system" / "configs" / "engines"
AWQ = PROFILES / "vllm_awq_int4_1gpu.yaml"
BF16 = PROFILES / "vllm_bf16_tp2.yaml"
ENGINE_FIELDS = {
    "tensor_parallel_size",
    "dtype",
    "quantization",
    "max_model_len",
    "gpu_memory_utilization",
    "max_num_seqs",
    "generation_config",
    "seed",
    "scheduling_policy",
    "enable_prefix_caching",
    "enforce_eager",
}
INSTALLED = {
    "vllm": "0.33.0",
    "engine_fields": ENGINE_FIELDS,
    "frontend_fields": {"host", "port", "api_key"},
}


def gpus(n, busy=()):
    sel = [
        {
            "index": str(i),
            "uuid": f"GPU-{i}",
            "name": "NVIDIA A2",
            "memory_total_mib": 15356,
            "memory_used_mib": 0,
        }
        for i in range(n)
    ]
    return {"available": True, "all": sel, "selected": sel, "busy_processes": list(busy)}


@pytest.mark.parametrize("path", [AWQ, BF16])
def test_profiles_are_internally_consistent(path):
    p = launch_vllm.load_profile(path)
    assert p["llm"]["model"] == p["server"]["model"]
    assert p["server"]["n_gpus"] == p["server"]["tensor_parallel_size"]
    assert p["server"]["generation_config"] == "vllm"
    assert p["server"]["vllm_version"] == "0.33.0"
    assert set(p["estimates"]) >= {"weights_gib_per_gpu", "kv_cache_tokens"}


def test_build_argv_uses_profile_and_never_contains_api_key():
    awq = launch_vllm.load_profile(AWQ)
    argv = launch_vllm.build_argv(awq["server"], "fcfs", smoke=True)
    assert argv[:3] == ["vllm", "serve", awq["server"]["model"]]
    assert "--quantization" not in argv  # null -> 자동 감지
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    assert argv[argv.index("--generation-config") + 1] == "vllm"
    assert "--enforce-eager" in argv and "--enable-prefix-caching" in argv
    assert not any("key" in a.lower() for a in argv)
    bf16 = launch_vllm.load_profile(BF16)
    argv2 = launch_vllm.build_argv(bf16["server"], "priority", smoke=False)
    assert argv2[argv2.index("--dtype") + 1] == "bfloat16"
    assert argv2[argv2.index("--scheduling-policy") + 1] == "priority"
    assert "--enforce-eager" not in argv2
    with pytest.raises(launch_vllm.LaunchError):
        launch_vllm.build_argv(bf16["server"], "lifo", smoke=False)


@pytest.mark.parametrize(
    "profile,visible,ok",
    [
        (AWQ, "0", True),
        (AWQ, "1", True),
        (AWQ, "0,1", False),
        (BF16, "0,1", True),
        (BF16, "0", False),
        (BF16, "0,0", False),
        (AWQ, None, False),
    ],
)
def test_gpu_count_must_match_tp(profile, visible, ok):
    server = launch_vllm.load_profile(profile)["server"]
    errors = launch_vllm.check_gpu_assignment(server, visible)
    assert (errors == []) == ok


def test_unsupported_option_is_reported():
    argv = launch_vllm.build_argv(launch_vllm.load_profile(AWQ)["server"], "fcfs", smoke=True)
    errs = launch_vllm.check_supported(
        argv, ENGINE_FIELDS - {"generation_config"}, {"host", "port"}
    )
    assert errs == ["설치된 vLLM 에 없는 옵션: --generation-config"]
    assert launch_vllm.check_supported(argv, ENGINE_FIELDS, {"host", "port"}) == []


def dry(profile, visible, installed=INSTALLED, gpu=None, extra=()):
    env = {"CUDA_VISIBLE_DEVICES": visible, "VLLM_API_KEY": "k"}
    n = len(visible.split(","))
    return launch_vllm.main(
        ["--profile", str(profile), "--dry-run", *extra],
        env=env,
        installed=installed,
        gpus=gpu or gpus(n),
    )


def test_dry_run_preflight(capsys):
    assert dry(AWQ, "0") == 0
    assert dry(BF16, "0,1") == 0
    assert dry(BF16, "0") == 2  # GPU 1장에 TP=2
    assert dry(AWQ, "0", installed={**INSTALLED, "vllm": "0.32.0"}) == 2
    assert (
        dry(AWQ, "0", installed={**INSTALLED, "vllm": "0.32.0"}, extra=["--allow-version-mismatch"])
        == 0
    )
    assert dry(AWQ, "0", gpu=gpus(1, busy=["GPU-0, 123, python, 9000"])) == 2
    assert (
        dry(AWQ, "0", gpu=gpus(1, busy=["GPU-0, 123, python, 9000"]), extra=["--allow-busy-gpu"])
        == 0
    )
    out = capsys.readouterr().out
    assert "preflight_errors" in out


class FakeProc:
    def __init__(self, cmd, env, **kw):
        self.cmd, self.env = cmd, env
        self.stdout = iter(
            [
                "INFO Model loading took 5.42 GiB memory and 12.3 seconds\n",
                "INFO Available KV cache memory: 6.10 GiB\n",
                "INFO GPU KV cache size: 49,952 tokens, Maximum concurrency for 6,144 tokens per "
                "request: 8.13x\n",
                "INFO:     Application startup complete.\n",
            ]
        )
        FakeProc.last = self

    def wait(self):
        return 0


def test_launch_writes_meta_and_log_then_client_checks_it(tmp_path, monkeypatch):
    monkeypatch.setattr(launch_vllm.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(launch_vllm, "port_free", lambda port: True)
    env = {"CUDA_VISIBLE_DEVICES": "0", "VLLM_API_KEY": "secret-k"}
    rc = launch_vllm.main(
        ["--profile", str(AWQ), "--smoke", "--log-dir", str(tmp_path)],
        env=env,
        installed=INSTALLED,
        gpus=gpus(1),
    )
    assert rc == 0
    assert FakeProc.last.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "VLLM_SERVER_DEV_MODE" not in FakeProc.last.env
    (meta_path,) = tmp_path.glob("*_launch.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "secret-k" not in meta_path.read_text(encoding="utf-8")
    assert meta["smoke"] is True and meta["profile"]["name"] == "vllm_awq_int4_1gpu"
    assert Path(meta["log"]).read_text(encoding="utf-8").count("\n") == 4

    prof = launch_vllm.load_profile(AWQ)
    info = check_launch_meta(meta_path, prof["llm"], prof["server"], prof["engine_profile"])
    assert info["tensor_parallel_size"] == 1 and info["enforce_eager"] is True
    bf16 = launch_vllm.load_profile(BF16)
    with pytest.raises(PreflightError):  # 다른 프로필의 client 설정으로 이 서버를 쓰려 함
        check_launch_meta(meta_path, bf16["llm"], bf16["server"], bf16["engine_profile"])
    with pytest.raises(PreflightError, match="engine_profile"):
        check_launch_meta(meta_path, prof["llm"], prof["server"], "vllm_bf16_tp2")

    # verify_launch: 로그·추정치 비교 (nvidia-smi / API 없이)
    monkeypatch.setattr(verify_launch.shutil, "which", lambda name: None)
    out = verify_launch.main(["--launch-meta", str(meta_path), "--no-api"])
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["checks"]["startup_complete"] and rep["checks"]["no_error_lines"]
    assert rep["log"]["kv_cache_tokens"] == 49952 and rep["log"]["max_concurrency"] == 8.13
    cmp = rep["compare"]
    assert cmp["weights_gib_per_gpu"] == {
        "estimate": 5.5,
        "actual": 5.42,
        "actual_over_estimate": 0.985,
    }
    assert cmp["kv_cache_tokens"]["estimate"] == 45000
    with pytest.raises(FileExistsError):  # 덮어쓰지 않음
        verify_launch.main(["--launch-meta", str(meta_path), "--no-api"])


def test_launch_refuses_tp_mismatch_without_starting(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(launch_vllm.subprocess, "Popen", lambda *a, **k: called.append(1))
    monkeypatch.setattr(launch_vllm, "port_free", lambda port: True)
    rc = launch_vllm.main(
        ["--profile", str(BF16), "--log-dir", str(tmp_path)],
        env={"CUDA_VISIBLE_DEVICES": "1", "VLLM_API_KEY": "k"},
        installed=INSTALLED,
        gpus=gpus(1),
    )
    assert rc == 2 and called == [] and not list(tmp_path.glob("*"))


def test_parse_tp2_log_and_errors():
    text = (
        "(Worker_TP0 pid=1) INFO Model loading took 7.52 GiB memory and 30.1 seconds\n"
        "(Worker_TP1 pid=2) INFO Model loading took 7.52 GiB memory and 30.4 seconds\n"
        "(Worker_TP0 pid=1) INFO Available KV cache memory: 4.05 GiB\n"
        "(Worker_TP1 pid=2) INFO Available KV cache memory: 4.05 GiB\n"
        "INFO GPU KV cache size: 66,336 tokens, Maximum concurrency for 6,144 tokens per "
        "request: 10.80x\n"
        "(Worker_TP1 pid=2) NCCL WARN Call to connect returned Connection refused\n"
    )
    r = verify_launch.parse_vllm_log(text)
    assert r["model_loading_gib"] == [7.52, 7.52] and r["available_kv_cache_gib"] == [4.05, 4.05]
    assert r["kv_cache_tokens"] == 66336 and not r["startup_complete"]
    assert len(r["error_lines"]) == 1 and "NCCL" in r["error_lines"][0]
    oom = verify_launch.parse_vllm_log("torch.OutOfMemoryError: CUDA out of memory.\n")
    assert oom["error_lines"]
