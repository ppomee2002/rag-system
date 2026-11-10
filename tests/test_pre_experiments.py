"""예비 측정 도구와 실험 구분 기능 검증 (가짜 엔진, 성능·정확도 결과 아님).

- smoke / pre / paper 구분과 본실험 가드
- PRE-K calibration (실제 runner pipeline, 처리량 포화 K 추천)
- PRE-Q 품질 비교 (같은 질문·prompt·replay, 엔진만 다름)
- vLLM /metrics 샘플링, 출력 텍스트 기록, prefix cache reset 응답 해석
"""

import json
import shutil
import sys
import threading
from pathlib import Path

import pytest

from rag.llm.engine_llm import EngineLLM
from rag.llm.vllm_engine import VLLMEngine
from rag.retrieval.web_cache import RecordingWebSearch
from rag.telemetry.calls import CallRecorder
from rag.telemetry.engine_metrics import MetricsSampler, snapshot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "crag_system"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "server"))
import calibrate_k  # noqa: E402
import launch_vllm  # noqa: E402
from compare_quality import compare_quality, mcnemar_exact  # noqa: E402
from crag_fixtures import FakeEngine, Retriever, Web, read_jsonl  # noqa: E402
from runner import Components, PurposeError, Runner, build_web_search, check_purpose  # noqa: E402

CLEAN = {"dirty": False, "commit": "abc"}
PAPER_LLM = {"provider": "vllm", "launch_meta": "/x/launch.json"}


# ================================================================ purpose
@pytest.mark.parametrize(
    "cfg,approved,git,ok",
    [
        ({"exp_id": "SMOKE-1", "run": {"purpose": "smoke"}}, False, None, True),
        ({"exp_id": "PRE-K", "run": {"purpose": "pre"}}, False, None, True),
        ({"exp_id": "SYS-1b", "run": {"purpose": "smoke"}}, False, None, False),
        ({"exp_id": "SMOKE-1", "run": {"purpose": "paper"}}, True, CLEAN, False),
        (
            {
                "exp_id": "SYS-1b",
                "run": {"purpose": "paper"},
                "engine_profile": "p",
                "llm": PAPER_LLM,
            },
            False,
            CLEAN,
            False,
        ),  # 승인 없음
        (
            {
                "exp_id": "SYS-1b",
                "run": {"purpose": "paper"},
                "engine_profile": "p",
                "llm": PAPER_LLM,
            },
            True,
            {"dirty": True},
            False,
        ),  # 커밋 안 된 변경
        (
            {
                "exp_id": "SYS-1b",
                "run": {"purpose": "paper", "limit_override": 5},
                "engine_profile": "p",
                "llm": PAPER_LLM,
            },
            True,
            CLEAN,
            False,
        ),  # --limit
        (
            {"exp_id": "SYS-1b", "run": {"purpose": "paper"}, "llm": {"provider": "vllm"}},
            True,
            CLEAN,
            False,
        ),  # 프로필·launch 기록 없음
        (
            {
                "exp_id": "SYS-1b",
                "run": {"purpose": "paper"},
                "engine_profile": "p",
                "llm": PAPER_LLM,
            },
            True,
            CLEAN,
            True,
        ),
        ({"exp_id": "X-1", "run": {"purpose": "lab"}}, False, None, False),
    ],
)
def test_check_purpose(cfg, approved, git, ok):
    if ok:
        assert check_purpose(cfg, approved, git)["purpose"] == cfg["run"]["purpose"]
    else:
        with pytest.raises(PurposeError):
            check_purpose(cfg, approved, git)


def test_runner_separates_purpose_dirs_and_blocks_unapproved_paper(setup):
    tmp, make = setup
    run_dir = Runner(*make(["exp_id=PRE-Q", "run.purpose=pre"]), run_id="r").run()
    assert run_dir.parent.parent.name == "pre"
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["purpose"]["purpose"] == "pre"
    with pytest.raises(PurposeError):
        Runner(*make(["exp_id=SYS-1b", "run.purpose=paper"]), run_id="p")
    assert not (tmp / "results" / "crag_system" / "paper").exists()


# ================================================================ PRE-K
class CappedEngine(FakeEngine):
    """동시 처리 2개가 한계인 가짜 서버 -> K=2 부근에서 처리량 포화."""

    def __init__(self):
        super().__init__(delay=0.02)
        self.sem = threading.Semaphore(2)

    def complete(self, request):
        with self.sem:
            return super().complete(request)


def test_calibrate_k_with_real_pipeline_shape(setup):
    tmp, make = setup
    cfg, _ = make()
    with pytest.raises(ValueError, match="replay"):
        calibrate_k.calibrate(cfg, [1, 2], 4)
    out = calibrate_k.calibrate(
        cfg,
        [1, 2, 4],
        10,
        rate=1000,
        frac=0.85,
        components_factory=lambda: Components(Retriever(), Web(), CappedEngine()),
        allow_live_web=True,
    )
    rep = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    rows = {r["K"]: r for r in rep["rows"]}
    assert set(rows) == {1, 2, 4} and all(r["n_errors"] == 0 for r in rows.values())
    assert rows[2]["throughput_qps"] > 1.4 * rows[1]["throughput_qps"]  # 포화 전에는 증가
    assert rep["recommendation"]["K_star"] in (2, 4)  # 2 에서 포화
    assert out.parent.name == "PRE-K" and out.parent.parent.name == "pre"
    k_run = Path(rows[2]["run_dir"])
    meta = json.loads((k_run / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["scheduler"]["slots"] == 2 and meta["mode"] == "live"
    assert meta["comparability"]["session"]["K"] == 2
    q = [r for r in read_jsonl(k_run / "queries.jsonl") if not r["warmup"]]
    assert len(q) == 10 and {r["calls_summary"]["n_calls"] for r in q} <= {4, 5}  # CRAG 호출 패턴


def test_recommend_rules():
    rows = [
        {
            "K": 1,
            "n_errors": 0,
            "throughput_qps": 1.0,
            "vllm_preemptions": 0,
            "vllm_waiting_max": 0,
        },
        {
            "K": 2,
            "n_errors": 0,
            "throughput_qps": 1.9,
            "vllm_preemptions": 0,
            "vllm_waiting_max": 0,
        },
        {
            "K": 4,
            "n_errors": 0,
            "throughput_qps": 2.0,
            "vllm_preemptions": 3,
            "vllm_waiting_max": 2,
        },
        {
            "K": 8,
            "n_errors": 2,
            "throughput_qps": 2.1,
            "vllm_preemptions": 0,
            "vllm_waiting_max": 4,
        },
    ]
    rec = calibrate_k.recommend(rows, 0.95)
    assert rec["K_star"] == 2 and rec["mu_max_qps"] == 1.9  # preemption / 오류 있는 K 제외


# ================================================================ PRE-Q
class AltEngine(FakeEngine):
    """다른 엔진 흉내: grade 출력 형식이 다르고 (숫자 + 설명), 점수가 약간 다름."""

    model = "alt-model"

    def complete(self, request):
        r = super().complete(request)
        if r.text == "0.9":
            r.text = "Score: 0.8"
        return r


def test_compare_quality_engine_vs_engine(setup, tmp_path):
    tmp, make = setup
    rec_file = tmp_path / "web_cache.jsonl"
    cfg_a, _ = make(["exp_id=PRE-Q", "run.purpose=pre", "run.record_texts=true"])
    run_a = Runner(
        cfg_a,
        Components(Retriever(), RecordingWebSearch(Web(), rec_file), FakeEngine()),
        run_id="a",
    ).run()
    shutil.copy(rec_file, run_a / "web_cache.jsonl")  # 실제 runner 의 record=true 와 같은 위치
    cfg_b, _ = make(
        [
            "exp_id=PRE-Q",
            "run.purpose=pre",
            "run.record_texts=true",
            "llm.model=alt-model",
            "crag.web_search.provider=replay",
            f"crag.web_search.replay_from_runs=['{run_a.as_posix()}']",
            "crag.web_search.latency_mode=zero",
        ]
    )
    run_b = Runner(
        cfg_b, Components(Retriever(), build_web_search(cfg_b, tmp_path), AltEngine()), run_id="b"
    ).run()
    rep = compare_quality(str(run_a), str(run_b))
    assert rep["prerequisites_ok"] and rep["web_replay"]["b_replays_a"]
    assert rep["web_replay"]["b_stats"]["misses"] == 0
    assert any(d["check"] == "llm" for d in rep["engine_differences"])
    assert rep["n_paired"] == 10 and rep["action_agreement"] == 1.0
    gs = rep["grade_scores"]
    assert gs["n_pairs"] == 30 and gs["mean_abs_diff"] > 0 and gs["band_agreement"] == 1.0
    fmt = rep["grade_output_format"]
    assert fmt["a"]["number_only_rate"] == 1.0 and fmt["b"]["number_only_rate"] < 1.0
    assert rep["accuracy"]["a_only"] == 0 and rep["accuracy"]["mcnemar_exact_p"] is None


def test_mcnemar():
    assert mcnemar_exact(0, 0) is None
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)


# ================================================================ 보조 기능
def test_metrics_sampler_and_snapshot():
    seq = iter(
        [
            {
                'vllm:num_requests_running{m="x"}': 1.0,
                'vllm:num_requests_waiting{m="x"}': 0.0,
                'vllm:num_preemptions_total{m="x"}': 2.0,
            },
            {
                'vllm:num_requests_running{m="x"}': 4.0,
                'vllm:num_requests_waiting{m="x"}': 3.0,
                'vllm:num_preemptions_total{m="x"}': 5.0,
            },
        ]
    )
    s = MetricsSampler(lambda: next(seq), interval=10)
    s._take()
    s._take()
    summ = s.summary()
    assert summ["running"]["max"] == 4.0 and summ["waiting"]["max"] == 3.0
    assert summ["counter_delta"]["preemptions"] == 3.0
    assert snapshot({})["running"] is None


def test_output_text_recorded_only_when_enabled():
    eng = FakeEngine(delay=0)
    off, on = CallRecorder(), CallRecorder(record_texts=True)
    EngineLLM(eng, recorder=off).generate("hello")
    EngineLLM(eng, recorder=on).generate("hello")
    assert off.to_dicts()[0]["output_text"] is None
    assert on.to_dicts()[0]["output_text"] == "ans"


def test_reset_prefix_cache_reads_success(mock_server):
    ok = mock_server("vllm")
    assert VLLMEngine("m", base_url=ok.url + "/v1", api_key_env=None).reset_prefix_cache()["ok"]
    busy = mock_server("vllm", reset_success=False)
    r = VLLMEngine("m", base_url=busy.url + "/v1", api_key_env=None).reset_prefix_cache()
    assert r["ok"] is False and r["success"] is False
    ollama_like = mock_server("ollama")  # 경로 없음 -> 404 (dev mode 아님)
    r = VLLMEngine("m", base_url=ollama_like.url + "/v1", api_key_env=None).reset_prefix_cache()
    assert r["ok"] is False and r["status"] == 404 and "dev mode" in r["error"]


def test_launch_dev_mode_sets_env(tmp_path, monkeypatch):
    seen = {}

    class P:
        def __init__(self, cmd, env, **kw):
            seen.update(env)
            self.stdout = iter([])

        def wait(self):
            return 0

    monkeypatch.setattr(launch_vllm.subprocess, "Popen", P)
    monkeypatch.setattr(launch_vllm, "port_free", lambda port: True)
    profile = Path(__file__).resolve().parents[1] / "experiments/crag_system/configs/engines"
    installed = {
        "vllm": "0.33.0",
        "engine_fields": {
            "tensor_parallel_size",
            "dtype",
            "max_model_len",
            "gpu_memory_utilization",
            "max_num_seqs",
            "generation_config",
            "seed",
            "scheduling_policy",
            "enable_prefix_caching",
        },
        "frontend_fields": {"host", "port"},
    }
    gpu = {"available": True, "selected": [{"index": "0", "uuid": "GPU-0"}], "busy_processes": []}
    rc = launch_vllm.main(
        [
            "--profile",
            str(profile / "vllm_awq_int4_1gpu.yaml"),
            "--dev-mode",
            "--log-dir",
            str(tmp_path),
        ],
        env={"CUDA_VISIBLE_DEVICES": "0", "VLLM_API_KEY": "k"},
        installed=installed,
        gpus=gpu,
    )
    assert rc == 0 and seen["VLLM_SERVER_DEV_MODE"] == "1"
