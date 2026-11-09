"""SYS-P: 실행 중인 추론 서버 점검 (vLLM / Ollama).

레포 환경 (클라이언트) 에서 실행한다.

    CFG=experiments/crag_system/configs/engine_vllm.yaml
    uv run python scripts/server/probe_engine.py --config $CFG
    # 선택: priority 순서 / 처리량 점검 (GPU 부하 발생, 공유 서버에서는 짧게)
    uv run python scripts/server/probe_engine.py --config $CFG --priority-load 8 --sweep 1,2,4,8

점검 항목 (결과: results/crag_system/SYS-P/<timestamp>/report.json, calls.jsonl, run_meta.json):
  describe      엔진 / 모델 / 버전 / (Ollama) quantization·GPU 적재
  smoke         짧은 요청 1회: 토큰 수, wall, TTFT
  determinism   같은 요청 3회 (temperature 0) 답 일치 여부
  grade_format  기존 GRADE prompt 응답이 CRAG 파서로 숫자로 읽히는지 (엔진 간 출력 형식 차이)
  auth          (vLLM) key 없이 요청 시 거부되는지
  priority      (vLLM) priority 0/1 요청 수락 여부 -> 서버의 scheduling policy 판단
  metrics       (vLLM) /metrics 지표 이름
                (queue / TTFT / prefill / decode / preemption / prefix cache)
  priority_order (--priority-load N) 긴 저우선 요청 N 개 사이에 짧은 고우선 요청을 넣어 대기 비교
  sweep         (--sweep K,...) 동시 요청 수 K 별 처리량 (continuous batching 효과의 기초 자료)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.config import load_config
from rag.llm.engine import EngineError, LLMRequest
from rag.llm.engine_llm import EngineLLM
from rag.llm.factory import build_engine
from rag.llm.vllm_engine import VLLMEngine, parse_prometheus
from rag.pipelines import prompts
from rag.telemetry.calls import CallRecorder
from rag.telemetry.run_meta import (
    JsonlWriter,
    collect_run_metadata,
    create_run_dir,
    write_new,
    write_run_files,
)
from rag.types import Document

EXPERIMENT_ID = "SYS-P"
SMOKE_PROMPT = "What is the capital of France? Answer in one word."
GRADE_PROMPT = prompts.GRADE.format(
    question="What is the capital of France?",
    document="Paris is the capital and most populous city of France.",
)
LONG_PROMPT = "Write a detailed, multi-paragraph history of the city of Paris."
EXPECTED_METRICS = {
    "queue_time": "request_queue_time",
    "ttft": "time_to_first_token",
    "e2e": "e2e_request_latency",
    "prefill": "request_prefill_time",
    "decode": "request_decode_time",
    "preemptions": "num_preemptions",
    "prefix_cache": "prefix_cache",
    "running": "num_requests_running",
    "waiting": "num_requests_waiting",
}


def call_once(engine, recorder, prompt, **kw):
    llm = EngineLLM(engine, recorder=recorder)
    return llm.generate(prompt, **kw)


def check(name, fn, report):
    print(f"[{name}] ...", flush=True)
    try:
        report[name] = fn()
    except Exception as e:  # noqa: BLE001 - 다른 점검은 계속
        report[name] = {"error": f"{type(e).__name__}: {e}"}
    print(f"[{name}] {report[name]}", flush=True)


def metric_sum(metrics: dict, needle: str) -> float:
    return sum(v for k, v in metrics.items() if needle in k and not k.endswith("_created"))


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument(
        "--priority-load", type=int, default=0, help="priority_order 배경 요청 수 (0=생략)"
    )
    ap.add_argument("--sweep", default="", help="예: 1,2,4,8 (비우면 생략)")
    ap.add_argument("--sweep-requests", type=int, default=4, help="K 당 worker 별 요청 수")
    ap.add_argument("--sweep-max-tokens", type=int, default=64)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    llm_cfg = cfg["llm"]
    engine = build_engine(llm_cfg)
    is_vllm = isinstance(engine, VLLMEngine)
    run_dir = create_run_dir(f"crag_system/{EXPERIMENT_ID}", a.output_dir)
    write_run_files(
        run_dir,
        cfg,
        collect_run_metadata(cfg, engines={"main": engine}, extra={"probe_args": vars(a)}),
    )
    calls = JsonlWriter(run_dir / "calls.jsonl")
    report: dict = {"engine": engine.name, "model": engine.model}

    def recorded(tag):
        rec = CallRecorder()

        def flush():
            for c in rec.to_dicts():
                calls.write({"check": tag, **c})

        return rec, flush

    def smoke():
        rec, flush = recorded("smoke")
        text = call_once(engine, rec, SMOKE_PROMPT, max_tokens=16)
        flush()
        c = rec.to_dicts()[0]
        return {
            "text": text,
            **{
                k: c[k]
                for k in (
                    "prompt_tokens",
                    "out_tokens",
                    "wall",
                    "ttft",
                    "server_total",
                    "finish_reason",
                )
            },
        }

    def determinism():
        rec, flush = recorded("determinism")
        texts = [call_once(engine, rec, SMOKE_PROMPT, max_tokens=32) for _ in range(3)]
        flush()
        return {"identical": len(set(texts)) == 1, "texts": texts}

    def grade_format():
        rec, flush = recorded("grade_format")
        raw = call_once(engine, rec, GRADE_PROMPT)
        flush()
        m = re.search(r"\d*\.?\d+", raw)  # rag.pipelines.crag.CRAG.grade 와 같은 파서
        return {
            "raw": raw[:300],
            "parsed": float(m.group()) if m else None,
            "out_tokens": rec.to_dicts()[0]["out_tokens"],
        }

    check("smoke", smoke, report)
    check("determinism", determinism, report)
    check("grade_format", grade_format, report)

    if is_vllm:

        def auth():
            anon = build_engine({**llm_cfg, "api_key_env": None, "stream": False})
            try:
                anon.complete(LLMRequest("hi", max_tokens=1))
                return {"rejected_without_key": False}
            except EngineError as e:
                return {"rejected_without_key": e.status == 401, "status": e.status}

        def priority():
            pe = build_engine({**llm_cfg, "send_priority": True, "stream": False})
            out = {}
            for p in (0, 1):
                try:
                    pe.complete(LLMRequest("hi", max_tokens=1, priority=p))
                    out[f"priority_{p}"] = "accepted"
                except EngineError as e:
                    out[f"priority_{p}"] = f"rejected (HTTP {e.status}): {(e.body or '')[:200]}"
            out["priority_scheduling_likely_enabled"] = out["priority_1"] == "accepted"
            return out

        def metrics():
            m = parse_prometheus(engine.metrics_text())
            names = sorted({k.split("{")[0] for k in m})
            return {
                "n_series": len(m),
                "names": names,
                "expected_present": {
                    k: any(v in n for n in names) for k, v in EXPECTED_METRICS.items()
                },
            }

        check("auth", auth, report)
        check("priority", priority, report)
        check("metrics", metrics, report)

        if a.priority_load > 0:

            def priority_order():
                if not (report.get("priority") or {}).get("priority_scheduling_likely_enabled"):
                    return {"skipped": "priority 요청 거부됨 (--scheduling-policy priority 아님)"}
                pe = build_engine({**llm_cfg, "send_priority": True})
                result = {}
                for label, probe_priority in (("probe_high", 0), ("probe_same", 10)):
                    before = parse_prometheus(engine.metrics_text())
                    rec, flush = recorded(f"priority_order:{label}")
                    bg_rec, bg_flush = recorded(f"priority_order:{label}:background")

                    def bg(bg_rec=bg_rec):
                        EngineLLM(pe, recorder=bg_rec, engine_priority_fn=lambda p: 10).generate(
                            LONG_PROMPT, max_tokens=256
                        )

                    threads = [threading.Thread(target=bg) for _ in range(a.priority_load)]
                    for t in threads:
                        t.start()
                    time.sleep(0.5)
                    EngineLLM(
                        pe, recorder=rec, engine_priority_fn=lambda p, q=probe_priority: q
                    ).generate(SMOKE_PROMPT, max_tokens=16)
                    for t in threads:
                        t.join()
                    after = parse_prometheus(engine.metrics_text())
                    flush()
                    bg_flush()
                    c = rec.to_dicts()[0]
                    result[label] = {
                        "probe_wall": c["wall"],
                        "probe_ttft": c["ttft"],
                        "preemptions_delta": metric_sum(after, "num_preemptions")
                        - metric_sum(before, "num_preemptions"),
                    }
                result["note"] = (
                    "probe_high 가 probe_same 보다 TTFT 가 뚜렷이 짧으면 priority 가 대기 순서에 "
                    "반영됨. max_num_seqs 보다 배경 요청이 적으면 차이가 없을 수 있음"
                )
                return result

            check("priority_order", priority_order, report)

    if a.sweep:

        def sweep():
            gen_prompt = prompts.GENERATE.format(
                context=prompts.format_context(
                    [Document(id="d", text="Paris is in France. " * 60)]
                ),
                question="Where is Paris?",
            )
            rows = []
            for k in [int(x) for x in a.sweep.split(",")]:
                rec, flush = recorded(f"sweep:K={k}")
                n = k * a.sweep_requests
                t0 = time.perf_counter()
                with ThreadPoolExecutor(k) as pool:
                    list(
                        pool.map(
                            lambda _, rec=rec: call_once(
                                engine, rec, gen_prompt, max_tokens=a.sweep_max_tokens
                            ),
                            range(n),
                        )
                    )
                span = time.perf_counter() - t0
                flush()
                cs = rec.to_dicts()
                out_tok = sum(c["out_tokens"] or 0 for c in cs)
                ttfts = [c["ttft"] for c in cs if c["ttft"] is not None]
                rows.append(
                    {
                        "K": k,
                        "n_requests": n,
                        "span_s": span,
                        "req_per_s": n / span,
                        "out_tok_per_s": out_tok / span,
                        "mean_wall_s": sum(c["wall"] for c in cs) / n,
                        "mean_ttft_s": sum(ttfts) / len(ttfts) if ttfts else None,
                    }
                )
                print(f"  K={k}: {rows[-1]}", flush=True)
            return rows

        check("sweep", sweep, report)

    write_new(
        run_dir / "report.json", json.dumps(report, indent=2, ensure_ascii=False, default=str)
    )
    print(f"saved -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
