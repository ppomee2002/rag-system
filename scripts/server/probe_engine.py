"""SMOKE-P: 실행 중인 추론 서버 점검 (vLLM / Ollama). smoke 용이며 논문 결과로 쓰지 않는다.

레포 환경 (클라이언트) 에서 실행한다. 엔진 프로필을 --config 로 준다.

    P=experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml
    uv run python scripts/server/probe_engine.py --config $P \
        --set llm.launch_meta=$HOME/crag_logs/<...>_launch.json
    # 선택 (GPU 부하 발생):
    #   --sweep 1,2,4,8 --sweep-profile crag     CRAG 호출 길이·패턴 부하 (K 예비 측정)
    #   --priority-order 6                       priority 처리 순서 검증
    #       (서버: launch_vllm.py --sched-policy priority --max-num-seqs 1, config 도
    #        --set llm.server_scheduling_policy=priority)

결과: results/crag_system/smoke/SMOKE-P/<timestamp>/ report.json, calls.jsonl, run_meta.json

점검 항목
  launch_meta    launch 기록 (모델, TP, GPU 수, max_model_len, 정책) 과 config 대조
  smoke          짧은 요청 1회: 토큰 수, wall, TTFT
  determinism    같은 요청 3회 (temperature 0) 답 일치 여부
  grade_format   기존 GRADE prompt 응답이 CRAG 파서로 숫자로 읽히는지
  auth           (vLLM) key 없이 /v1 요청 시 거부되는지
  priority_api   (vLLM) priority 0/1 요청 수락 여부 (참고용. 0.33.0 은 FCFS 서버도 수락할 수 있음)
  metrics        (vLLM) /metrics 지표 이름
  priority_order (vLLM) 처리 순서로 priority 검증:
                 blocker 요청으로 서버 slot 을 채운 뒤 test 요청 M 개를
                 우선순위가 도착 순서와 반대가 되도록 보낸다.
                 실제 처리 시작 순서 (첫 토큰 시각) 가 priority 순서와 도착 순서 중 어느
                 쪽에 가까운지 Kendall tau 로 판정하고, 요청별 대기 (첫 토큰까지) 를 기록한다.
  sweep          동시 요청 수 K 별 처리량.
                 profile=crag 는 query 단위 (grade 5회 순차 + generate 1회),
                 문서 길이는 실제 chunk 와 비슷하게 (기본 2,000자 x 5), 요청마다 다른 텍스트
                 (prefix cache 가 처리량을 부풀리지 않도록). vLLM 대기열·preemption 을 함께 기록.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.config import load_layered_config
from rag.llm.engine import EngineError, LLMRequest
from rag.llm.engine_llm import EngineLLM
from rag.llm.factory import build_engine
from rag.llm.vllm_engine import VLLMEngine, parse_prometheus
from rag.pipelines import prompts
from rag.scheduling.preflight import check_launch_meta
from rag.telemetry.calls import CallRecorder
from rag.telemetry.engine_metrics import MetricsSampler, delta, snapshot
from rag.telemetry.run_meta import (
    JsonlWriter,
    collect_run_metadata,
    create_run_dir,
    write_new,
    write_run_files,
)
from rag.types import Document

EXPERIMENT_ID = "SMOKE-P"
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
VOCAB = (
    "river museum council harbor dynasty treaty valley painter novel railway cathedral island "
    "senate fortress orchestra village province chemist monastery bridge festival archive "
    "governor mountain merchant colony theater library garrison academy pilgrim lighthouse "
    "republic parish vineyard emperor manuscript observatory canal citadel tribunal guild"
).split()


def call_once(engine, recorder, prompt, **kw):
    return EngineLLM(engine, recorder=recorder).generate(prompt, **kw)


def check(name, fn, report):
    print(f"[{name}] ...", flush=True)
    try:
        report[name] = fn()
    except Exception as e:  # noqa: BLE001 - 다른 점검은 계속
        report[name] = {"error": f"{type(e).__name__}: {e}"}
    print(f"[{name}] {json.dumps(report[name], ensure_ascii=False, default=str)[:600]}", flush=True)


def kendall_tau(order_a: list, order_b: list) -> float:
    """두 순서의 Kendall tau (1 = 같은 순서, -1 = 반대)."""
    pos = {x: i for i, x in enumerate(order_b)}
    seq = [pos[x] for x in order_a]
    n = len(seq)
    if n < 2:
        return 1.0
    conc = sum(1 for i in range(n) for j in range(i + 1, n) if seq[i] < seq[j])
    pairs = n * (n - 1) / 2
    return (2 * conc - pairs) / pairs


def pseudo_text(rng: random.Random, n_chars: int) -> str:
    words: list[str] = []
    while sum(len(w) + 1 for w in words) < n_chars:
        words.append(rng.choice(VOCAB))
    return " ".join(words).capitalize() + "."


def crag_unit(seed: int, doc_chars: int, n_docs: int) -> dict:
    """CRAG query 한 건과 같은 모양의 prompt 묶음 (grade n_docs 개 + generate 1개)."""
    rng = random.Random(seed)
    question = (
        f"Which {rng.choice(VOCAB)} was associated with the {rng.choice(VOCAB)} of "
        f"{rng.choice(VOCAB).capitalize()} {seed}?"
    )
    docs = [Document(id=f"s{seed}-{i}", text=pseudo_text(rng, doc_chars)) for i in range(n_docs)]
    return {
        "grade": [prompts.GRADE.format(question=question, document=d.text) for d in docs],
        "generate": prompts.GENERATE.format(
            context=prompts.format_context(docs), question=question
        ),
    }


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", required=True, help="엔진 프로필 YAML (또는 llm: 블록이 있는 config)"
    )
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--priority-order", type=int, default=0, help="test 요청 수 (0=생략, 권장 6)")
    ap.add_argument(
        "--priority-blockers",
        type=int,
        default=None,
        help="slot 을 채울 blocker 수 (기본: launch 기록의 max_num_seqs 또는 1)",
    )
    ap.add_argument("--priority-blocker-tokens", type=int, default=256)
    ap.add_argument("--priority-gap", type=float, default=0.05, help="test 요청 간 전송 간격 (초)")
    ap.add_argument("--sweep", default="", help="예: 1,2,4,8 (비우면 생략)")
    ap.add_argument("--sweep-profile", choices=["crag", "synthetic"], default="crag")
    ap.add_argument("--sweep-requests", type=int, default=2, help="K 당 worker 별 query (요청) 수")
    ap.add_argument("--sweep-max-tokens", type=int, default=64, help="synthetic profile 전용")
    ap.add_argument("--crag-doc-chars", type=int, default=2000)
    ap.add_argument("--crag-docs", type=int, default=5)
    a = ap.parse_args(argv)

    cfg = load_layered_config(a.config, a.overlay, a.set)
    cfg.setdefault("run", {})["purpose"] = "smoke"
    llm_cfg = cfg["llm"]
    engine = build_engine(llm_cfg)
    is_vllm = isinstance(engine, VLLMEngine)
    run_dir = create_run_dir(f"crag_system/smoke/{EXPERIMENT_ID}", a.output_dir)
    write_run_files(
        run_dir,
        cfg,
        collect_run_metadata(cfg, engines={"main": engine}, extra={"probe_args": vars(a)}),
    )
    calls = JsonlWriter(run_dir / "calls.jsonl")
    report: dict = {
        "purpose": "smoke",
        "engine": engine.name,
        "model": engine.model,
        "engine_profile": cfg.get("engine_profile"),
    }

    def recorded(tag, t0=None):
        rec = CallRecorder(t0)

        def flush():
            for c in rec.to_dicts():
                calls.write({"check": tag, **c})

        return rec, flush

    def fetch_metrics():
        return parse_prometheus(engine.metrics_text())

    launch_info: dict = {}
    if llm_cfg.get("launch_meta"):

        def launch_meta():
            info = check_launch_meta(
                llm_cfg["launch_meta"], llm_cfg, cfg.get("server"), cfg.get("engine_profile")
            )
            launch_info.update(info)
            return {
                k: info[k]
                for k in (
                    "model",
                    "tensor_parallel_size",
                    "cuda_visible_devices",
                    "scheduling_policy",
                    "max_num_seqs",
                    "max_model_len",
                    "enforce_eager",
                    "profile_name",
                )
            }

        check("launch_meta", launch_meta, report)

    def smoke():
        rec, flush = recorded("smoke")
        text = call_once(engine, rec, SMOKE_PROMPT, max_tokens=16)
        flush()
        c = rec.to_dicts()[0]
        keys = ("prompt_tokens", "out_tokens", "wall", "ttft", "server_total", "finish_reason")
        return {"text": text, **{k: c[k] for k in keys}}

    def determinism():
        rec, flush = recorded("determinism")
        texts = [call_once(engine, rec, SMOKE_PROMPT, max_tokens=32) for _ in range(3)]
        flush()
        return {"identical": len(set(texts)) == 1, "texts": texts}

    def grade_format():
        rec, flush = recorded("grade_format")
        raw = call_once(engine, rec, GRADE_PROMPT)
        flush()
        m = re.search(r"\d*\.?\d+", raw)  # rag.grading.base.parse_score 와 같은 규칙
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

        def priority_api():
            pe = build_engine({**llm_cfg, "send_priority": True, "stream": False})
            out = {}
            for p in (0, 1):
                try:
                    pe.complete(LLMRequest("hi", max_tokens=1, priority=p))
                    out[f"priority_{p}"] = "accepted"
                except EngineError as e:
                    out[f"priority_{p}"] = f"rejected (HTTP {e.status}): {(e.body or '')[:200]}"
            out["note"] = (
                "수락 여부만으로 서버 정책을 판별할 수 없음 (0.33.0). priority_order 로 검증"
            )
            return out

        def metrics():
            m = fetch_metrics()
            names = sorted({k.split("{")[0] for k in m})
            return {
                "n_series": len(m),
                "names": names,
                "expected_present": {
                    k: any(v in n for n in names) for k, v in EXPECTED_METRICS.items()
                },
            }

        check("auth", auth, report)
        check("priority_api", priority_api, report)
        check("metrics", metrics, report)

        if a.priority_order > 0:

            def priority_order():
                return run_priority_order(a, llm_cfg, launch_info, fetch_metrics, recorded)

            check("priority_order", priority_order, report)

    if a.sweep:

        def sweep():
            return run_sweep(a, engine, is_vllm, fetch_metrics, recorded)

        check("sweep", sweep, report)

    write_new(
        run_dir / "report.json", json.dumps(report, indent=2, ensure_ascii=False, default=str)
    )
    print(f"saved -> {run_dir}")
    return run_dir


def wait_running(fetch, n: int, timeout: float = 15.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if (snapshot(fetch()).get("running") or 0) >= n:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.1)
    return False


def run_priority_order(a, llm_cfg, launch_info, fetch, recorded) -> dict:
    """blocker 로 slot 을 채운 뒤, 도착 순서와 반대 우선순위의 test 요청 M 개 처리 순서를 본다."""
    pe = build_engine({**llm_cfg, "send_priority": True, "stream": True})
    mns = launch_info.get("max_num_seqs")
    n_block = a.priority_blockers or (int(mns) if mns else 1)
    m = a.priority_order
    t0 = time.perf_counter()
    before = snapshot(fetch())
    brec, bflush = recorded("priority_order:blocker", t0)
    blockers = [
        threading.Thread(
            target=lambda: EngineLLM(pe, recorder=brec, engine_priority_fn=lambda p: 0).generate(
                LONG_PROMPT, max_tokens=a.priority_blocker_tokens
            )
        )
        for _ in range(n_block)
    ]
    for th in blockers:
        th.start()
    slots_full = wait_running(fetch, n_block)
    trec, tflush = recorded("priority_order:test", t0)

    def submit(i: int, prio: int):
        EngineLLM(pe, recorder=trec, engine_priority_fn=lambda p: prio).generate(
            f"Request {i}. {SMOKE_PROMPT}", max_tokens=8
        )

    tests = []
    for i in range(m):
        th = threading.Thread(target=submit, args=(i, m - i))  # 늦게 보낸 요청일수록 높은 우선순위
        th.start()
        tests.append(th)
        time.sleep(a.priority_gap)
    for th in tests + blockers:
        th.join()
    after = snapshot(fetch())
    bflush()
    tflush()
    test_calls = sorted(trec.to_dicts(), key=lambda c: c["t_enqueue"])  # 전송 순서 = i
    rows = []
    for i, c in enumerate(test_calls):
        rows.append(
            {
                "i": i,
                "priority": m - i,
                "t_submit": c["t_start"],
                "t_first_token": c["t_first_token"],
                "t_end": c["t_end"],
                "wait_to_first_token": c["ttft"],
                "error": c["error"],
            }
        )
    started = sorted(
        rows, key=lambda r: r["t_first_token"] if r["t_first_token"] is not None else r["t_end"]
    )
    observed = [r["i"] for r in started]
    by_priority = [r["i"] for r in sorted(rows, key=lambda r: (r["priority"], r["i"]))]
    by_arrival = [r["i"] for r in rows]
    tau_p = kendall_tau(observed, by_priority)
    tau_f = kendall_tau(observed, by_arrival)
    verdict = "priority_order" if tau_p >= 0.8 else "fcfs_order" if tau_f >= 0.8 else "inconclusive"
    declared = llm_cfg.get("server_scheduling_policy")
    half = m // 2
    hi = sorted(rows, key=lambda r: r["priority"])[:half]
    lo = sorted(rows, key=lambda r: r["priority"])[half:]

    def mean_wait(rs):
        w = [r["wait_to_first_token"] for r in rs if r["wait_to_first_token"] is not None]
        return sum(w) / len(w) if w else None

    return {
        "declared_server_policy": declared,
        "n_blockers": n_block,
        "slots_full_before_test": slots_full,
        "observed_start_order": observed,
        "priority_order": by_priority,
        "arrival_order": by_arrival,
        "kendall_tau_vs_priority": tau_p,
        "kendall_tau_vs_arrival": tau_f,
        "verdict": verdict,
        "consistent_with_declared": {
            "priority": verdict == "priority_order",
            "fcfs": verdict == "fcfs_order",
        }.get(declared),
        "mean_wait_high_priority_s": mean_wait(hi),
        "mean_wait_low_priority_s": mean_wait(lo),
        "requests": rows,
        "vllm_counter_delta": delta(before, after),
        "note": "slots_full_before_test=false 면 첫 test 요청이 바로 실행되어 "
        "판정이 흐려질 수 있음. "
        "서버 --max-num-seqs 를 작게 (1) 띄울 것",
    }


def run_sweep(a, engine, is_vllm, fetch, recorded) -> list[dict]:
    rows = []
    synthetic = prompts.GENERATE.format(
        context=prompts.format_context([Document(id="d", text="Paris is in France. " * 60)]),
        question="Where is Paris?",
    )
    seed = 0
    for k in [int(x) for x in a.sweep.split(",")]:
        rec, flush = recorded(f"sweep:{a.sweep_profile}:K={k}")
        units = []
        for _ in range(k * a.sweep_requests):
            units.append(crag_unit(seed, a.crag_doc_chars, a.crag_docs))
            seed += 1

        def run_unit(u, rec=rec):
            if a.sweep_profile == "synthetic":
                call_once(engine, rec, synthetic, max_tokens=a.sweep_max_tokens)
                return
            for p in u["grade"]:  # 실제 CRAG 처럼 순차 grading 후 generation
                call_once(engine, rec, p)
            call_once(engine, rec, u["generate"])

        sampler = MetricsSampler(fetch, 0.5).start() if is_vllm else None
        t0 = time.perf_counter()
        with ThreadPoolExecutor(k) as pool:
            list(pool.map(run_unit, units))
        span = time.perf_counter() - t0
        msum = sampler.stop() if sampler else None
        flush()
        cs = rec.to_dicts()

        def stats(kind, cs=cs):
            sel = [c for c in cs if c["kind"] == kind]
            if not sel:
                return None
            walls = sorted(c["wall"] for c in sel)
            ttfts = [c["ttft"] for c in sel if c["ttft"] is not None]
            return {
                "n": len(sel),
                "mean_wall_s": sum(walls) / len(walls),
                "p95_wall_s": walls[min(len(walls) - 1, int(0.95 * len(walls)))],
                "mean_ttft_s": sum(ttfts) / len(ttfts) if ttfts else None,
                "mean_prompt_tokens": sum(c["prompt_tokens"] or 0 for c in sel) / len(sel),
                "mean_out_tokens": sum(c["out_tokens"] or 0 for c in sel) / len(sel),
            }

        row = {
            "K": k,
            "profile": a.sweep_profile,
            "n_units": len(units),
            "span_s": span,
            "units_per_s": len(units) / span,
            "calls_per_s": len(cs) / span,
            "prompt_tok_per_s": sum(c["prompt_tokens"] or 0 for c in cs) / span,
            "out_tok_per_s": sum(c["out_tokens"] or 0 for c in cs) / span,
            "by_kind": {kind: stats(kind) for kind in ("grade", "generate", "other")},
            "n_errors": sum(c["error"] is not None for c in cs),
            "vllm": msum,
        }
        rows.append(row)
        print(
            f"  K={k}: units/s={row['units_per_s']:.3f} calls/s={row['calls_per_s']:.2f}",
            flush=True,
        )
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
