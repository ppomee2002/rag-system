"""PRE-K: 동시 전송 수 K calibration (실제 CRAG pipeline 으로).

실제 질문·검색 결과·prompt 길이·호출 순서 (grading → [rewrite → web search] → generation) 를 그대로
쓰기 위해 runner 를 live 모드로 돌린다. 모든 요청이 거의 동시에 도착하는 burst workload 를 K 마다
같은 파일로 재사용하고, client gate 를 FCFS + slots=K 로 둔다.

    P=experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml
    uv run python experiments/crag_system/calibrate_k.py --overlay $P \
        --set llm.launch_meta=$HOME/crag_logs/<...>_launch.json \
        --set crag.web_search.provider=replay \
        --set "crag.web_search.replay_from_runs=['results/crag_system/smoke/SMOKE-1/<run>']" \
        --ks 1,2,4,6,8,12 --n-requests 40

- web search 는 기록 재생 (기본 latency_mode=zero: GPU 처리량만 보기 위해 외부 지연 제거).
  실시간 Tavily 로 burst 를 보내지 않도록 provider=tavily 는 거부한다 (--allow-live-web 로 허용).
- 각 K 결과는 results/crag_system/pre/PRE-K/<ts>_K<k>/ (runner 결과 + engine_metrics.jsonl),
  요약은 results/crag_system/pre/PRE-K/<ts>_calibration/calibration.json.
- 추천 K* = 처리량이 최대치의 frac (기본 0.95) 이상인 가장 작은 K (오류 0, preemption 0 인 K 중).
  μ_max = K* 의 처리량 (query/s) -> make_workload.py --capacity-rps 로 사용.
  추천값은 참고이며 사용자가 결과를 보고 확정한다.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import (  # noqa: E402
    Runner,
    apply_split,
    load_examples,
    load_layered_config,
    split_warmup,
)
from workload import make_workload  # noqa: E402

from rag.telemetry.run_meta import create_run_dir, write_new  # noqa: E402

EXP_ID = "PRE-K"


def k_config(cfg: dict, k: int, order_index: int, session: str, interval: float) -> dict:
    c = copy.deepcopy(cfg)
    c["exp_id"] = EXP_ID
    run = c.setdefault("run", {})
    run.update(
        purpose="pre",
        mode="live",
        sample_engine_metrics_sec=interval,
        session={"id": session, "order_index": order_index, "K": k},
    )
    c["scheduler"] = {
        **(c.get("scheduler") or {}),
        "policy": "fcfs",
        "slots": k,
        "engine_priority": "none",
        "aging_alpha": 0.0,
        "wait_threshold": None,
        "cost_model": None,
    }
    c.setdefault("llm", {})["send_priority"] = False
    return c


def row_for(k: int, run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    end = json.loads((run_dir / "run_end.json").read_text(encoding="utf-8"))
    sampled = end.get("engine_metrics_sampled") or {}
    counters = sampled.get("counter_delta") or {}
    calls = [
        json.loads(x)
        for x in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    measured = [c for c in calls if not c.get("warmup")]
    span = None
    if measured:
        span = max(c["t_end"] for c in measured) - min(c["t_enqueue"] for c in measured)

    def g(key, stat="max"):
        v = sampled.get(key)
        return v.get(stat) if isinstance(v, dict) else None

    return {
        "K": k,
        "run_dir": str(run_dir),
        "n_ok": summary.get("n_ok"),
        "n_errors": summary.get("n_errors"),
        "throughput_qps": summary.get("throughput_rps"),
        "llm_calls_per_s": len(measured) / span if span else None,
        "prompt_tok_per_s": sum(c["prompt_tokens"] or 0 for c in measured) / span if span else None,
        "latency": summary.get("latency"),
        "mean_gate_wait_s": summary.get("mean_gate_wait"),
        "vllm_running_max": g("running"),
        "vllm_waiting_max": g("waiting"),
        "vllm_kv_usage_max": g("kv_usage"),
        "vllm_preemptions": counters.get("preemptions"),
    }


def recommend(rows: list[dict], frac: float = 0.95) -> dict:
    ok = [
        r
        for r in rows
        if not r["n_errors"] and r["throughput_qps"] and not (r["vllm_preemptions"] or 0)
    ]
    if not ok:
        return {"K_star": None, "reason": "오류·preemption 없는 K 가 없음"}
    best = max(r["throughput_qps"] for r in ok)
    k_star = min((r for r in ok if r["throughput_qps"] >= frac * best), key=lambda r: r["K"])
    return {
        "K_star": k_star["K"],
        "mu_max_qps": k_star["throughput_qps"],
        "max_throughput_qps": best,
        "frac": frac,
        "vllm_waiting_at_K_star": k_star["vllm_waiting_max"],
        "rule": f"오류·preemption 0 인 K 중 처리량 >= {frac} x 최대 인 가장 작은 K",
        "note": "참고 추천값. 결과 표를 보고 확정. "
        "vLLM waiting>0 이면 K 가 서버 동시 처리 한도를 넘음",
    }


def calibrate(
    cfg: dict,
    ks: list[int],
    n_requests: int,
    rate: float = 1000.0,
    seed: int = 0,
    frac: float = 0.95,
    interval: float = 0.5,
    components_factory=None,
    allow_live_web: bool = False,
) -> Path:
    wcfg = (cfg.get("crag") or {}).get("web_search") or {}
    if wcfg.get("provider", "tavily") == "tavily" and not allow_live_web:
        raise ValueError(
            "K calibration 은 web search replay 로 (crag.web_search.provider=replay). "
            "실시간 Tavily 로 burst 를 보내려면 --allow-live-web"
        )
    _, pool = split_warmup(cfg, load_examples(cfg))
    pool = apply_split(cfg, pool)
    wl = make_workload(
        [ex.id for ex in pool], n_requests, seed=seed, rate=rate, name=f"{EXP_ID}-burst"
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for order, k in enumerate(ks):
        ck = k_config(cfg, k, order, f"{EXP_ID}-{ts}", interval)
        comp = components_factory() if components_factory else None
        run_dir = Runner(ck, comp, workload=wl, run_id=f"{ts}_K{k:02d}").run()
        rows.append(row_for(k, run_dir))
        r = rows[-1]
        print(
            f"K={k}: qps={r['throughput_qps']} waiting_max={r['vllm_waiting_max']} "
            f"preempt={r['vllm_preemptions']} errors={r['n_errors']}",
            flush=True,
        )
    out = create_run_dir(
        f"crag_system/pre/{EXP_ID}", cfg.get("output_dir", "results"), f"{ts}_calibration"
    )
    report = {
        "purpose": "pre",
        "engine_profile": cfg.get("engine_profile"),
        "workload": {"sha256": wl["sha256"], "params": wl["params"]},
        "ks": ks,
        "rows": rows,
        "recommendation": recommend(rows, frac),
    }
    write_new(out / "calibration.json", json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["recommendation"], indent=2, ensure_ascii=False))
    print(f"saved -> {out}")
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/crag_system/config.yaml")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--ks", default="1,2,4,6,8,12")
    ap.add_argument("--n-requests", type=int, default=40)
    ap.add_argument("--rate", type=float, default=1000.0, help="burst 도착률 (req/s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.95)
    ap.add_argument("--sample-sec", type=float, default=0.5)
    ap.add_argument("--allow-live-web", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_layered_config(a.config, a.overlay, a.set)
    cfg.setdefault("crag", {}).setdefault("web_search", {}).setdefault("latency_mode", "zero")
    return calibrate(
        cfg,
        [int(x) for x in a.ks.split(",")],
        a.n_requests,
        a.rate,
        a.seed,
        a.frac,
        a.sample_sec,
        allow_live_web=a.allow_live_web,
    )


if __name__ == "__main__":
    main()
