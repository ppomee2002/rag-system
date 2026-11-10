"""B 스케줄링 run 진단 (읽기 전용).

정책이 실제로 개입할 여지가 있었는지, 개입했는지, 비용 예측이 맞았는지 확인한다.

    uv run python experiments/crag_system/diag_sched.py <run_dir> <run_dir> [...] [--name b-rho0.8]

같은 workload 로 돌린 FCFS / SJF / SRPT-est 등의 live run 폴더를 받는다. 추가 추론·API 호출 없음.
읽는 파일: scheduler_log.jsonl, calls.jsonl, queries.jsonl, run_meta.json, run_end.json,
summary.json

지표 정의 (warmup 요청 제외)
  dispatch 기회      gate 가 호출을 내보낼 때 대기열 길이 (= 남은 대기 n_waiting + 1).
                     2 이상이어야 정책이 순서를 고를 수 있다. choice_ge2 = 그런 dispatch 비율.
  재정렬             gate 일련번호 (seq, gate lock 안에서 enqueue 순서대로 부여) 기준으로,
                     더 작은 seq 의 호출이 아직 대기 중인데 나간 dispatch 를 센다.
                     FCFS (호출 단위) 는 정의상 0 이어야 한다 (검증 기준).
                     skipped = 그 dispatch 가 앞질러 간 대기 호출 수.
  gate 대기          gate 안에서 기다린 시간 (scheduler_log.wait, gate 시계). calls.gate_wait 는
                     스레드 깨어남 지연까지 포함한 참고값.
  비용 추정          QueryState.est_history (정책이 dispatch 우선순위로 쓴 값, SJF/SRPT 만).
                     - post_grading_ratio: grading 이후 첫 호출 추정치 / 마지막 grade 호출 추정치
                       (correction 여부별) -> 상태 변화가 추정에 반영된 정도
                     - spearman_vs_actual: 각 호출 시점 추정치와 실제 남은 호출 시간
                       (그 호출 포함 이후 호출들의 wall 합) 의 순위 상관. 낮으면 SRPT 가 짧은 일을
                       골라낼 수 없다. wall 은 부하 중 측정값이라 순수 서비스 시간보다 크다.
  vLLM 내부          run 중 /metrics 샘플: running / waiting (gauge), request_queue_time 평균
                     (histogram sum/count 증가분), preemption 증가분. 요청별 vLLM 대기는 측정 불가.

결과: results/crag_system/analysis/<name>/<timestamp>/ diag.json, per_run.csv, summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag.retrieval.web_cache import read_jsonl as _read_jsonl  # noqa: E402

GRADE_KINDS = ("grade", "grade_batch")


def read_jsonl(path: Path) -> list[dict]:
    return _read_jsonl(path) if Path(path).exists() else []


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def pct(xs, q) -> float | None:
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if len(xs) else None


def mean(xs) -> float | None:
    return float(np.mean(xs)) if len(xs) else None


def spearman(x, y) -> float | None:
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------- dispatch
def dispatch_metrics(log: list[dict], measured_keys: set[str]) -> dict:
    """gate 로그 (dispatch 순서) 로 선택 기회와 재정렬을 계산한다."""
    if not log:
        return {"available": False}
    pending = sorted(e["seq"] for e in log)  # 아직 나가지 않은 seq (오름차순)
    pend_set = set(pending)
    choices, reorders, skipped, waits = Counter(), 0, [], []
    n_meas = 0
    for e in log:  # 로그는 dispatch 순서대로 기록됨
        seq = e["seq"]
        earlier = 0
        # seq 보다 작은데 아직 대기 중인 호출 수
        while pending and pending[0] not in pend_set:
            pending.pop(0)
        if pending and pending[0] < seq:
            earlier = sum(1 for s in pending if s < seq and s in pend_set)
        pend_set.discard(seq)
        if e.get("request_key") not in measured_keys:
            continue
        n_meas += 1
        choices[e["n_waiting"] + 1] += 1
        waits.append(e["wait"])
        if earlier:
            reorders += 1
            skipped.append(earlier)
    ge2 = sum(v for k, v in choices.items() if k >= 2)
    return {
        "available": True,
        "n_dispatch": n_meas,
        "choice_ge2": ge2 / n_meas if n_meas else None,
        "queue_len_mean": mean([k for k, v in choices.items() for _ in range(v)]),
        "queue_len_max": max(choices) if choices else None,
        "queue_len_hist": {str(k): v for k, v in sorted(choices.items())},
        "reorder_dispatches": reorders,
        "reorder_rate": reorders / n_meas if n_meas else None,
        "reorder_rate_given_choice": reorders / ge2 if ge2 else None,
        "skipped_mean": mean(skipped),
        "gate_wait_mean_s": mean(waits),
        "gate_wait_p95_s": pct(waits, 95),
        "gate_wait_share_gt_10ms": mean([w > 0.01 for w in waits]),
    }


# ---------------------------------------------------------------- estimates
def estimate_metrics(rows: list[dict], calls_by_req: dict) -> dict:
    est_vs_actual, post_ratio = [], defaultdict(list)
    n_with_est = 0
    for r in rows:
        hist = (r.get("query_state") or {}).get("est_history") or []
        if not hist:
            continue
        n_with_est += 1
        calls = sorted(calls_by_req.get(r["request_index"], []), key=lambda c: c["t_start"])
        walls = [c["wall"] for c in calls]
        remaining = {}  # (kind, idx) -> 이 호출 포함 이후 wall 합
        for i, c in enumerate(calls):
            remaining[(c["kind"], c["idx"])] = float(sum(walls[i:]))
        for h in hist:
            act = remaining.get((h["kind"], h["idx"]))
            if act is not None:
                est_vs_actual.append((h["value"], act))
        grades = [h for h in hist if h["kind"] in GRADE_KINDS]
        post = [h for h in hist if h["kind"] not in GRADE_KINDS]
        if grades and post and grades[-1]["value"]:
            corrected = r.get("corrective_action") not in (None, "correct")
            key = "corrected" if corrected else "not_corrected"
            post_ratio[key].append(post[0]["value"] / grades[-1]["value"])
    if not n_with_est:
        return {"available": False, "note": "추정치 없음 (FCFS 등 비용 기반이 아닌 정책)"}
    e = [x for x, _ in est_vs_actual]
    a = [y for _, y in est_vs_actual]
    return {
        "available": True,
        "requests_with_estimates": n_with_est,
        "n_points": len(est_vs_actual),
        "spearman_vs_actual": spearman(e, a),
        "mean_est_s": mean(e),
        "mean_actual_remaining_s": mean(a),
        "post_grading_ratio": {k: {"n": len(v), "mean": mean(v)} for k, v in post_ratio.items()},
    }


# ---------------------------------------------------------------- per run
def diag_run(run_dir: str | Path) -> dict:
    d = Path(run_dir)
    meta = read_json(d / "run_meta.json")
    end = read_json(d / "run_end.json")
    summ = read_json(d / "summary.json")
    cfg_sched = meta.get("scheduler") or {}
    rows = [
        r for r in read_jsonl(d / "queries.jsonl") if not r.get("warmup") and not r.get("error")
    ]
    errors = [r for r in read_jsonl(d / "queries.jsonl") if not r.get("warmup") and r.get("error")]
    calls = [c for c in read_jsonl(d / "calls.jsonl") if not c.get("warmup")]
    calls_by_req = defaultdict(list)
    for c in calls:
        calls_by_req[c["request_index"]].append(c)
    measured = {f"{r['request_index']}:{r['query_id']}" for r in rows + errors}
    sampled = end.get("engine_metrics_sampled") or {}
    cd = sampled.get("counter_delta") or {}
    qn = cd.get("queue_time_count")
    lat = [r["latency"] for r in rows]
    corr = [r["latency"] for r in rows if r.get("corrective_action") not in (None, "correct")]
    comp = meta.get("comparability") or {}
    ws = comp.get("web_search") or {}
    return {
        "run": str(d),
        "policy": (cfg_sched.get("policy") or {}).get("name") if cfg_sched else "none",
        "K": cfg_sched.get("slots") if cfg_sched else None,
        "engine_priority": cfg_sched.get("engine_priority") if cfg_sched else None,
        "conditions": {
            "workload_sha256": (meta.get("workload") or {}).get("sha256"),
            "mode": meta.get("mode"),
            "grading": meta.get("grading"),
            "engine_profile": (comp.get("llm") or {}).get("model"),
            "web_search": {k: ws.get(k) for k in ("provider", "latency_mode")},
            "cost_model_type": (meta.get("cost_model") or {}).get("type", "single")
            if meta.get("cost_model")
            else None,
            "leakage_check": ((meta.get("preflight") or {}).get("cost_model_leakage") or {}).get(
                "status"
            ),
        },
        "n_requests": len(rows),
        "n_errors": len(errors),
        "dispatch": dispatch_metrics(read_jsonl(d / "scheduler_log.jsonl"), measured),
        "calls_gate_wait_mean_s": mean([c["gate_wait"] for c in calls]),
        "estimates": estimate_metrics(rows, calls_by_req),
        "vllm": {
            "samples": sampled.get("n_samples"),
            "running_mean": (sampled.get("running") or {}).get("mean"),
            "running_max": (sampled.get("running") or {}).get("max"),
            "waiting_mean": (sampled.get("waiting") or {}).get("mean"),
            "waiting_max": (sampled.get("waiting") or {}).get("max"),
            "kv_usage_max": (sampled.get("kv_usage") or {}).get("max"),
            "queue_time_mean_s": (cd["queue_time_sum"] / qn) if qn else None,
            "preemptions": cd.get("preemptions"),
        },
        "ttft_mean_s": {
            k: mean([c["ttft"] for c in calls if c["kind"] == k and c.get("ttft") is not None])
            for k in sorted({c["kind"] for c in calls})
        },
        "wall_mean_s": {
            k: mean([c["wall"] for c in calls if c["kind"] == k])
            for k in sorted({c["kind"] for c in calls})
        },
        "latency": {
            "mean": mean(lat),
            "p95": pct(lat, 95),
            "corrected_mean": mean(corr),
            "corrected_p95": pct(corr, 95),
            "n_corrected": len(corr),
            "throughput_rps": summ.get("throughput_rps"),
        },
        "web_replay": end.get("web_search"),
    }


def consistency(runs: list[dict]) -> list[str]:
    warns = []
    for key in ("workload_sha256", "mode", "grading", "engine_profile", "web_search"):
        vals = {json.dumps(r["conditions"].get(key), sort_keys=True) for r in runs}
        if len(vals) > 1:
            warns.append(f"조건 불일치: {key} = {sorted(vals)}")
    if len({r["K"] for r in runs}) > 1:
        warns.append(f"K 불일치: {[r['K'] for r in runs]}")
    for r in runs:
        d = r["dispatch"]
        if r["policy"] == "fcfs" and d.get("available") and d["reorder_dispatches"]:
            warns.append(
                f"FCFS 인데 재정렬 {d['reorder_dispatches']}건: gate 로그 확인 필요 ({r['run']})"
            )
        if r["n_errors"]:
            warns.append(f"오류 요청 {r['n_errors']}건 ({r['policy']})")
    return warns


def flags(r: dict) -> list[str]:
    out = []
    d = r["dispatch"]
    if d.get("available") and d["choice_ge2"] is not None and d["choice_ge2"] < 0.1:
        out.append("대기열 2 이상인 dispatch < 10%: 정책이 순서를 바꿀 기회가 거의 없음")
    if (r["vllm"]["waiting_max"] or 0) > 0:
        out.append("vLLM 내부 대기 발생: 일부 대기는 client gate 밖에서 일어남")
    if r["vllm"]["preemptions"]:
        out.append("vLLM preemption 발생")
    sp = (r["estimates"] or {}).get("spearman_vs_actual")
    if sp is not None and sp < 0.3:
        out.append(f"추정치와 실제 남은 시간의 순위 상관 낮음 ({sp:.2f})")
    return out


# ---------------------------------------------------------------- output
def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def summary_md(runs: list[dict], warns: list[str], name: str) -> str:
    cols = [
        ("policy", lambda r: r["policy"]),
        ("K", lambda r: r["K"]),
        ("n", lambda r: r["n_requests"]),
        ("lat mean", lambda r: r["latency"]["mean"]),
        ("lat p95", lambda r: r["latency"]["p95"]),
        ("corr mean", lambda r: r["latency"]["corrected_mean"]),
        ("corr p95", lambda r: r["latency"]["corrected_p95"]),
        ("choice>=2", lambda r: r["dispatch"].get("choice_ge2")),
        ("queue mean", lambda r: r["dispatch"].get("queue_len_mean")),
        ("reorder", lambda r: r["dispatch"].get("reorder_rate")),
        ("gate wait", lambda r: r["dispatch"].get("gate_wait_mean_s")),
        ("est rho", lambda r: (r["estimates"] or {}).get("spearman_vs_actual")),
        ("vLLM wait max", lambda r: r["vllm"]["waiting_max"]),
        ("vLLM queue s", lambda r: r["vllm"]["queue_time_mean_s"]),
    ]
    lines = [
        f"# 스케줄링 진단 ({name})",
        "",
        "| " + " | ".join(c for c, _ in cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in runs:
        lines.append("| " + " | ".join(fmt(f(r)) for _, f in cols) + " |")
    lines += [
        "",
        "- choice>=2: 대기열 2 이상 dispatch 비율, reorder: FCFS 순서를 바꾼 dispatch 비율",
        "- est rho: 비용 추정치와 실제 남은 호출 시간의 Spearman 순위 상관",
        "- gate wait: gate 안 대기 평균 (초), vLLM queue s: vLLM 내부 대기 평균 (집계)",
        "",
    ]
    lines.append("## 조건 확인")
    lines += [f"- {w}" for w in warns] or ["- 비교 조건 일치"]
    lines.append("")
    for r in runs:
        lines.append(f"## {r['policy']} ({Path(r['run']).name})")
        fl = flags(r)
        lines += [f"- {x}" for x in fl] or ["- 특이사항 없음"]
        pr = (r["estimates"] or {}).get("post_grading_ratio")
        if pr:
            lines.append(f"- grading 이후 추정치 / grading 마지막 추정치: {json.dumps(pr)}")
        lines.append("")
    return "\n".join(lines)


def per_run_rows(runs: list[dict]) -> list[dict]:
    out = []
    for r in runs:
        d, e, v, la = r["dispatch"], r["estimates"] or {}, r["vllm"], r["latency"]
        out.append(
            {
                "policy": r["policy"],
                "K": r["K"],
                "run": r["run"],
                "n": r["n_requests"],
                "errors": r["n_errors"],
                "lat_mean": la["mean"],
                "lat_p95": la["p95"],
                "corr_mean": la["corrected_mean"],
                "corr_p95": la["corrected_p95"],
                "throughput_rps": la["throughput_rps"],
                "choice_ge2": d.get("choice_ge2"),
                "queue_len_mean": d.get("queue_len_mean"),
                "reorder_rate": d.get("reorder_rate"),
                "reorder_rate_given_choice": d.get("reorder_rate_given_choice"),
                "gate_wait_mean_s": d.get("gate_wait_mean_s"),
                "gate_wait_p95_s": d.get("gate_wait_p95_s"),
                "est_spearman": e.get("spearman_vs_actual"),
                "vllm_waiting_max": v["waiting_max"],
                "vllm_queue_time_mean_s": v["queue_time_mean_s"],
                "preemptions": v["preemptions"],
            }
        )
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="비교할 run 폴더 (같은 workload)")
    ap.add_argument("--name", default="diag_sched")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    runs = [diag_run(r) for r in a.runs]
    warns = consistency(runs)
    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "created": datetime.now().isoformat(),
        "runs": runs,
        "warnings": warns,
        "flags": {r["run"]: flags(r) for r in runs},
        "note": "기술 통계. 정책 간 지연 차이의 신뢰구간은 analyze_runs.py 로",
    }
    (out / "diag.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    rows = per_run_rows(runs)
    with open(out / "per_run.csv", "x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = summary_md(runs, warns, a.name)
    (out / "summary.md").write_text(md, encoding="utf-8")
    if not a.quiet:
        print(md)
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
