"""스케줄링 검증 실험 (SV) 분석 (읽기 전용, 추가 추론·API 호출 없음).

실험 A (경로별 작업량) 와 실험 B / C (같은 workload 의 정책 비교) 를 한 번에 정리한다.

    uv run python experiments/crag_system/analyze_sched_validation.py \
        --run none=<dir> --run fcfs=<dir> --run srpt_est=<dir> [...] --ref fcfs \
        [--gpu-log none=<nvidia-smi csv> ...] [--paths-only] --name sv-rho0.8

  --run label=dir   run 폴더 (runner 결과: queries.jsonl, calls.jsonl, run_meta.json, run_end.json)
  --ref label       paired 차이의 기준 run (기본: 첫 번째 --run)
  --gpu-log         run 동안 기록한 `nvidia-smi --query-gpu=timestamp,index,utilization.gpu,...
                    --format=csv,noheader,nounits -lms 500` 출력. 없으면 GPU 사용률은 "-"
  --paths-only      실험 A 표만 (sequential run 하나도 가능)

지표 (warmup·오류 요청 제외)
  latency            완료 - 도착 (runner `latency`). 모든 조건에서 같은 시작·종료점
  throughput         n / (마지막 완료 - 첫 도착)  (runner summary.throughput_rps)
  gate_wait          요청별 gate 누적 대기 (query_state.cum_wait). gate 없는 run 은 0
  vllm_queue_s       vLLM /metrics request_queue_time 증가분 평균 (run.sample_engine_metrics_sec
                     를 켠 run 만). 요청별 vLLM 대기는 측정 불가 (집계값)
  gpu_util           --gpu-log 의 utilization.gpu 평균 (기록 구간 전체, warmup 포함)
  paired             같은 (request_index, query_id) 끼리 짝지어 run - ref. 평균 차이와 p95 차이의
                     bootstrap 95% CI (요청을 복원 추출, 2000회). 기술 통계

조건 일치 검사는 diag_sched.consistency (workload hash, mode, grading, 엔진, web search, K) 를 쓴다.
B 실험에서 none (gate 없음) 은 K 가 없으므로 K 불일치 경고가 나오는 것이 정상이다.

결과: results/crag_system/analysis/<name>/<timestamp>/ (덮어쓰지 않음)
  report.md, per_run.csv, paired.csv, paths.csv, analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_sched import consistency, diag_run, mean, pct, read_json, read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GRADE_KINDS = ("grade", "grade_batch")
N_BOOT = 2000


# ---------------------------------------------------------------- loading
def load_run(run_dir: str | Path) -> dict:
    d = Path(run_dir)
    rows = [r for r in read_jsonl(d / "queries.jsonl") if not r.get("warmup")]
    ok = [r for r in rows if not r.get("error")]
    calls = defaultdict(list)
    for c in read_jsonl(d / "calls.jsonl"):
        if not c.get("warmup") and c.get("request_index", -1) >= 0:
            calls[c["request_index"]].append(c)
    return {
        "dir": str(d),
        "rows": ok,
        "n_errors": len(rows) - len(ok),
        "calls": calls,
        "summary": read_json(d / "summary.json"),
        "end": read_json(d / "run_end.json"),
        "meta": read_json(d / "run_meta.json"),
    }


def is_corrected(r: dict) -> bool:
    qs = r.get("query_state") or {}
    if qs.get("corrected") is not None:
        return bool(qs["corrected"])
    return r.get("corrective_action") not in (None, "correct")


def gpu_util(path: str | Path | None) -> dict | None:
    """nvidia-smi csv (timestamp, index, utilization.gpu, memory.used ...) -> GPU 별 평균 사용률."""
    if not path or not Path(path).exists():
        return None
    by_gpu = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            util = float(parts[2])
        except ValueError:
            continue  # header 등
        by_gpu[parts[1]].append(util)
    if not by_gpu:
        return None
    return {g: {"mean": float(np.mean(v)), "n": len(v)} for g, v in sorted(by_gpu.items())}


# ---------------------------------------------------------------- experiment A
def _path_row(label: str, run: dict, path: str, rs: list[dict], n_all: int) -> dict:
    per = [(run["calls"].get(r["request_index"], []), r) for r in rs]

    def wall(kinds):
        return mean([sum(c["wall"] for c in cs if c["kind"] in kinds) for cs, _ in per])

    def stage(k):
        return mean([(r.get("stage_latencies") or {}).get(k, 0.0) for r in rs])

    lat = [r["latency"] for r in rs]
    return {
        "run": label,
        "path": path,
        "n": len(rs),
        "share": len(rs) / n_all,
        "llm_calls": mean([len(cs) for cs, _ in per]),
        "grade_calls": mean([sum(c["kind"] in GRADE_KINDS for c in cs) for cs, _ in per]),
        "llm_wall_s": mean([sum(c["wall"] for c in cs) for cs, _ in per]),
        "grade_wall_s": wall(GRADE_KINDS),
        "rewrite_wall_s": wall(("rewrite",)),
        "generate_wall_s": wall(("generate",)),
        "prompt_tokens": mean([sum(c.get("prompt_tokens") or 0 for c in cs) for cs, _ in per]),
        "retrieval_s": stage("retrieval"),
        "web_search_s": stage("web_search"),
        "gate_wait_s": mean([(r.get("query_state") or {}).get("cum_wait", 0.0) for r in rs]),
        "latency_mean": mean(lat),
        "latency_p95": pct(lat, 95),
    }


def path_rows(label: str, run: dict) -> list[dict]:
    groups = defaultdict(list)
    for r in run["rows"]:
        groups["corrected" if is_corrected(r) else "not_corrected"].append(r)
    groups["all"] = list(run["rows"])
    n_all = len(run["rows"]) or 1
    return [
        _path_row(label, run, path, groups[path], n_all)
        for path in ("not_corrected", "corrected", "all")
        if groups.get(path)
    ]


# ---------------------------------------------------------------- experiment B / C
def run_row(label: str, run: dict, diag: dict, gpu: dict | None) -> dict:
    rows = run["rows"]
    lat = [r["latency"] for r in rows]
    corr = [r["latency"] for r in rows if is_corrected(r)]
    nc = [r["latency"] for r in rows if not is_corrected(r)]
    gw = [(r.get("query_state") or {}).get("cum_wait", 0.0) for r in rows]
    gen_ttft = [
        c["ttft"]
        for cs in run["calls"].values()
        for c in cs
        if c["kind"] == "generate" and c.get("ttft") is not None
    ]
    v = diag["vllm"]
    return {
        "run": label,
        "policy": diag["policy"],
        "K": diag["K"],
        "n": len(rows),
        "errors": run["n_errors"],
        "lat_mean": mean(lat),
        "lat_p50": pct(lat, 50),
        "lat_p95": pct(lat, 95),
        "lat_p99": pct(lat, 99),
        "throughput_qps": run["summary"].get("throughput_rps"),
        "n_corrected": len(corr),
        "corr_mean": mean(corr),
        "corr_p95": pct(corr, 95),
        "notcorr_mean": mean(nc),
        "notcorr_p95": pct(nc, 95),
        "gate_wait_mean": mean(gw),
        "gate_wait_p95": pct(gw, 95),
        "vllm_queue_s": v["queue_time_mean_s"],
        "vllm_waiting_max": v["waiting_max"],
        "vllm_running_mean": v["running_mean"],
        "preemptions": v["preemptions"],
        "gpu_util": None if not gpu else mean([g["mean"] for g in gpu.values()]),
        "generate_ttft": mean(gen_ttft),
        "est_spearman": (diag["estimates"] or {}).get("spearman_vs_actual"),
        "choice_ge2": diag["dispatch"].get("choice_ge2"),
    }


def _key(r):
    return (r["request_index"], r["query_id"])


def paired(label: str, run: dict, ref_label: str, ref: dict, seed: int = 0) -> list[dict]:
    a = {_key(r): r for r in run["rows"]}
    b = {_key(r): r for r in ref["rows"]}
    keys = sorted(set(a) & set(b))
    out = []
    subsets = {
        "all": keys,
        # 경로는 기준 run 의 결과로 나눈다 (같은 요청이 run 마다 다른 경로를 탈 수 있음)
        "corrected(ref)": [k for k in keys if is_corrected(b[k])],
        "not_corrected(ref)": [k for k in keys if not is_corrected(b[k])],
    }
    path_mismatch = sum(is_corrected(a[k]) != is_corrected(b[k]) for k in keys)
    rng = np.random.default_rng(seed)
    for subset, ks in subsets.items():
        if len(ks) < 2:
            continue
        x = np.array([a[k]["latency"] for k in ks])
        y = np.array([b[k]["latency"] for k in ks])
        idx = rng.integers(0, len(ks), (N_BOOT, len(ks)))
        d_mean = (x[idx] - y[idx]).mean(axis=1)
        d_p95 = np.percentile(x[idx], 95, axis=1) - np.percentile(y[idx], 95, axis=1)
        out.append(
            {
                "run": label,
                "ref": ref_label,
                "subset": subset,
                "n_pairs": len(ks),
                "mean_diff": float(x.mean() - y.mean()),
                "mean_rel": float(x.mean() / y.mean() - 1) if y.mean() else None,
                "mean_ci_lo": float(np.percentile(d_mean, 2.5)),
                "mean_ci_hi": float(np.percentile(d_mean, 97.5)),
                "p95_diff": float(np.percentile(x, 95) - np.percentile(y, 95)),
                "p95_ci_lo": float(np.percentile(d_p95, 2.5)),
                "p95_ci_hi": float(np.percentile(d_p95, 97.5)),
                "path_mismatch": path_mismatch,
            }
        )
    return out


# ---------------------------------------------------------------- output
def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        if np.isnan(v):
            return "-"
        return f"{v:.{nd}f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return lines


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def parse_pairs(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{what} 는 label=path 형식: {it}")
        k, v = it.split("=", 1)
        if k in out:
            raise SystemExit(f"{what} label 중복: {k}")
        out[k] = v
    return out


def analyze(
    runs: dict[str, str],
    ref: str | None = None,
    gpu_logs: dict[str, str] | None = None,
    paths_only: bool = False,
) -> dict:
    gpu_logs = gpu_logs or {}
    loaded = {k: load_run(v) for k, v in runs.items()}
    for k, r in loaded.items():
        if not r["rows"]:
            raise SystemExit(f"측정 요청이 없음: {k} ({r['dir']})")
    paths = [row for k, r in loaded.items() for row in path_rows(k, r)]
    res = {"runs": runs, "paths": paths}
    if paths_only:
        return res
    ref = ref or next(iter(runs))
    if ref not in loaded:
        raise SystemExit(f"--ref {ref} 가 --run 에 없음")
    diags = {k: diag_run(v) for k, v in runs.items()}
    res["ref"] = ref
    res["warnings"] = consistency(list(diags.values()))
    res["per_run"] = [run_row(k, loaded[k], diags[k], gpu_util(gpu_logs.get(k))) for k in loaded]
    res["paired"] = [
        row for k in loaded if k != ref for row in paired(k, loaded[k], ref, loaded[ref])
    ]
    return res


def report_md(res: dict, name: str) -> str:
    lines = [f"# 스케줄링 검증 분석 ({name})", ""]
    if "per_run" in res:
        lines += ["## 조건별 요약 (실험 B / C)", ""]
        lines += md_table(
            res["per_run"],
            [
                "run", "policy", "K", "n", "errors", "lat_mean", "lat_p50", "lat_p95", "lat_p99",
                "throughput_qps", "gate_wait_mean", "gate_wait_p95", "vllm_queue_s",
                "vllm_waiting_max", "preemptions", "gpu_util", "generate_ttft",
            ],
        )  # fmt: skip
        lines += ["", "### 경로별 (corrected = 해당 run 에서 correction 이 일어난 요청)", ""]
        lines += md_table(
            res["per_run"],
            ["run", "n_corrected", "corr_mean", "corr_p95", "notcorr_mean", "notcorr_p95",
             "est_spearman", "choice_ge2"],
        )  # fmt: skip
        lines += ["", f"## 기준 `{res['ref']}` 대비 paired 차이 (초, bootstrap 95% CI)", ""]
        lines += md_table(
            res["paired"],
            ["run", "subset", "n_pairs", "mean_diff", "mean_rel", "mean_ci_lo", "mean_ci_hi",
             "p95_diff", "p95_ci_lo", "p95_ci_hi", "path_mismatch"],
        )  # fmt: skip
        lines += ["", "## 조건 확인", ""]
        lines += [f"- {w}" for w in res["warnings"]] or ["- 비교 조건 일치"]
        lines += [""]
    lines += ["## 실행 경로별 작업량 (실험 A)", ""]
    lines += md_table(
        res["paths"],
        ["run", "path", "n", "share", "llm_calls", "grade_calls", "llm_wall_s", "grade_wall_s",
         "rewrite_wall_s", "generate_wall_s", "prompt_tokens", "retrieval_s", "web_search_s",
         "gate_wait_s", "latency_mean", "latency_p95"],
    )  # fmt: skip
    lines += [
        "",
        "- llm_wall_s: 요청의 LLM 호출 wall 합"
        " (부하가 있으면 batching·대기로 순수 서비스 시간보다 큼)",
        "- paired 의 corrected(ref)/not_corrected(ref) 는 기준 run 의 경로로 나눈 것."
        " path_mismatch = 두 run 에서 경로가 다른 요청 수 (batching 비결정성)",
        "- 기술 통계. 반복 (seed) 간 변동은 seed 별 분석을 비교해서 본다",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--gpu-log", action="append", default=[], metavar="LABEL=CSV")
    ap.add_argument("--paths-only", action="store_true")
    ap.add_argument("--name", default="sched_validation")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    res = analyze(
        parse_pairs(a.run, "--run"), a.ref, parse_pairs(a.gpu_log, "--gpu-log"), a.paths_only
    )
    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    res["created"] = datetime.now().isoformat()
    (out / "analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    write_csv(out / "paths.csv", res["paths"])
    write_csv(out / "per_run.csv", res.get("per_run", []))
    write_csv(out / "paired.csv", res.get("paired", []))
    md = report_md(res, a.name)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not a.quiet:
        print(md)
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
