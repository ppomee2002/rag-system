"""B / C / Cbn 결과 집계 (하나 이상의 run 폴더, repetition 평균 + 범위).

    uv run python experiments/c2_gating/analyze.py results/c2_gating/pre/<run_id> [...] --name \
        pre_1_4_16
-> results/c2_gating/analysis/<name>/{report.md, cells.csv, cells.json}

표 해석
  - 각 셀 (mode, concurrency) 값은 repetition 평균, [min, max] 는 repetition 범위. 반복 2회는 분산
    추정이 약하므로 차이가 범위 안에 있으면 "차이 없음" 으로 읽는다.
  - Cbn 의 저부하 이득 유지율 = (B - Cbn) / (B - C) (평균 E2E). B-C 가 범위 안이면 계산하지 않는다.
  - 고부하 throughput 손실 = 1 - thr(X) / thr(B).
  - 병목 확인 (4절 첫 단계): B 의 (a) c=1 직렬 대기 = Incorrect/Ambiguous 요청에서 grading 후
    시작하는
    correction 경로 시간, (b) 동시 요청 증가에 따른 T5 utilization·grading queue wait.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(run_dirs: list[str]) -> list[dict]:
    out = []
    for d in run_dirs:
        s = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
        for cid, c in s["conditions"].items():
            out.append({"run": d, "cond_id": cid, **c})
    return out


def g(d: dict, path: str):
    for k in path.split("."):
        if d is None:
            return None
        d = d.get(k)
    return d


METRICS = {
    "throughput_rps": "throughput_rps",
    "e2e_mean": "e2e.mean",
    "e2e_p95": "e2e.p95",
    "e2e_correct": "e2e_by_action.correct.mean",
    "e2e_ambiguous": "e2e_by_action.ambiguous.mean",
    "e2e_incorrect": "e2e_by_action.incorrect.mean",
    "t5_grading_s": "t5.busy_per_request_mean_s.grading",
    "t5_internal_s": "t5.busy_per_request_mean_s.internal_refine",
    "t5_ext_required_s": "t5.busy_per_request_mean_s.external_required",
    "t5_ext_wasted_s": "t5.busy_per_request_mean_s.external_wasted",
    "t5_util": "t5.utilization",
    "qwait_grading_mean": "t5.queue_wait_s.grading.mean",
    "qwait_grading_p95": "t5.queue_wait_s.grading.p95",
    "qwait_ext_mean": "t5.queue_wait_s.external_required.mean",
    "waste_rewrite_s": "waste.rewrite_s",
    "waste_web_s": "waste.web_s",
    "waste_n_rewrite": "waste.n_rewrite",
    "official_match": "accuracy.official_match",
    "acc": "accuracy.acc",
    "n_errors": "n_errors",
}


def cells(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["cond"]["mode"], r["cond"]["concurrency"])].append(r)
    out = {}
    for key, rs in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cell = {"n_reps": len(rs), "runs": sorted({r["run"] for r in rs})}
        for name, path in METRICS.items():
            vals = [g(r, path) for r in rs]
            vals = [v for v in vals if v is not None]
            cell[name] = {
                "mean": sum(vals) / len(vals) if vals else None,
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
            }
        gpu = [r.get("gpu") or {} for r in rs]
        cell["gpu_util"] = {
            k: sum((x.get(k) or {}).get("util_mean") or 0 for x in gpu) / len(gpu)
            for k in sorted({k for x in gpu for k in x})
        }
        cell["leak_ok"] = all(r["leak"]["ok"] for r in rs)
        cell["cbn_early_submit"] = sum(r["external_submit_before_decision"] for r in rs)
        cell["correct_ext_jobs_executed"] = sum(
            r["correct_external_refine"]["n_jobs_executed"] for r in rs
        )
        out[f"{key[0]}@c{key[1]}"] = cell
    return out


def _m(cell, k):
    return (cell.get(k) or {}).get("mean")


def overlap_ranges(a: dict, b: dict) -> bool:
    return not (a["max"] < b["min"] or b["max"] < a["min"])


def compare(cs: dict) -> list[dict]:
    concs = sorted({int(k.split("@c")[1]) for k in cs})
    out = []
    for c in concs:
        B, C, N = cs.get(f"B@c{c}"), cs.get(f"C@c{c}"), cs.get(f"Cbn@c{c}")
        if not B:
            continue
        row = {"concurrency": c}
        for name, X in (("C", C), ("Cbn", N)):
            if not X:
                continue
            row[f"{name}_thr_loss_vs_B"] = 1 - _m(X, "throughput_rps") / _m(B, "throughput_rps")
            row[f"{name}_e2e_ratio_vs_B"] = _m(X, "e2e_mean") / _m(B, "e2e_mean")
            row[f"{name}_p95_ratio_vs_B"] = _m(X, "e2e_p95") / _m(B, "e2e_p95")
            row[f"{name}_thr_within_rep_range_of_B"] = overlap_ranges(
                X["throughput_rps"], B["throughput_rps"]
            )
        if C and N:
            gain_c = _m(B, "e2e_mean") - _m(C, "e2e_mean")
            sep = not overlap_ranges(B["e2e_mean"], C["e2e_mean"])
            row["C_gain_s"] = gain_c
            row["Cbn_gain_s"] = _m(B, "e2e_mean") - _m(N, "e2e_mean")
            row["Cbn_retention_of_C_gain"] = (
                (row["Cbn_gain_s"] / gain_c) if sep and gain_c else None
            )
        out.append(row)
    return out


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def report(cs: dict, cmp: list[dict], name: str) -> str:
    L = [
        f"# C2 gating — {name}",
        "",
        "값 = repetition 평균 [min, max]. mock run 이면 수치는 의미 없음.",
        "",
        "## 1. 조건별",
        "",
        "| 셀 | reps | thr (req/s) | E2E mean | P95 | correct | ambiguous | incorrect | 오류 "
        "| 누수 OK |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, c in cs.items():

        def r(m, c=c):
            x = c[m]
            return f"{fmt(x['mean'])} [{fmt(x['min'])}, {fmt(x['max'])}]"

        L.append(
            f"| {k} | {c['n_reps']} | {r('throughput_rps')} | {r('e2e_mean')} | {r('e2e_p95')} | "
            f"{fmt(_m(c, 'e2e_correct'))} | {fmt(_m(c, 'e2e_ambiguous'))} | "
            f"{fmt(_m(c, 'e2e_incorrect'))} | {fmt(_m(c, 'n_errors'), 0)} | {c['leak_ok']} |"
        )
    L += [
        "",
        "## 2. T5 (요청당 busy 초, utilization, queue wait)",
        "",
        "| 셀 | grading | internal | ext 유효 | ext 낭비 | util | qwait grading mean / p95 | "
        "qwait ext | "
        "Correct ext 실행 job | Cbn 판정 전 submit | GPU util |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, c in cs.items():
        L.append(
            f"| {k} | {fmt(_m(c, 't5_grading_s'), 4)} | {fmt(_m(c, 't5_internal_s'), 4)} | "
            f"{fmt(_m(c, 't5_ext_required_s'), 4)} | {fmt(_m(c, 't5_ext_wasted_s'), 4)} | "
            f"{fmt(_m(c, 't5_util'))} | {fmt(_m(c, 'qwait_grading_mean'))} / "
            f"{fmt(_m(c, 'qwait_grading_p95'))} | {fmt(_m(c, 'qwait_ext_mean'))} | "
            f"{c['correct_ext_jobs_executed']} | {c['cbn_early_submit']} | "
            f"{ {g_: round(v, 1) for g_, v in c['gpu_util'].items()} } |"
        )
    L += [
        "",
        "## 3. rewrite / web 낭비, 정확도",
        "",
        "| 셀 | 낭비 rewrite 수 | 낭비 rewrite s | 낭비 web s | official_match | acc |",
        "|---|---|---|---|---|---|",
    ]
    for k, c in cs.items():
        L.append(
            f"| {k} | {fmt(_m(c, 'waste_n_rewrite'), 1)} | {fmt(_m(c, 'waste_rewrite_s'))} | "
            f"{fmt(_m(c, 'waste_web_s'))} | {fmt(_m(c, 'official_match'), 4)} | "
            f"{fmt(_m(c, 'acc'), 4)} |"
        )
    L += [
        "",
        "## 4. B 대비",
        "",
        "| c | C thr 손실 | Cbn thr 손실 | C E2E 비 | Cbn E2E 비 | C P95 비 | Cbn P95 비 | "
        "C 이득 s | Cbn 이득 s | Cbn 유지율 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cmp:
        L.append(
            f"| {r['concurrency']} | {fmt(r.get('C_thr_loss_vs_B'))} | "
            f"{fmt(r.get('Cbn_thr_loss_vs_B'))} | {fmt(r.get('C_e2e_ratio_vs_B'))} | "
            f"{fmt(r.get('Cbn_e2e_ratio_vs_B'))} | {fmt(r.get('C_p95_ratio_vs_B'))} | "
            f"{fmt(r.get('Cbn_p95_ratio_vs_B'))} | {fmt(r.get('C_gain_s'))} | "
            f"{fmt(r.get('Cbn_gain_s'))} | {fmt(r.get('Cbn_retention_of_C_gain'))} |"
        )
    L += [
        "",
        "thr 손실 > 0 = B 보다 느림. 유지율은 B 와 C 의 E2E 범위가 겹치지 않을 때만 계산.",
        "thr 차이가 repetition 범위 안이면 (cells.json `*_thr_within_rep_range_of_B`) 차이로 보지 "
        "않는다.",
        "",
    ]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    rows = load(a.runs)
    cs = cells(rows)
    cmp = compare(cs)
    d = ROOT / "results" / "c2_gating" / "analysis" / a.name
    d.mkdir(parents=True, exist_ok=False)
    (d / "cells.json").write_text(
        json.dumps({"cells": cs, "vs_B": cmp}, indent=2), encoding="utf-8"
    )
    with open(d / "cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", *METRICS])
        for k, c in cs.items():
            w.writerow([k, *[c[m]["mean"] for m in METRICS]])
    text = report(cs, cmp, a.name)
    (d / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"saved -> {d}")


if __name__ == "__main__":
    main()
