"""조건 (mode x concurrency x repetition) 단위 집계와 조건 간 동일성 검사.

T5 시간 분류 (요청당):
  grading            판정용 10문서
  internal_refine    correct / ambiguous 의 internal refinement (memo hit 은 forward 없음)
  external_required  ambiguous / incorrect 의 external refinement (필요한 작업)
  external_wasted    correct 요청의 external refinement (C 의 speculation 낭비. B/Cbn 은 0 이어야
    함)
낭비 rewrite / web: correct 로 판정된 요청에서 speculative 로 실행된 rewrite / web 시간.
"""

from __future__ import annotations

import math
from collections import defaultdict


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def dist(xs) -> dict:
    xs = [x for x in xs if x is not None]
    return {
        "n": len(xs),
        "mean": mean(xs),
        "p50": pct(xs, 0.5),
        "p95": pct(xs, 0.95),
        "p99": pct(xs, 0.99),
        "max": max(xs) if xs else None,
    }


def overlap(a0, a1, b0, b1) -> float:
    if None in (a0, a1, b0, b1):
        return 0.0
    return max(0.0, min(a1, b1) - max(a0, b0))


def job_category(job: dict, action: str | None) -> str:
    k = job["kind"]
    if k == "external_refine":
        return "external_wasted" if action == "correct" else "external_required"
    return k


def condition_summary(
    rows: list[dict], jobs: list[dict], chains: list[dict], makespan: float, gpu: dict | None
) -> dict:
    ok = [r for r in rows if not r.get("error")]
    chain_of = {c["request_id"]: c for c in chains}
    jobs_by_req: dict[str, list[dict]] = defaultdict(list)
    for j in jobs:
        jobs_by_req[j["request_id"]].append(j)

    # ---- T5 시간 (요청당)
    per_req = {
        c: [] for c in ("grading", "internal_refine", "external_required", "external_wasted")
    }
    qwait = defaultdict(list)
    for r in ok:
        acc = dict.fromkeys(per_req, 0.0)
        for j in jobs_by_req.get(r["request_id"], []):
            cat = job_category(j, r["action"])
            acc[cat] = acc.get(cat, 0.0) + (j["busy_s"] or 0.0)
            if j["status"] != "cancelled" or j.get("t_start") is not None:
                qwait[cat].append(j["queue_wait_s"])
        for c in per_req:
            per_req[c].append(acc[c])
    busy_total = sum(j["busy_s"] or 0.0 for j in jobs)

    # ---- correct 요청의 external refinement
    corr = [r for r in ok if r["action"] == "correct"]
    ext_corr = [
        j
        for r in corr
        for j in jobs_by_req.get(r["request_id"], [])
        if j["kind"] == "external_refine"
    ]
    # ---- Cbn: 판정 전 submit
    decided = {r["request_id"]: r["decided_at"] for r in ok}
    early = [
        j
        for j in jobs
        if j["kind"] == "external_refine"
        and j["request_id"] in decided
        and j["t_submit"] < decided[j["request_id"]]
    ]

    # ---- rewrite / web 낭비, 실행 중첩
    waste = {"rewrite_s": 0.0, "web_s": 0.0, "n_rewrite": 0, "n_web": 0, "rewrite_out_tokens": 0}
    ov_stage, ov_exec, n_spec = [], [], 0
    for r in ok:
        c = chain_of.get(r["request_id"])
        if c is None:
            continue
        steps = c["steps"]
        if r["action"] == "correct" and c["speculative"]:
            if "rewrite" in steps:
                waste["n_rewrite"] += 1
                waste["rewrite_s"] += steps["rewrite"].get("wall") or 0.0
                waste["rewrite_out_tokens"] += sum(
                    (x.get("out_tokens") or 0) for x in c["calls"] if x["kind"] == "rewrite"
                )
            if "web_search" in steps:
                waste["n_web"] += 1
                waste["web_s"] += steps["web_search"].get("wall") or 0.0
        if c["speculative"]:
            n_spec += 1
            g0 = r["decided_at"] - r["grading_wall"]
            corr0 = steps.get("rewrite", {}).get("t_start")
            corr1 = (steps.get("web_search") or steps.get("rewrite") or {}).get("t_end")
            ov_stage.append(overlap(g0, r["decided_at"], corr0, corr1))
            gj = next((j for j in jobs_by_req[r["request_id"]] if j["kind"] == "grading"), None)
            rw = next((x for x in c["calls"] if x["kind"] == "rewrite"), None)
            ov_exec.append(
                overlap(gj["t_start"], gj["t_end"], rw["t_start"], rw["t_end"])
                if gj and rw
                else 0.0
            )

    by_action = defaultdict(list)
    for r in ok:
        by_action[r["action"]].append(r["e2e_latency"])
    om = [r["official_match"] for r in ok if r.get("official_match") is not None]
    ac = [r["acc"] for r in ok if r.get("acc") is not None]
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "n_errors": len(rows) - len(ok),
        "errors": [r["error"] for r in rows if r.get("error")][:10],
        "makespan_s": makespan,
        "throughput_rps": len(ok) / makespan if makespan > 0 else None,
        "e2e": dist([r["e2e_latency"] for r in ok]),
        "e2e_by_action": {a: dist(v) for a, v in sorted(by_action.items())},
        "action_counts": {a: len(v) for a, v in sorted(by_action.items())},
        "accuracy": {
            "official_match": mean(om),
            "acc": mean(ac),
            "n_scored": len(om),
        },
        "t5": {
            "busy_per_request_mean_s": {c: mean(v) for c, v in per_req.items()},
            "busy_total_s": busy_total,
            "utilization": busy_total / makespan if makespan > 0 else None,
            "queue_wait_s": {c: dist(v) for c, v in qwait.items()},
            "n_jobs": len(jobs),
            "memo_hits": sum(j["n_memo_hits"] for j in jobs),
            "memo_mismatch": sum(j["memo_mismatch"] for j in jobs),
            "max_memo_abs_diff": max([j["max_memo_abs_diff"] for j in jobs] or [0.0]),
            "n_forward_pairs": sum(j["n_forward"] for j in jobs),
            "cancel_effects": _count(
                j.get("cancel_effect") for j in jobs if j.get("cancel_effect")
            ),
        },
        "correct_external_refine": {
            "n_correct": len(corr),
            "n_jobs_submitted": len(ext_corr),
            "n_jobs_executed": sum(1 for j in ext_corr if j["n_forward"] > 0),
            "n_pairs_forwarded": sum(j["n_forward"] for j in ext_corr),
            "busy_s": sum(j["busy_s"] or 0.0 for j in ext_corr),
        },
        "external_submit_before_decision": len(early),
        "generation_precondition": {
            "n_checked": sum(1 for r in ok if r.get("generation_precondition")),
            "n_violations": sum(
                1 for r in ok if not (r.get("generation_precondition") or {}).get("ok", False)
            ),
            "problems": _count(
                p for r in ok for p in (r.get("generation_precondition") or {}).get("problems", [])
            ),
        },
        "waste": waste,
        "overlap": {
            "n_speculative": n_spec,
            "stage_overlap_frac": (sum(x > 0 for x in ov_stage) / n_spec) if n_spec else None,
            "stage_overlap_mean_s": mean(ov_stage),
            "exec_overlap_frac": (sum(x > 0 for x in ov_exec) / n_spec) if n_spec else None,
            "exec_overlap_mean_s": mean(ov_exec),
        },
        "chain_status": _count(c["status"] for c in chains),
        "gpu": gpu,
    }


def _count(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


IDENTITY_KEYS = ("action", "evidence_id", "evidence_sha16", "prompt_sha16")


def identity_check(cond_rows: dict[str, list[dict]]) -> dict:
    """조건별 행 -> 같은 index 끼리 action / evidence / prompt / prediction 비교."""
    by_idx: dict[int, dict[str, dict]] = defaultdict(dict)
    for cond, rows in cond_rows.items():
        for r in rows:
            if not r.get("error"):
                by_idx[r["index"]][cond] = r
    mism = {k: [] for k in (*IDENTITY_KEYS, "prediction")}
    for idx, per in by_idx.items():
        for k in mism:
            vals = {c: r.get(k) for c, r in per.items()}
            if len(set(map(str, vals.values()))) > 1:
                mism[k].append({"index": idx, "values": vals})
    n = len(by_idx)
    return {
        "n_indices": n,
        "conditions": sorted(cond_rows),
        "all_identical": {k: not v for k, v in mism.items()},
        "n_mismatch": {k: len(v) for k, v in mism.items()},
        "examples": {k: v[:5] for k, v in mism.items() if v},
        "note": "prediction 불일치는 evidence/prompt 가 같아도 vLLM batch 구성 차이 (동시 요청) 로 "
        "greedy 출력이 달라진 경우일 수 있다",
    }
