"""EVC live 결과 정합성 및 queueing 진단 (읽기 전용, 추가 추론·API 호출 없음).

evidence preservation live run (run_evidence_cost.sh live, run_evidence_live.py) 을 기존 로그만으로
분석한다. 정책·그래프·scheduler 를 다시 실행하지 않고, 기존 결과 폴더에는 쓰지 않는다.

    uv run python experiments/crag_system/diag_evc_live.py \
        --live-root results/crag_system/pre/PRE-EVC-SV1-LIVE \
        --seq results/crag_system/pre/PRE-SV1-A/eval_seq \
        --evf results/crag_system/pre/PRE-EVF-SV1-EVAL/<ts> \
        [--evc-seq results/crag_system/pre/PRE-EVC-SV1-SEQ/<ts>] \
        [--sv-live-root results/crag_system/pre/PRE-SV1-LIVE] [--calib <calibration.json>] \
        --name EVC-SV1-live-diag

run 폴더: <live-root>/rho<rho>_s<seed>_<evidence policy>_K<k> (+ 같은 이름 .gpu.csv)

A 정합성·정확도  질문 집합 (seq / evf 공통 230 / evf 제외 / live), 중복·누락, seed 간 동일성,
                 부분집합별 정확도 (live / seq / evf), action 분포·전이, web replay hit / empty,
                 action_agree 의 분모
B 지연 분해      요청별 (시각은 모두 runner t0 기준, 같은 시계):
                   start_lag  = 스레드 시작 - 도착
                   gate_wait  = 호출 enqueue -> gate 통과 (외부 scheduler 대기) [관측]
                   exec       = gate 통과 -> 응답 완료 (vLLM 내부 대기 + prefill + decode) [관측]
                   wait_only  = |U[enqueue,end]| - |U[start,end]| (이 요청의 LLM 호출이 하나도
                                실행 중이지 않고 기다리기만 한 시간. 병렬 grading 중첩 제거)
                   retrieval / web_search = stage 시간 [관측]
                   other      = 나머지 (그래프·스레드 overhead)
                   noload     = 같은 호출의 부하 없는 실행시간:
                                seq / evc-seq 의 같은 호출 [관측·측정]
                                없으면 토큰 선형 모델 [추정]. inflation = exec - noload
                                (batching·GPU 공유·vLLM 내부 대기에 의한 실행시간 증가)
                 vLLM 내부 대기는 요청별로 관측되지 않음 (engine_metrics 의 집계값만)
C queueing       도착률·처리량·Little, 시간 구간별 N(t)·gate 대기 호출 수·실행 중 호출 수·
                 vLLM running / waiting / KV, 지연 추세 (도착 시각 회귀), offered no-load work
                 (λ x E[noload]) vs 달성 work rate, gate 포화 비율, 대기 있는데 slot 이 빈 시간 (H2)
D scheduling     호출 종류별 gate 대기, 대기 중 slot 을 generate 가 점유한 비율 (HOL),
                 요청별 no-load 작업량 분포, SRPT-est 추정치 (run_meta cost model 로 재계산) vs
                 no-load 작업량 (정책별 편향)

근거 표시: [관측] 로그 값 / [측정] evc-seq 측정 / [추정] 모델 / [재구성] 관측값의 조합.
결과: results/crag_system/analysis/evc_live/<name>/<timestamp>/ (덮어쓰지 않음)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_sched import mean, pct, read_json, read_jsonl, spearman  # noqa: E402
from diag_sv_policy import corrected, cost_model_of, load_run, qfeats  # noqa: E402
from evidence_cost import LIVE_DIR, _cfg_policy, efa, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = "crag_rule"
GRADE_KINDS = ("grade", "grade_batch")
KINDS = ("grade", "rewrite", "generate")


def kgroup(kind: str) -> str:
    return "grade" if kind in GRADE_KINDS else kind


def union_len(iv: list[tuple[float, float]]) -> float:
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + (cur_e - cur_s if cur_e is not None else 0.0)


def step_segments(series: dict[str, list[tuple[float, float]]], t_lo: float, t_hi: float):
    """{이름: 구간들} -> [(t0, t1, {이름: 동시 개수})] (구간 [s, e) 의 겹친 개수, 정확한 sweep)."""
    ev = []
    for name, iv in series.items():
        for s, e in iv:
            if e > s:
                ev += [(s, 1, name), (e, -1, name)]
    ev.sort(key=lambda x: (x[0], x[1]))
    cnt = dict.fromkeys(series, 0)
    out, t_prev = [], t_lo
    for t, d, name in ev:
        t = min(max(t, t_lo), t_hi)
        if t > t_prev:
            out.append((t_prev, t, dict(cnt)))
            t_prev = t
        cnt[name] += d
    if t_hi > t_prev:
        out.append((t_prev, t_hi, dict(cnt)))
    return out


def tw_mean(segs, f) -> float | None:
    tot = sum(t1 - t0 for t0, t1, _ in segs)
    return sum((t1 - t0) * f(c) for t0, t1, c in segs) / tot if tot > 0 else None


# ================================================================ inputs
def discover(root: Path) -> dict[tuple, Path]:
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = LIVE_DIR.match(d.name)
        if m and (d / "queries.jsonl").exists():
            out[(m["rho"], int(m["seed"]), m["policy"], int(m["k"]))] = d
    return out


def load_seq_rows(seq: Path) -> tuple[dict, dict, list[dict]]:
    """seq -> (qid -> 행, qid -> 호출 목록, 원시 행 (warmup 제외, 오류 포함))."""
    raw = [r for r in read_jsonl(seq / "queries.jsonl") if not r.get("warmup")]
    rows = {r["query_id"]: r for r in raw if not r.get("error")}
    calls = defaultdict(list)
    for c in read_jsonl(seq / "calls.jsonl"):
        if not c.get("warmup"):
            calls[c["query_id"]].append(c)
    return rows, calls, raw


def load_evf(evf: Path | None) -> dict:
    if evf is None:
        return {"items": {}, "gens": {}}
    items = {it["query_id"]: it for it in read_jsonl(evf / "items.jsonl")}
    gens = defaultdict(dict)
    for g in read_jsonl(evf / "generations.jsonl"):
        gens[g["condition"]][g["query_id"]] = g
    return {"items": items, "gens": gens}


def load_evc(evc: Path | None) -> dict[tuple[str, str], dict]:
    """evc-seq 측정 -> (query_id, condition) -> {wall (repeat 중앙값), prompt/out tokens}."""
    if evc is None:
        return {}
    by = defaultdict(list)
    for r in read_jsonl(evc / "measurements.jsonl"):
        if not r.get("warmup") and not r.get("error"):
            by[r["unit_id"]].append(r)
    out = {}
    for rs in by.values():
        rec = {
            "wall": float(np.median([r["wall"] for r in rs])),
            "prompt_tokens": rs[0].get("prompt_tokens"),
            "out_tokens": rs[0].get("out_tokens"),
        }
        for c in rs[0]["conditions"]:
            out[(rs[0]["query_id"], c)] = rec
    return out


def fit_noload(seq_calls: dict, evc: dict) -> dict | None:
    """부하 없는 호출 (seq + evc 측정) -> wall ~ a + b*pt + c*ot.

    정확히 같은 호출이 없을 때만 사용."""
    pts = [(c["prompt_tokens"], c["out_tokens"], c["wall"]) for cs in seq_calls.values()
           for c in cs if not c.get("error") and c.get("prompt_tokens") is not None
           and c.get("out_tokens") is not None]  # fmt: skip
    pts += [
        (m["prompt_tokens"], m["out_tokens"], m["wall"])
        for m in evc.values()
        if m.get("prompt_tokens") is not None and m.get("out_tokens") is not None
    ]
    if len(pts) < 5:
        return None
    a = np.array(pts, float)
    X = np.column_stack([np.ones(len(a)), a[:, 0], a[:, 1]])
    coef, *_ = np.linalg.lstsq(X, a[:, 2], rcond=None)
    pred = X @ coef
    r2 = 1 - ((a[:, 2] - pred) ** 2).sum() / max(((a[:, 2] - a[:, 2].mean()) ** 2).sum(), 1e-12)
    return {"coef": coef.tolist(), "r2": float(r2), "n": len(a), "pt_max": float(a[:, 0].max())}


# ================================================================ A
def accuracy_report(runs: dict, seq_rows: dict, seq_raw: list, evf: dict) -> dict:
    items = evf["items"]
    common = {q for q, it in items.items() if not it.get("problems")}
    excluded = {q: "; ".join(it["problems"]) for q, it in items.items() if it.get("problems")}
    seq_ids = set(seq_rows)
    reasons = Counter()
    for why in excluded.values():
        for p in why.split("; "):
            reasons[re.sub(r"\d+", "N", p)] += 1
    seq_info = {
        "n_seq_rows": len(seq_raw), "n_seq_ok": len(seq_rows),
        "n_seq_errors": sum(1 for r in seq_raw if r.get("error")),
        "seq_dup_qids": len(seq_raw) - len({r["query_id"] for r in seq_raw}),
        "n_evf_items": len(items), "n_evf_common": len(common), "n_evf_excluded": len(excluded),
        "evf_exclusion_reasons": dict(reasons.most_common()),
        "evf_items_not_in_seq": len(set(items) - seq_ids),
    }  # fmt: skip
    # seq (= crag_rule sequential) 정확도·action 분포를 부분집합별로
    subsets_seq = {"seq_all": seq_ids, "evf_common": common & seq_ids,
                   "evf_excluded": set(excluded) & seq_ids}  # fmt: skip
    seq_tab = []
    for name, qs in subsets_seq.items():
        rs = [seq_rows[q] for q in qs]
        seq_tab.append({
            "subset": name, "n": len(rs), "seq_acc": mean([r.get("acc") for r in rs]),
            **{f"share_{a}": mean([r.get("corrective_action") == a for r in rs])
               for a in efa.ACTIONS},
            **{f"acc_{a}": mean([r.get("acc") for r in rs if r.get("corrective_action") == a])
               for a in efa.ACTIONS},
        })  # fmt: skip
    for c in (BASE, "no_correct_filter", "keep_all"):
        g = evf["gens"].get(c) or {}
        if g:
            seq_tab.append({"subset": f"evf_{c}@evf_common", "n": len(g),
                            "seq_acc": mean([x.get("acc") for x in g.values()])})  # fmt: skip

    per_run, trans, by_action, by_web = [], [], [], []
    qsets = {}
    for (rho, seed, pol, k), run in runs.items():
        rows = list(run["rows"].values())
        qids = [r["query_id"] for r in rows]
        qsets[(rho, seed, pol)] = frozenset(qids)
        matched = [r for r in rows if r["query_id"] in seq_rows]
        agree = [r.get("corrective_action") == seq_rows[r["query_id"]].get("corrective_action")
                 for r in matched]  # fmt: skip
        ans_same = [
            r.get("answer") == seq_rows[r["query_id"]].get("answer")
            for r in matched
            if r.get("corrective_action") == seq_rows[r["query_id"]].get("corrective_action")
        ]
        evf_g = evf["gens"].get(pol) or {}
        row = {
            "rho": rho, "seed": seed, "policy": pol, "K": k, "n_ok": len(rows),
            "n_errors": run["n_errors"], "n_unique_qids": len(set(qids)),
            "dup_qids": len(qids) - len(set(qids)), "n_in_seq": len(matched),
            "n_not_in_seq": len(rows) - len(matched),
            "n_evf_common": sum(q in common for q in qids),
            "n_evf_excluded": sum(q in excluded for q in qids),
            "action_agree_seq": mean(agree), "action_agree_denominator": len(agree),
            "answer_same_seq_if_same_action": mean(ans_same) if pol == BASE else None,
            "correction_rate": mean([r.get("corrective_action") != "correct" for r in rows]),
            "acc_all": mean([r.get("acc") for r in rows]),
        }  # fmt: skip
        for name, qs in (("evf_common", common), ("evf_excluded", set(excluded)),
                         ("not_in_seq", set(qids) - seq_ids)):  # fmt: skip
            sub = [r for r in rows if r["query_id"] in qs]
            row[f"n_{name}_rows"] = len(sub)
            row[f"acc_live_{name}"] = mean([r.get("acc") for r in sub])
            row[f"acc_seq_{name}"] = mean([seq_rows[r["query_id"]].get("acc") for r in sub
                                           if r["query_id"] in seq_rows])  # fmt: skip
        row["acc_evf_same_policy_common"] = mean([evf_g[r["query_id"]].get("acc") for r in rows
                                                  if r["query_id"] in evf_g])  # fmt: skip
        st = (run.get("end") or {}).get("web_search") or {}
        row.update(web_hits=st.get("hits"), web_misses=st.get("misses"),
                   web_empty=st.get("empty_returned"), web_rewrite_mismatch=st.get(
                       "rewrite_mismatch"))  # fmt: skip
        per_run.append(row)
        tc = Counter((seq_rows[r["query_id"]].get("corrective_action"), r.get("corrective_action"))
                     for r in matched)  # fmt: skip
        trans += [{"rho": rho, "seed": seed, "policy": pol, "seq_action": a, "live_action": b,
                   "n": n} for (a, b), n in sorted(tc.items(), key=str)]  # fmt: skip
        for a in efa.ACTIONS:
            for name, qs in (("all", None), ("evf_common", common),
                             ("evf_excluded", set(excluded))):  # fmt: skip
                sub = [r for r in rows if r.get("corrective_action") == a
                       and (qs is None or r["query_id"] in qs)]  # fmt: skip
                if sub:
                    by_action.append({"rho": rho, "seed": seed, "policy": pol, "action": a,
                                      "subset": name, "n": len(sub),
                                      "acc": mean([r.get("acc") for r in sub])})  # fmt: skip
        wc = defaultdict(list)
        for r in rows:
            w = r.get("web_replay") or {}
            src = w.get("source") or ("none" if r.get("corrective_action") == "correct" else "?")
            wc[(src, r["query_id"] in common)].append(r.get("acc"))
        by_web += [{"rho": rho, "seed": seed, "policy": pol, "web_source": s,
                    "evf_common": c, "n": len(v), "acc": mean(v)}
                   for (s, c), v in sorted(wc.items(), key=str)]  # fmt: skip
    same_q = {}
    for (_, seed, _), qs in qsets.items():
        same_q.setdefault(seed, set()).add(qs)
    seed_check = {
        "same_qset_within_seed": {s: len(v) == 1 for s, v in same_q.items()},
        "same_qset_all_runs": len(set(qsets.values())) == 1,
    }
    return {"seq": seq_info, "seq_subsets": seq_tab, "per_run": per_run, "transitions": trans,
            "by_action": by_action, "by_web": by_web, "seed_check": seed_check}  # fmt: skip


# ================================================================ B
def call_noload(c: dict, row: dict, pol: str, seq_calls: dict, seq_rows: dict, evc: dict,
                model: dict | None) -> tuple[float | None, str]:  # fmt: skip
    """호출 하나의 부하 없는 실행시간과 출처."""
    q = row["query_id"]
    s_row = seq_rows.get(q)
    same_path = s_row is not None and s_row.get("corrective_action") == row.get("corrective_action")
    kind = c["kind"]
    if kind in GRADE_KINDS or (kind == "rewrite" and same_path):
        for sc in seq_calls.get(q, []):
            if sc["kind"] == kind and sc["idx"] == c["idx"] and not sc.get("error"):
                return float(sc["wall"]), "관측"
    if kind == "generate" and same_path:
        m = evc.get((q, pol))
        if m is not None:
            return m["wall"], "측정"
        if pol == BASE:
            for sc in seq_calls.get(q, []):
                if sc["kind"] == "generate" and not sc.get("error"):
                    return float(sc["wall"]), "관측"
    if model and c.get("prompt_tokens") is not None and c.get("out_tokens") is not None:
        a, b, d = model["coef"]
        return a + b * c["prompt_tokens"] + d * c["out_tokens"], "추정"
    return None, "없음"


def decompose(run: dict, pol: str, seq_calls: dict, seq_rows: dict, evc: dict,
              model: dict | None) -> list[dict]:  # fmt: skip
    out = []
    for key, r in run["rows"].items():
        cs = run["calls"].get(key, [])
        st = r.get("stage_latencies") or {}
        d = {"request_key": key, "query_id": r["query_id"], "arrival": r["arrival"],
             "finish": r["finish"], "latency": r["latency"], "action": r.get("corrective_action"),
             "corrected": corrected(r), "acc": r.get("acc"), "n_calls": len(cs),
             "start_lag": r.get("start_lag") or 0.0, "retrieval": st.get("retrieval", 0.0),
             "web_search": st.get("web_search", 0.0)}  # fmt: skip
        u_llm = union_len([(c["t_enqueue"], c["t_end"]) for c in cs])
        u_exec = union_len([(c["t_start"], c["t_end"]) for c in cs])
        d["llm_span"] = u_llm
        d["exec_union"] = u_exec
        d["wait_only"] = u_llm - u_exec
        d["other"] = r["latency"] - d["start_lag"] - d["retrieval"] - d["web_search"] - u_llm
        d["parallel_overlap"] = sum(c["wall"] for c in cs) - u_exec
        srcs = Counter()
        noload_tot = 0.0
        for k in KINDS:
            ks = [c for c in cs if kgroup(c["kind"]) == k]
            d[f"gate_wait_{k}"] = sum(c["gate_wait"] for c in ks)
            d[f"exec_{k}"] = sum(c["wall"] for c in ks)
            d[f"ttft_{k}"] = sum(c.get("ttft") or 0.0 for c in ks)
            nl = 0.0
            for c in ks:
                v, s = call_noload(c, r, pol, seq_calls, seq_rows, evc, model)
                srcs[s] += 1
                nl += v or 0.0
            d[f"noload_{k}"] = nl
            noload_tot += nl
        d["gen_prompt_tokens"] = sum(c.get("prompt_tokens") or 0 for c in cs
                                     if c["kind"] == "generate")  # fmt: skip
        d["gate_wait"] = sum(d[f"gate_wait_{k}"] for k in KINDS)
        d["exec"] = sum(d[f"exec_{k}"] for k in KINDS)
        d["noload"] = noload_tot
        d["inflation"] = d["exec"] - noload_tot
        d["noload_src"] = ",".join(f"{s}:{n}" for s, n in sorted(srcs.items()))
        d["noload_exact_share"] = (srcs["관측"] + srcs["측정"]) / max(sum(srcs.values()), 1)
        out.append(d)
    if out:
        cut = np.percentile([x["latency"] for x in out], 95)
        for x in out:
            x["tail5"] = bool(x["latency"] >= cut)
    return out


COMP = ("start_lag", "retrieval", "web_search", "wait_only", "exec_union", "other")


def decomp_summary(key, rows: list[dict]) -> list[dict]:
    out = []
    for sub in ("all", "tail5", "corrected", "not_corrected"):
        rs = [r for r in rows if sub == "all" or (sub == "tail5" and r["tail5"])
              or (sub == "corrected" and r["corrected"])
              or (sub == "not_corrected" and not r["corrected"])]  # fmt: skip
        if not rs:
            continue
        lat = mean([r["latency"] for r in rs])
        row = {"rho": key[0], "seed": key[1], "policy": key[2], "subset": sub, "n": len(rs),
               "latency": lat}  # fmt: skip
        for c in (*COMP, "gate_wait", "exec", "noload", "inflation", "parallel_overlap",
                  *[f"gate_wait_{k}" for k in KINDS], *[f"exec_{k}" for k in KINDS],
                  *[f"noload_{k}" for k in KINDS], "noload_exact_share"):  # fmt: skip
            row[c] = mean([r[c] for r in rs])
        row["share_wait_only"] = row["wait_only"] / lat if lat else None
        row["share_exec"] = row["exec_union"] / lat if lat else None
        row["sum_check"] = sum(row[c] for c in COMP) - lat  # 0 이어야 함 (other 정의상)
        row["neg_other_share"] = mean([r["other"] < -1e-3 for r in rs])
        out.append(row)
    return out


def paired_decomp(bd: dict, rho: str, seed: int, pol: str) -> dict | None:
    a = {r["request_key"]: r for r in bd.get((rho, seed, BASE), [])}
    b = {r["request_key"]: r for r in bd.get((rho, seed, pol), [])}
    ks = sorted(set(a) & set(b))
    if pol == BASE or not ks:
        return None
    d = lambda c: mean([b[k][c] - a[k][c] for k in ks])  # noqa: E731
    out = {"rho": rho, "seed": seed, "policy": pol, "n_pairs": len(ks), "d_latency": d("latency")}
    for c in (
        "wait_only",
        "exec_union",
        "gate_wait",
        "exec",
        "noload",
        "inflation",
        "start_lag",
        "retrieval",
        "web_search",
        "other",
        "noload_generate",
        "exec_generate",
    ):
        out[f"d_{c}"] = d(c)
    out["d_wait_share_of_d_latency"] = (out["d_wait_only"] / out["d_latency"]
                                        if out["d_latency"] else None)  # fmt: skip
    return out


# ================================================================ C
def gpu_stats(path: Path) -> dict | None:
    if not path.exists():
        return None
    util, mem = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3:
            continue
        try:
            util.append(float(p[2]))
            mem.append(float(p[3]) if len(p) > 3 else np.nan)
        except ValueError:
            continue
    if not util:
        return None
    u = np.array(util)
    return {"gpu_util_mean": float(u.mean()), "gpu_util_p50": float(np.median(u)),
            "gpu_util_ge90_share": float((u >= 90).mean()), "gpu_util_zero_share":
            float((u == 0).mean()), "gpu_mem_max_mib": float(np.nanmax(mem)) if mem else None,
            "gpu_samples": len(u)}  # fmt: skip


def queueing(key, run: dict, dec: list[dict], k: int, n_bins: int, d: Path) -> tuple[dict, list]:
    rows = list(run["rows"].values())
    arr = np.array(sorted(r["arrival"] for r in rows))
    fin = np.array([r["finish"] for r in rows])
    t_lo, t_hi = float(arr.min()), float(fin.max())
    span_arr = float(arr.max() - arr.min())
    calls = [c for cs in run["calls"].values() for c in cs]
    segs = step_segments({
        "system": [(r["arrival"], r["finish"]) for r in rows],
        "running": [(c["t_start"], c["t_end"]) for c in calls],
        "waiting": [(c["t_enqueue"], c["t_start"]) for c in calls],
        "gen_running": [(c["t_start"], c["t_end"]) for c in calls if c["kind"] == "generate"],
    }, t_lo, t_hi)  # fmt: skip
    lam = (len(arr) - 1) / span_arr if span_arr > 0 else None
    thr = len(rows) / (t_hi - t_lo) if t_hi > t_lo else None
    lat = np.array([r["latency"] for r in rows])
    wl = (run["meta"].get("workload") or {}).get("params") or {}
    noload = np.array([x["noload"] for x in dec])
    # 지연 추세: 도착 시각에 대한 기울기 (s/s). 양수로 크면 대기열이 run 동안 계속 증가
    a_t = np.array([r["arrival"] for r in rows])
    slope = float(np.polyfit(a_t, lat, 1)[0]) if len(rows) > 2 else None
    q = np.quantile(a_t, [0.25, 0.75])
    em = read_jsonl(d / "engine_metrics.jsonl") if (d / "engine_metrics.jsonl").exists() else []
    em_t = [e for e in em if e.get("t") is not None]

    def em_stat(f, k_):
        v = [e[k_] for e in em_t if e.get(k_) is not None]
        return f(v) if v else None

    end = run.get("end") or {}
    cd = ((end.get("engine_metrics_sampled") or {}).get("counter_delta")) or {}
    qt = (cd.get("queue_time_sum") / cd["queue_time_count"]) if cd.get("queue_time_count") else None
    n_end_arrival = sum(1 for r in rows if r["arrival"] <= arr.max() < r["finish"])
    out = {
        "rho": key[0], "seed": key[1], "policy": key[2], "K": k, "n": len(rows),
        "lambda_nominal": wl.get("rate"), "lambda_obs": lam, "throughput": thr,
        "thr_over_lambda": thr / lam if lam and thr else None,
        "run_span_s": t_hi - t_lo, "arrival_span_s": span_arr,
        "drain_after_last_arrival_s": t_hi - float(arr.max()),
        "N_mean": tw_mean(segs, lambda c: c["system"]),
        "little_lambdaW": (lam or 0) * float(lat.mean()),
        "N_at_last_arrival": n_end_arrival,
        "lat_slope_per_s": slope,
        "lat_mean_first_quartile": float(lat[a_t <= q[0]].mean()),
        "lat_mean_last_quartile": float(lat[a_t >= q[1]].mean()),
        "running_mean": tw_mean(segs, lambda c: c["running"]),
        "gate_waiting_mean": tw_mean(segs, lambda c: c["waiting"]),
        "slots_full_share": tw_mean(segs, lambda c: float(c["running"] >= k)),
        "idle_slot_while_waiting_share": tw_mean(segs, lambda c: float(c["waiting"] > 0
                                                                     and c["running"] < k)),
        "idle_slot_while_waiting_s": (t_hi - t_lo) * (tw_mean(segs, lambda c: float(
            c["waiting"] > 0 and c["running"] < k)) or 0.0),
        "gen_slot_share": tw_mean(segs, lambda c: c["gen_running"] / k),
        "noload_per_q_mean": float(noload.mean()) if len(noload) else None,
        "offered_work_rate": (lam or 0) * float(noload.mean()) if len(noload) else None,
        "achieved_work_rate": float(noload.sum()) / (t_hi - t_lo) if t_hi > t_lo else None,
        "vllm_running_mean": em_stat(np.mean, "running"),
        "vllm_waiting_mean": em_stat(np.mean, "waiting"),
        "vllm_waiting_max": em_stat(max, "waiting"),
        "vllm_waiting_gt0_share": em_stat(lambda v: float(np.mean([x > 0 for x in v])), "waiting"),
        "kv_usage_mean": em_stat(np.mean, "kv_usage"), "kv_usage_max": em_stat(max, "kv_usage"),
        "vllm_preemptions": cd.get("preemptions"), "vllm_queue_time_mean": qt,
        "engine_metric_samples": len(em_t),
        **(gpu_stats(Path(str(d) + ".gpu.csv")) or {}),
    }  # fmt: skip
    # 시간 구간별 (run 진행률)
    edges = np.linspace(t_lo, t_hi, n_bins + 1)
    bins = []
    for i in range(n_bins):
        b0, b1 = edges[i], edges[i + 1]
        sg = [(max(s0, b0), min(s1, b1), c) for s0, s1, c in segs if s1 > b0 and s0 < b1]
        arr_in = [r for r in rows if b0 <= r["arrival"] < b1]
        e_in = [e for e in em_t if b0 <= e["t"] < b1]
        bins.append({
            "rho": key[0], "seed": key[1], "policy": key[2], "bin": i, "t0": float(b0),
            "t1": float(b1), "arrivals": len(arr_in),
            "completions": int(((fin >= b0) & (fin < b1)).sum()),
            "N_mean": tw_mean(sg, lambda c: c["system"]),
            "gate_waiting_mean": tw_mean(sg, lambda c: c["waiting"]),
            "running_mean": tw_mean(sg, lambda c: c["running"]),
            "lat_mean_of_arrivals": mean([r["latency"] for r in arr_in]),
            "vllm_waiting_mean": mean([e.get("waiting") for e in e_in]),
            "kv_usage_mean": mean([e.get("kv_usage") for e in e_in]),
        })  # fmt: skip
    return out, bins


# ================================================================ D
def scheduling(key, run: dict, dec: list[dict], model) -> tuple[dict, list[dict]]:
    calls = [c for cs in run["calls"].values() for c in cs]
    run_iv = [(c["t_start"], c["t_end"], c["kind"]) for c in calls]
    hol = []
    for c in calls:
        if kgroup(c["kind"]) != "grade" or c["gate_wait"] <= 1e-3:
            continue
        w0, w1 = c["t_enqueue"], c["t_start"]
        occ_gen = sum(max(0.0, min(e, w1) - max(s, w0)) for s, e, k in run_iv if k == "generate")
        occ_all = sum(max(0.0, min(e, w1) - max(s, w0)) for s, e, _ in run_iv)
        hol.append(occ_gen / occ_all if occ_all > 0 else 0.0)
    out = {"rho": key[0], "seed": key[1], "policy": key[2]}
    for k in KINDS:
        ws = [c["gate_wait"] for c in calls if kgroup(c["kind"]) == k]
        out[f"gate_wait_{k}_mean"] = mean(ws)
        out[f"gate_wait_{k}_p95"] = pct(ws, 95)
        out[f"exec_{k}_mean"] = mean([c["wall"] for c in calls if kgroup(c["kind"]) == k])
    out["grade_wait_gen_occupancy"] = mean(hol)
    nl = np.array([x["noload"] for x in dec])
    out["noload_mean"] = float(nl.mean()) if len(nl) else None
    out["noload_cv"] = float(nl.std() / nl.mean()) if len(nl) and nl.mean() > 0 else None
    out["noload_p95_over_p50"] = (float(np.percentile(nl, 95) / np.percentile(nl, 50))
                                  if len(nl) else None)  # fmt: skip
    out["spearman_noload_latency"] = spearman([x["noload"] for x in dec],
                                              [x["latency"] for x in dec])  # fmt: skip
    est_rows = []
    if model is not None:
        by_key = {x["request_key"]: x for x in dec}
        for rk_, r in run["rows"].items():
            cs = run["calls"].get(rk_, [])
            g0 = next((c for c in cs if c["kind"] == "grade" and c["idx"] == 0), None)
            gen = next((c for c in cs if c["kind"] == "generate"), None)
            if g0 is None or rk_ not in by_key:
                continue
            feats, n_docs = qfeats(r)
            m = model.for_query(r["query_id"])
            est0 = m.est_remaining("grade", 0, g0["prompt_chars"], n_docs, feats)
            est_gen = m.call_est("generate", gen["prompt_chars"]) if gen else None
            x = by_key[rk_]
            est_rows.append({"rho": key[0], "seed": key[1], "policy": key[2],
                             "query_id": r["query_id"], "action": r.get("corrective_action"),
                             "est_first_grade": est0, "noload_total": x["noload"],
                             "est_gen_call": est_gen, "noload_gen": x["noload_generate"],
                             "gen_prompt_tokens": x["gen_prompt_tokens"]})  # fmt: skip
        e0 = [(e["est_first_grade"], e["noload_total"]) for e in est_rows]
        eg = [(e["est_gen_call"], e["noload_gen"]) for e in est_rows if e["est_gen_call"]]
        out["srpt_est_first_bias"] = mean([a - b for a, b in e0])
        out["srpt_est_first_mae"] = mean([abs(a - b) for a, b in e0])
        out["srpt_est_first_spearman"] = spearman([a for a, _ in e0], [b for _, b in e0])
        out["srpt_est_gen_bias"] = mean([a - b for a, b in eg])
        out["srpt_est_gen_spearman"] = spearman([a for a, _ in eg], [b for _, b in eg])
    return out, est_rows


# ================================================ SV1 대조 (같은 workload 의 fcfs_query)
def sv_consistency(sv_root: Path | None, runs: dict, k: int) -> list[dict]:
    if sv_root is None:
        return []
    out = []
    for (rho, seed, pol, kk), run in runs.items():
        if pol != BASE:
            continue
        d = sv_root / f"rho{rho}_s{seed}_fcfs_query_K{kk}"
        if not (d / "queries.jsonl").exists():
            continue
        sv = load_run(d)
        ks = sorted(set(sv["rows"]) & set(run["rows"]))
        a = [run["rows"][x]["latency"] for x in ks]
        b = [sv["rows"][x]["latency"] for x in ks]
        out.append({
            "rho": rho, "seed": seed, "n_pairs": len(ks),
            "same_workload": (sv["meta"].get("workload") or {}).get("sha256")
            == (run["meta"].get("workload") or {}).get("sha256"),
            "evc_crag_rule_lat_mean": mean(a), "sv1_fcfs_query_lat_mean": mean(b),
            "evc_crag_rule_lat_p95": pct(a, 95), "sv1_fcfs_query_lat_p95": pct(b, 95),
            "evc_acc": mean([run["rows"][x].get("acc") for x in ks]),
            "sv1_acc": mean([sv["rows"][x].get("acc") for x in ks]),
            "action_agree": mean([run["rows"][x].get("corrective_action")
                                  == sv["rows"][x].get("corrective_action") for x in ks]),
        })  # fmt: skip
    return out


# ================================================================ report
def analyze(a) -> dict:
    root = Path(a.live_root)
    paths = discover(root)
    if not paths:
        raise SystemExit(f"live run 폴더 없음: {root}")
    runs, warns = {}, []
    for key, d in paths.items():
        r = load_run(d)
        r["end"] = read_json(d / "run_end.json") if (d / "run_end.json").exists() else {}
        cp = _cfg_policy(d)
        if cp != key[2]:
            warns.append(f"{d.name}: config evidence_policy={cp!r} != 폴더 이름 {key[2]!r}")
        if r["policy"] != a.sched_policy:
            warns.append(f"{d.name}: scheduler={r['policy']} (기대 {a.sched_policy})")
        runs[key] = r
    seq_rows, seq_calls, seq_raw = load_seq_rows(Path(a.seq))
    evf = load_evf(Path(a.evf) if a.evf else None)
    evc = load_evc(Path(a.evc_seq) if a.evc_seq else None)
    model = fit_noload(seq_calls, evc)
    cm, cm_id, w = cost_model_of({paths[k].name: r for k, r in runs.items()}, a.cost_model)
    warns += w
    wl = defaultdict(set)
    for (rho, seed, _, _), r in runs.items():
        wl[(rho, seed)].add((r["meta"].get("workload") or {}).get("sha256"))
    warns += [f"rho{k[0]} s{k[1]}: 정책 간 workload hash 다름" for k, v in wl.items() if len(v) > 1]

    acc = accuracy_report(runs, seq_rows, seq_raw, evf)
    bd, dsum, pairs, qrows, bins, sched, est = {}, [], [], [], [], [], []
    for key, run in runs.items():
        rho, seed, pol, k = key
        dec = decompose(run, pol, seq_calls, seq_rows, evc, model)
        bd[(rho, seed, pol)] = dec
        dsum += decomp_summary((rho, seed, pol), dec)
        q, b = queueing((rho, seed, pol), run, dec, k, a.bins, paths[key])
        qrows.append(q)
        bins += b
        s, e = scheduling((rho, seed, pol), run, dec, cm)
        sched.append(s)
        est += e
    for rho, seed, pol, _ in runs:
        p = paired_decomp(bd, rho, seed, pol)
        if p:
            pairs.append(p)
    # 정책별 offered work 비 (crag_rule 대비) -> 유효 부하 (가정: no-load 초 단위 처리 용량이 같음)
    base_nl = {(q["rho"], q["seed"]): q["noload_per_q_mean"] for q in qrows if q["policy"] == BASE}
    for q in qrows:
        b = base_nl.get((q["rho"], q["seed"]))
        q["work_ratio_vs_crag_rule"] = q["noload_per_q_mean"] / b if b else None
        q["rho_eff_est"] = float(q["rho"]) * q["work_ratio_vs_crag_rule"] if b else None
    calib = read_json(Path(a.calib)) if a.calib else None
    return {
        "live_root": str(root), "seq": a.seq, "evf": a.evf, "evc_seq": a.evc_seq,
        "warnings": warns, "noload_model": model, "cost_model": cm_id,
        "calibration": (calib or {}).get("recommendation") if calib else None,
        "accuracy": acc, "decomp_summary": dsum, "decomp_paired": pairs, "queueing": qrows,
        "bins": bins, "scheduling": sched, "estimates": est,
        "sv1_consistency": sv_consistency(Path(a.sv_live_root) if a.sv_live_root else None,
                                          runs, a.k),
        "request_decomp": [x for rs in bd.values() for x in rs],
        "keys": [list(k) for k in runs],
    }  # fmt: skip


def report_md(res: dict, name: str) -> str:
    tbl = efa.md_table
    A = res["accuracy"]
    L = [f"# EVC live 진단 ({name})", "", f"- live: `{res['live_root']}`", f"- seq: `{res['seq']}`",
         f"- evf: `{res['evf']}`  evc-seq: `{res['evc_seq']}`",
         f"- cost model (SRPT-est 재계산): {res['cost_model']}",
         f"- no-load [추정] 모델: {json.dumps(res['noload_model'])}", ""]  # fmt: skip
    if res["warnings"]:
        L += ["## 경고", ""] + [f"- {w}" for w in res["warnings"]] + [""]
    L += ["## A. 질문 집합·정확도", "", "### seq / evf 집합", "",
          "```", json.dumps(A["seq"], ensure_ascii=False, indent=1), "```", "",
          f"- seed 간 질문 집합: {json.dumps(A['seed_check'])}", "",
          "### seq (sequential crag_rule) 부분집합별 [관측]", ""]  # fmt: skip
    L += tbl(A["seq_subsets"], ["subset", "n", "seq_acc", "share_correct", "share_ambiguous",
                                "share_incorrect", "acc_correct", "acc_ambiguous",
                                "acc_incorrect"])  # fmt: skip
    L += ["", "### live run 별 (같은 질문 부분집합에서 live / seq / evf 정확도)", ""]
    L += tbl(
        A["per_run"],
        [
            "rho",
            "seed",
            "policy",
            "n_ok",
            "n_errors",
            "dup_qids",
            "n_in_seq",
            "n_evf_common",
            "n_evf_excluded",
            "acc_all",
            "acc_live_evf_common",
            "acc_seq_evf_common",
            "acc_evf_same_policy_common",
            "acc_live_evf_excluded",
            "acc_seq_evf_excluded",
            "correction_rate",
            "action_agree_seq",
            "action_agree_denominator",
            "answer_same_seq_if_same_action",
            "web_misses",
            "web_empty",
        ],
    )
    L += ["", "- acc_seq_*: 같은 질문의 sequential crag_rule 정확도 (정책과 무관한 기준).",
          "  acc_evf_same_policy_common: evf 의 같은 정책 sequential 정확도 (evf 공통 질문만)", "",
          "### action 별 정확도 (live action 기준)", ""]  # fmt: skip
    L += tbl(
        [r for r in A["by_action"] if r["seed"] == 0],
        ["rho", "seed", "policy", "action", "subset", "n", "acc"],
    )
    L += ["", "### web replay 출처별 정확도 (seed 0)", ""]
    L += tbl(
        [r for r in A["by_web"] if r["seed"] == 0],
        ["rho", "policy", "web_source", "evf_common", "n", "acc"],
    )
    L += ["", "## B. 지연 분해 (요청 평균, 초) [관측] / noload [관측·측정·추정]", ""]
    L += tbl([r for r in res["decomp_summary"] if r["subset"] in ("all", "tail5")],
             ["rho", "seed", "policy", "subset", "n", "latency", "start_lag", "retrieval",
              "web_search", "wait_only", "exec_union", "other", "share_wait_only", "gate_wait",
              "exec", "noload", "inflation", "gate_wait_generate", "exec_generate",
              "noload_generate", "parallel_overlap", "noload_exact_share", "sum_check",
              "neg_other_share"])  # fmt: skip
    L += ["", "- latency = start_lag + retrieval + web_search + wait_only + exec_union + other "
          "(정의상 sum_check=0). gate_wait / exec 는 호출 합 (병렬 grading 이면 중첩 포함)",
          "- noload: 같은 호출의 부하 없는 실행시간. inflation = exec - noload "
          "(batching·GPU 공유·vLLM 내부 대기. 요청별로 내부 대기만 분리할 수 없음)", "",
          "### crag_rule 대비 같은 요청 차이", ""]  # fmt: skip
    L += tbl(res["decomp_paired"], ["rho", "seed", "policy", "n_pairs", "d_latency",
                                    "d_wait_only", "d_exec_union", "d_noload", "d_inflation",
                                    "d_noload_generate", "d_exec_generate", "d_other",
                                    "d_wait_share_of_d_latency"])  # fmt: skip
    L += ["", "## C. Queueing / 부하", ""]
    L += tbl(res["queueing"], ["rho", "seed", "policy", "lambda_nominal", "lambda_obs",
                               "throughput", "thr_over_lambda", "drain_after_last_arrival_s",
                               "N_mean", "little_lambdaW", "N_at_last_arrival", "lat_slope_per_s",
                               "lat_mean_first_quartile", "lat_mean_last_quartile",
                               "noload_per_q_mean", "work_ratio_vs_crag_rule", "rho_eff_est",
                               "offered_work_rate", "achieved_work_rate"])  # fmt: skip
    L += ["", "- little_lambdaW = λ_obs x 평균 지연. N_mean (관측 구간 시간 평균) 보다 크게 크면 "
          "도착 구간 이후 drain 이 길다 = 관측 구간에서 정상상태가 아님 (처리량 < 도착률)",
          "- offered_work_rate = λ x E[no-load LLM 시간] (초/초). rho_eff_est = rho x "
          "work_ratio [추정: no-load 초 단위 처리 용량이 정책 간 같다는 가정]", ""]  # fmt: skip
    L += tbl(
        res["queueing"],
        [
            "rho",
            "seed",
            "policy",
            "running_mean",
            "slots_full_share",
            "gate_waiting_mean",
            "idle_slot_while_waiting_share",
            "idle_slot_while_waiting_s",
            "gen_slot_share",
            "vllm_running_mean",
            "vllm_waiting_mean",
            "vllm_waiting_max",
            "vllm_waiting_gt0_share",
            "kv_usage_mean",
            "kv_usage_max",
            "vllm_preemptions",
            "vllm_queue_time_mean",
            "gpu_util_mean",
            "gpu_util_ge90_share",
            "gpu_util_zero_share",
        ],
    )
    L += ["", "- slots_full_share: gate slot (K) 이 모두 찬 시간 비율. "
          "idle_slot_while_waiting_share > 0 이면 대기 호출이 있는데 slot 이 빈 시간 "
          "(scheduler 손실)",
          "- GPU util 은 nvidia-smi 의 kernel 활동 시간 비율 (SM 포화도 아님). "
          "run 전체 (warmup 포함)",
          "", "## D. Scheduling", ""]  # fmt: skip
    L += tbl(res["scheduling"], ["rho", "seed", "policy", "gate_wait_grade_mean",
                                 "gate_wait_grade_p95", "gate_wait_rewrite_mean",
                                 "gate_wait_generate_mean", "exec_grade_mean", "exec_generate_mean",
                                 "grade_wait_gen_occupancy", "noload_mean", "noload_cv",
                                 "noload_p95_over_p50", "spearman_noload_latency",
                                 "srpt_est_first_bias", "srpt_est_first_mae",
                                 "srpt_est_first_spearman", "srpt_est_gen_bias",
                                 "srpt_est_gen_spearman"])  # fmt: skip
    L += ["", "- grade_wait_gen_occupancy: grade 호출이 gate 에서 기다린 동안 실행 중 slot-시간 중 "
          "generate 의 비율 (긴 generation 이 짧은 grading 을 막는 정도)",
          "- srpt_est_*: run_meta cost model 로 첫 grade 호출 시점 남은 작업 추정 "
          "(정책 코드와 같은 식)"
          " - 요청의 no-load 작업량. bias < 0 = 과소 추정", ""]  # fmt: skip
    if res["sv1_consistency"]:
        L += ["## SV1 fcfs_query 와 대조 (같은 workload, 같은 scheduler, crag_rule)", ""]
        L += tbl(res["sv1_consistency"], ["rho", "seed", "n_pairs", "same_workload",
                                          "evc_crag_rule_lat_mean", "sv1_fcfs_query_lat_mean",
                                          "evc_crag_rule_lat_p95", "sv1_fcfs_query_lat_p95",
                                          "evc_acc", "sv1_acc", "action_agree"])  # fmt: skip
        L += [""]
    L += ["## 로그 한계", "",
          "- vLLM 내부 대기 (scheduler waiting) 는 요청별로 기록되지 않음: engine_metrics 집계값만",
          "- prefill / decode 분리는 TTFT (streaming) 기준. TTFT 에는 vLLM 내부 대기가 포함됨",
          "- GPU util 시계열은 run 시각과 정렬하지 않음 "
          "(nvidia-smi timestamp 와 t0 대응 기록 없음)",
          "- noload 는 prefix cache 초기화 (evc-seq reset) / seq 기록 기준. "
          "live 는 prefix cache 켜짐",
          ""]  # fmt: skip
    return "\n".join(L)


def new_out_dir(root: Path, name: str) -> Path:
    """results/crag_system/analysis/evc_live/<name>/<ts> (이미 있으면 _2, _3 ... 덮어쓰지 않음)."""
    from datetime import datetime

    base = root / "evc_live" / name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out, n = base, 1
    while out.exists():
        n += 1
        out = base.with_name(f"{base.name}_{n}")
    out.mkdir(parents=True, exist_ok=False)
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--evf", default=None)
    ap.add_argument("--evc-seq", default=None)
    ap.add_argument("--sv-live-root", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--cost-model", default=None, help="run_meta 의 cost model 대신")
    ap.add_argument("--sched-policy", default="fcfs_query")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--name", default="EVC-live-diag")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    res = analyze(a)
    out = new_out_dir(Path(a.out_root), a.name)
    (out / "analysis.json").write_text(json.dumps(
        {k: v for k, v in res.items() if k not in ("request_decomp", "estimates", "bins")},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")  # fmt: skip
    A = res["accuracy"]
    for fname, rows in (("acc_per_run", A["per_run"]), ("acc_seq_subsets", A["seq_subsets"]),
                        ("action_transitions", A["transitions"]), ("acc_by_action", A["by_action"]),
                        ("acc_by_web", A["by_web"]), ("decomp_summary", res["decomp_summary"]),
                        ("decomp_paired", res["decomp_paired"]), ("queueing", res["queueing"]),
                        ("timeline_bins", res["bins"]), ("scheduling", res["scheduling"]),
                        ("srpt_estimates", res["estimates"]),
                        ("request_decomp", res["request_decomp"]),
                        ("sv1_consistency", res["sv1_consistency"])):  # fmt: skip
        write_csv(out / f"{fname}.csv", rows)
    md = report_md(res, a.name)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not a.quiet:
        print(md)
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
