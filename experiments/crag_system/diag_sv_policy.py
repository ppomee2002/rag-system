"""SV live run 상세 진단 (읽기 전용, 추가 추론·API 호출 없음).

SV1 결과 (FCFS-query 는 FCFS 대비 개선, SRPT-est 는 FCFS-query / SRPT-nopath 보다
일관되게 낫지 않음) 의 원인을 기존 로그만으로 진단한다.
정책 동작은 바꾸지 않고, 정책 클래스를 다시 실행하지도 않는다.
우선순위 key 는 run_meta.json 에 저장된 cost model 과 기록된 호출 특징으로 오프라인 재계산한다
(srpt_est / srpt_nopath / sjf_pred run 에서는 기록된 dispatch key 와 일치하는지 검사).

    uv run python experiments/crag_system/diag_sv_policy.py \
        --live-root results/crag_system/pre/PRE-SV1-LIVE \
        --seq results/crag_system/pre/PRE-SV1-A/eval_seq \
        --k 4 --rhos 0.8 0.95 --seeds 0 1 --name SV1-policy-diag

run 폴더 이름은 run_sched_validation.sh 와 같다: <live-root>/rho<rho>_s<seed>_<policy>_K<k>

분석 항목 (조건 = rho x seed 마다, warmup·오류 요청 제외)
  1 재정렬       dispatch 시점 대기열 (enqueue = t - wait 로 복원) 에서
                 call 순서 / 요청 도착 순서를 바꾼 dispatch 비율 (diag_sched.dispatch_metrics)
  2 선택 차이    각 dispatch 의 같은 후보 집합에 여러 규칙을 적용했을 때 무엇을 골랐을지
                 (반사실 선택). 같은 대기열에서의 즉시 선택만 비교한다.
                 다른 선택 이후의 전개는 시뮬레이션하지 않는다.
                 oracle = 부하 없는 sequential run (--seq) 의 같은 query 실제 남은 호출
                 wall 합이 최소인 호출. regret = 고른 호출의 실제 남은 작업 - 후보 중 최소.
                 concordance = 동시 대기 후보 쌍의 순위 일치율
  3 추정 오차    srpt 계열 추정치 (기록된 key) vs 실제 남은 작업 (sequential wall 합,
                 같은 경로인 요청만). 호출 종류 x 경로별. 참고: live wall 합·완료까지 남은 시간
  4 오차 큰 요청 첫 grade 호출 시점 추정치 vs 실제 전체 작업량 오차 상위 10% 의 특징
  5 경로별 지연  run 별 corrected / not_corrected latency (해당 run 의 경로 기준)
  6 지연 분해    요청별 latency = start_lag + retrieval + gate 대기 (호출 종류별)
                 + LLM wall (종류별) + web_search + 기타.
                 지연 상위 5% 요청의 구성, 도착이 늦은 요청에게 추월당한 횟수
  7 악화 원인    같은 요청끼리 (target - base) 의 구성 요소별 차이 (기본: srpt_est - fcfs_query,
                 srpt_est - srpt_nopath). 전체 / target 의 지연 상위 5% / 경로별
  8 미사용 정보  grade 점수 (scores_full) 로 경로가 몇 번째 grade 호출에서 확정되는지
                 (정책은 쓰지 않음)

로그로 알 수 없는 항목은 추정하지 않고 report.md 의 "로그 한계" 에 적는다.
결과: results/crag_system/analysis/<name>/<timestamp>/ (덮어쓰지 않음)
  report.md, analysis.json, choices.csv, estimates.csv, high_error_requests.csv,
  request_breakdown.csv, decomposition.csv, latency_by_path.csv, grade_decision.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diag_sched import dispatch_metrics, mean, pct, read_json, read_jsonl, spearman  # noqa: E402

from rag.scheduling.cost_model import cost_model_from_dict, features  # noqa: E402

GRADE_KINDS = ("grade", "grade_batch")
POLICIES = ("none", "fcfs", "fcfs_query", "sjf_pred", "srpt_nopath", "srpt_est")
RULES = ("fcfs", "fcfs_query", "sjf_pred", "srpt_nopath", "srpt_est")
EST_POLICIES = ("sjf_pred", "srpt_nopath", "srpt_est")
DEFAULT_COMPARE = ("fcfs_query:srpt_est", "srpt_nopath:srpt_est")
COMPONENTS = (
    "start_lag", "retrieval", "gate_wait_grade", "gate_wait_rewrite", "gate_wait_generate",
    "wall_grade", "wall_rewrite", "wall_generate", "web_search", "other",
)  # fmt: skip
EPS = 1e-9


# ---------------------------------------------------------------- loading
def rk(row: dict) -> str:
    return f"{row['request_index']}:{row['query_id']}"


def corrected(row: dict) -> bool:
    qs = row.get("query_state") or {}
    if qs.get("corrected") is not None:
        return bool(qs["corrected"])
    return row.get("corrective_action") not in (None, "correct")


def load_run(d: str | Path) -> dict:
    d = Path(d)
    rows_all = [r for r in read_jsonl(d / "queries.jsonl") if not r.get("warmup")]
    rows = {rk(r): r for r in rows_all if not r.get("error")}
    calls = defaultdict(list)
    for c in read_jsonl(d / "calls.jsonl"):
        if not c.get("warmup") and c.get("request_index", -1) >= 0:
            calls[f"{c['request_index']}:{c['query_id']}"].append(c)
    for cs in calls.values():
        cs.sort(key=lambda c: c["t_start"])
    meta = read_json(d / "run_meta.json")
    sched = meta.get("scheduler") or {}
    return {
        "dir": str(d),
        "rows": rows,
        "n_errors": len(rows_all) - len(rows),
        "calls": calls,
        "log": read_jsonl(d / "scheduler_log.jsonl"),
        "meta": meta,
        "policy": ((sched.get("policy") or {}).get("name")) if sched else "none",
        "K": sched.get("slots") if sched else None,
    }


def load_seq(d: str | Path | None) -> dict:
    """sequential run -> query_id 별 {action, calls [(kind, idx, wall)], ...}."""
    if not d:
        return {}
    run = load_run(d)
    out = {}
    for key, r in run["rows"].items():
        cs = run["calls"].get(key, [])
        out[r["query_id"]] = {
            "corrected": corrected(r),
            "calls": [(c["kind"], c["idx"], float(c["wall"])) for c in cs],
            "out_tokens_gen": sum(c.get("out_tokens") or 0 for c in cs if c["kind"] == "generate"),
        }
    return out


def cost_model_of(runs: dict[str, dict], override: str | None):
    if override:
        d = json.loads(Path(override).read_text(encoding="utf-8"))
        return cost_model_from_dict(d), "override:" + override, []
    dicts = {}
    for label, run in runs.items():
        cm = run["meta"].get("cost_model")
        if cm:
            dicts[label] = json.dumps(cm, sort_keys=True)
    if not dicts:
        return None, None, ["cost model 이 run_meta 에 없음 -> 반사실 key / 추정 오차 분석 생략"]
    warns = []
    if len(set(dicts.values())) > 1:
        warns.append(f"run 간 cost model 이 다름: {sorted(dicts)} (첫 번째 것을 사용)")
    label = next(iter(dicts))
    sha = hashlib.sha256(dicts[label].encode()).hexdigest()[:12]
    return cost_model_from_dict(json.loads(dicts[label])), f"{label}:{sha}", warns


def qfeats(row: dict):
    scores = [s if s is not None else 0.0 for s in (row.get("extra") or {}).get(
        "retrieval_scores") or []]  # fmt: skip
    return features(scores, row.get("retrieval_chars") or [], row.get("s_pop")), len(scores)


# ---------------------------------------------------------------- tickets
def true_remaining(seq: dict, qid: str, kind: str, idx: int, live_corr: bool) -> float | None:
    """부하 없는 sequential run 의 같은 query 에서 (kind, idx) 호출 포함 이후 wall 합.
    경로가 다르거나 같은 호출이 없으면 None (추정하지 않음)."""
    s = seq.get(qid)
    if s is None or s["corrected"] != live_corr:
        return None
    for i, (k, j, _) in enumerate(s["calls"]):
        if k == kind and j == idx:
            return float(sum(w for _, _, w in s["calls"][i:]))
    return None


def build_tickets(run: dict, model, seq: dict) -> dict[int, dict]:
    """scheduler_log (dispatch 순서) -> seq 별 ticket. 측정 요청만."""
    tickets = {}
    for e in run["log"]:
        key = e.get("request_key")
        row = run["rows"].get(key)
        if row is None:
            continue
        call = next(
            (
                c
                for c in run["calls"].get(key, [])
                if c["kind"] == e["kind"] and c["idx"] == e["idx"]
            ),
            None,
        )
        corr = corrected(row)
        t = {
            "seq": e["seq"],
            "rk": key,
            "qid": row["query_id"],
            "arrival": row["arrival"],
            "kind": e["kind"],
            "idx": e["idx"],
            "enq": e["t"] - e["wait"],
            "grant": e["t"],
            "wait": e["wait"],
            "logged_key": e.get("key"),
            "corrected": corr,
            "chars": call["prompt_chars"] if call else None,
            "true_rem": true_remaining(seq, row["query_id"], e["kind"], e["idx"], corr),
        }
        if call is not None:
            later = [c for c in run["calls"][key] if c["t_start"] >= call["t_start"]]
            t["live_wall_rem"] = float(sum(c["wall"] for c in later))
        t["time_to_finish"] = row["finish"] - t["enq"]
        t["est"] = rule_estimates(t, row, model)
        tickets[e["seq"]] = t
    return tickets


def rule_estimates(t: dict, row: dict, model) -> dict:
    """같은 ticket 에 각 정책 key 의 primary 값 (작을수록 먼저). 정책 코드와 같은 식."""
    out = {"fcfs": float(t["seq"]), "fcfs_query": float(t["arrival"])}
    if model is None or t["chars"] is None:
        return out
    feats, n_docs = qfeats(row)
    m = model.for_query(t["qid"])
    srpt = m.est_remaining(t["kind"], t["idx"], t["chars"], n_docs, feats)
    out["srpt_est"] = srpt
    out["srpt_nopath"] = m.pred_post(feats) if t["kind"] == "rewrite" else srpt
    out["sjf_pred"] = m.pred_total(feats)
    return out


def key_check(run: dict, tickets: dict) -> dict:
    """정책 key 재계산이 기록과 같은지 (같아야 반사실 비교가 정책 코드와 일치)."""
    pol = run["policy"]
    diffs = [
        abs(t["logged_key"][1] - t["est"][pol])
        for t in tickets.values()
        if t["logged_key"] and pol in t["est"]
    ]
    if not diffs:
        return {"policy": pol, "checked": 0}
    return {
        "policy": pol,
        "checked": len(diffs),
        "max_abs_diff": float(max(diffs)),
        "n_mismatch_gt_1e-6": int(sum(d > 1e-6 for d in diffs)),
    }


# ---------------------------------------------------------------- dispatch replay
def replay(run: dict, tickets: dict):
    """dispatch 순서대로 (dispatch 된 ticket, 그 시각 다른 대기 ticket 목록) 을 낸다."""
    order = [e["seq"] for e in run["log"] if e["seq"] in tickets]
    by_enq = sorted(tickets, key=lambda s: tickets[s]["enq"])
    pending: set[int] = set()
    p = 0
    for s in order:
        now = tickets[s]["grant"]
        while p < len(by_enq) and tickets[by_enq[p]]["enq"] <= now + EPS:
            pending.add(by_enq[p])
            p += 1
        pending.discard(s)
        yield s, [tickets[o] for o in pending]


def _pick(cands: list[dict], rule: str):
    if rule == "oracle":
        if any(c["true_rem"] is None for c in cands):
            return None
        return min(cands, key=lambda c: (c["true_rem"], c["seq"]))
    if any(rule not in c["est"] for c in cands):
        return None
    return min(cands, key=lambda c: (c["est"][rule], c["seq"]))


def choice_metrics(run: dict, tickets: dict, overtaken: Counter) -> dict:
    n = n_choice = 0
    agree = Counter()
    avail = Counter()
    regret = defaultdict(list)
    conc = Counter()
    conc_n = Counter()
    differ = Counter()  # srpt_est vs fcfs_query 선택이 다른 경우의 (srpt pick kind, fq pick kind)
    differ_oracle = Counter()
    for s, others in replay(run, tickets):
        t = tickets[s]
        n += 1
        for o in others:  # 늦게 도착한 요청에게 추월당함
            if o["rk"] != t["rk"] and o["arrival"] < t["arrival"]:
                overtaken[o["rk"]] += 1
        if not others:
            continue
        n_choice += 1
        cands = [t, *others]
        picks = {r: _pick(cands, r) for r in (*RULES, "oracle")}
        trues = [c["true_rem"] for c in cands]
        best = min(trues) if all(x is not None for x in trues) else None
        for r, pk in picks.items():
            if pk is None:
                continue
            avail[r] += 1
            agree[r] += pk["seq"] == s
            if best is not None:
                regret[r].append(pk["true_rem"] - best)
        if best is not None:
            regret["actual"].append(t["true_rem"] - best)
        ps, pf = picks.get("srpt_est"), picks.get("fcfs_query")
        if ps is not None and pf is not None and ps["seq"] != pf["seq"]:
            differ[f"{ps['kind']}>{pf['kind']}"] += 1
            if ps["true_rem"] is not None and pf["true_rem"] is not None:
                differ_oracle["srpt_shorter" if ps["true_rem"] < pf["true_rem"] else
                              "fq_shorter_or_equal"] += 1  # fmt: skip
        # 동시 대기 후보 쌍의 순위 일치율 (실제 남은 작업 기준)
        known = [c for c in cands if c["true_rem"] is not None]
        for r in ("sjf_pred", "srpt_nopath", "srpt_est", "fcfs_query"):
            if not known or r not in known[0]["est"]:
                continue
            for i in range(len(known)):
                for j in range(i + 1, len(known)):
                    a, b = known[i], known[j]
                    dt = a["true_rem"] - b["true_rem"]
                    de = a["est"][r] - b["est"][r]
                    if abs(dt) < EPS or abs(de) < EPS:
                        continue
                    conc_n[r] += 1
                    conc[r] += (dt > 0) == (de > 0)
    out = {
        "n_dispatch": n,
        "n_with_choice": n_choice,
        "agree_with_actual": {r: agree[r] / avail[r] for r in avail if avail[r]},
        "n_rule_available": dict(avail),
        "regret_mean_s": {r: mean(v) for r, v in regret.items()},
        "pair_concordance": {r: conc[r] / conc_n[r] for r in conc_n if conc_n[r]},
        "pair_n": dict(conc_n),
        "srpt_vs_fq_differ": sum(differ.values()),
        "srpt_vs_fq_differ_rate": sum(differ.values()) / n_choice if n_choice else None,
        "srpt_vs_fq_differ_kinds": dict(differ.most_common()),
        "srpt_vs_fq_differ_oracle": dict(differ_oracle),
    }
    return out


# ---------------------------------------------------------------- estimates
def _kind_group(t: dict) -> str:
    if t["kind"] in GRADE_KINDS:
        return "grade_idx0" if t["idx"] == 0 else "grade_idx1+"
    return t["kind"]


def estimate_rows(cond: str, label: str, run: dict, tickets: dict) -> list[dict]:
    pol = run["policy"]
    if pol not in EST_POLICIES:
        return []
    groups = defaultdict(list)
    for t in tickets.values():
        if t["logged_key"] is None:
            continue
        path = "corrected" if t["corrected"] else "not_corrected"
        for g in (_kind_group(t), "all"):
            groups[(g, path)].append(t)
            groups[(g, "all")].append(t)
    out = []
    for (g, path), ts in sorted(groups.items()):
        est = [t["logged_key"][1] for t in ts if t["true_rem"] is not None]
        tru = [t["true_rem"] for t in ts if t["true_rem"] is not None]
        err = np.asarray(est) - np.asarray(tru) if est else np.array([])
        out.append(
            {
                "cond": cond, "run": label, "policy": pol, "kind": g, "path": path,
                "n": len(ts), "n_with_truth": len(tru),
                "est_mean": mean(est), "true_mean": mean(tru),
                "err_mean": mean(err), "abs_err_mean": mean(np.abs(err)),
                "abs_err_p90": pct(np.abs(err), 90),
                "rel_err_median": pct(np.abs(err) / np.maximum(tru, 1e-6), 50) if tru else None,
                "spearman": spearman(est, tru),
                "live_wall_rem_mean": mean([t.get("live_wall_rem") for t in ts
                                            if t.get("live_wall_rem") is not None]),
                "time_to_finish_mean": mean([t["time_to_finish"] for t in ts]),
            }
        )  # fmt: skip
    return out


def high_error_rows(cond: str, run: dict, tickets: dict, seq: dict, upper: float) -> list[dict]:
    """첫 grade 호출의 추정치 vs 실제 전체 작업량 (sequential). |오차| 상위 10% 표시."""
    out = []
    for t in tickets.values():
        if t["kind"] not in GRADE_KINDS or t["idx"] != 0 or t["true_rem"] is None:
            continue
        if t["logged_key"] is None:
            continue
        row = run["rows"][t["rk"]]
        s = seq[t["qid"]]
        ex = row.get("extra") or {}
        sc = [x for x in (ex.get("retrieval_scores") or []) if x is not None]
        gs = (ex.get("grading") or {}).get("scores_full") or []
        cs = run["calls"].get(t["rk"], [])
        walls = defaultdict(float)
        for k, _, w in s["calls"]:
            walls[k] += w
        out.append(
            {
                "cond": cond, "request_key": t["rk"], "query_id": t["qid"],
                "action": row.get("corrective_action"), "corrected": t["corrected"],
                "est0": t["logged_key"][1], "true_total_seq": t["true_rem"],
                "err": t["logged_key"][1] - t["true_rem"],
                "seq_wall_grade": walls["grade"] + walls["grade_batch"],
                "seq_wall_rewrite": walls["rewrite"], "seq_wall_generate": walls["generate"],
                "seq_gen_out_tokens": s["out_tokens_gen"],
                "gen_prompt_chars": sum(c["prompt_chars"] for c in cs if c["kind"] == "generate"),
                "retrieval_chars": sum(row.get("retrieval_chars") or []),
                "top1_score": max(sc) if sc else None,
                "mean_score": mean(sc),
                "s_pop": row.get("s_pop"),
                "n_web_docs": len(ex.get("web_doc_ids") or []),
                "first_grade_ge_upper": decided_at(gs, upper),
                "latency": row["latency"],
            }
        )  # fmt: skip
    if out:
        cut = np.percentile([abs(r["err"]) for r in out], 90)
        for r in out:
            r["top10_abs_err"] = abs(r["err"]) >= cut
    return out


def decided_at(scores: list, upper: float) -> int | None:
    """correct 경로가 확정되는 grade 순번 (score >= upper 인 첫 문서). 없으면 None."""
    for i, s in enumerate(scores):
        if s is not None and s >= upper:
            return i
    return None


# ---------------------------------------------------------------- breakdown
def breakdown(cond: str, label: str, run: dict, overtaken: Counter) -> list[dict]:
    out = []
    for key, r in run["rows"].items():
        cs = run["calls"].get(key, [])
        st = r.get("stage_latencies") or {}
        comp = {
            "start_lag": r.get("start_lag") or 0.0,
            "retrieval": st.get("retrieval", 0.0),
            "web_search": st.get("web_search", 0.0),
        }
        for kind in ("grade", "rewrite", "generate"):
            ks = (kind,) if kind != "grade" else GRADE_KINDS
            comp[f"gate_wait_{kind}"] = sum(c["gate_wait"] for c in cs if c["kind"] in ks)
            comp[f"wall_{kind}"] = sum(c["wall"] for c in cs if c["kind"] in ks)
        comp["other"] = r["latency"] - sum(comp.values())
        out.append(
            {
                "cond": cond, "run": label, "policy": run["policy"], "request_key": key,
                "query_id": r["query_id"], "arrival": r["arrival"], "latency": r["latency"],
                "corrected": corrected(r), "action": r.get("corrective_action"),
                "n_calls": len(cs), "overtaken_by_later": overtaken.get(key, 0),
                **{c: comp[c] for c in COMPONENTS},
            }
        )  # fmt: skip
    if out:
        cut = np.percentile([x["latency"] for x in out], 95)
        for x in out:
            x["tail5"] = x["latency"] >= cut
    return out


def tail_summary(rows: list[dict]) -> list[dict]:
    out = []
    by = defaultdict(list)
    for r in rows:
        by[(r["cond"], r["run"])].append(r)
    for (cond, label), rs in by.items():
        for name, sub in (("all", rs), ("tail5", [r for r in rs if r["tail5"]])):
            if not sub:
                continue
            out.append(
                {
                    "cond": cond, "run": label, "subset": name, "n": len(sub),
                    "corrected_share": mean([r["corrected"] for r in sub]),
                    "latency_mean": mean([r["latency"] for r in sub]),
                    "overtaken_mean": mean([r["overtaken_by_later"] for r in sub]),
                    **{c: mean([r[c] for r in sub]) for c in COMPONENTS},
                }
            )  # fmt: skip
    return out


def decomposition(cond: str, base: str, target: str, bd: dict[str, list[dict]]) -> list[dict]:
    """같은 요청 (request_key) 끼리 target - base 의 구성 요소별 평균 차이."""
    a = {r["request_key"]: r for r in bd[base]}
    b = {r["request_key"]: r for r in bd[target]}
    keys = sorted(set(a) & set(b))
    subsets = {
        "all": keys,
        "target_tail5": [k for k in keys if b[k]["tail5"]],
        "base_tail5": [k for k in keys if a[k]["tail5"]],
        "corrected_both": [k for k in keys if a[k]["corrected"] and b[k]["corrected"]],
        "not_corrected_both": [k for k in keys if not a[k]["corrected"] and not b[k]["corrected"]],
        "path_mismatch": [k for k in keys if a[k]["corrected"] != b[k]["corrected"]],
    }
    out = []
    for name, ks in subsets.items():
        if not ks:
            continue
        d = {c: mean([b[k][c] - a[k][c] for k in ks]) for c in COMPONENTS}
        out.append(
            {
                "cond": cond, "base": base, "target": target, "subset": name, "n": len(ks),
                "d_latency": mean([b[k]["latency"] - a[k]["latency"] for k in ks]),
                "d_overtaken": mean([b[k]["overtaken_by_later"] - a[k]["overtaken_by_later"]
                                     for k in ks]),
                **{f"d_{c}": v for c, v in d.items()},
            }
        )  # fmt: skip
    return out


def latency_by_path(cond: str, label: str, run: dict) -> list[dict]:
    rows = list(run["rows"].values())
    out = []
    for path, sub in (
        ("all", rows),
        ("corrected", [r for r in rows if corrected(r)]),
        ("not_corrected", [r for r in rows if not corrected(r)]),
    ):
        lat = [r["latency"] for r in sub]
        out.append(
            {
                "cond": cond, "run": label, "policy": run["policy"], "path": path,
                "n": len(sub), "mean": mean(lat), "p50": pct(lat, 50), "p95": pct(lat, 95),
                "gate_wait_mean": mean([(r.get("query_state") or {}).get("cum_wait", 0.0)
                                        for r in sub]),
            }
        )  # fmt: skip
    return out


def grade_decision_rows(cond: str, seq_dir_rows: dict, upper: float) -> list[dict]:
    """sequential run 기준: correct 경로가 몇 번째 grade 에서 확정되는지 분포."""
    cnt = Counter()
    n_corr = n_not = 0
    for r in seq_dir_rows.values():
        if corrected(r):
            n_corr += 1
            continue
        n_not += 1
        gs = ((r.get("extra") or {}).get("grading") or {}).get("scores_full") or []
        cnt[decided_at(gs, upper)] += 1
    out = []
    cum = 0
    for i in sorted(k for k in cnt if k is not None):
        cum += cnt[i]
        out.append(
            {
                "cond": cond, "decided_at_grade": i, "n_correct": cnt[i],
                "share_of_correct": cnt[i] / n_not if n_not else None,
                "cum_share_of_correct": cum / n_not if n_not else None,
                "n_corrected_path": n_corr,
            }
        )  # fmt: skip
    if cnt.get(None):
        out.append({"cond": cond, "decided_at_grade": "none(>=upper 없음)", "n_correct": cnt[None]})
    return out


# ---------------------------------------------------------------- driver
def discover(live_root: str, rhos, seeds, k: int, policies) -> tuple[dict, list[str]]:
    conds, missing = {}, []
    for rho in rhos:
        for s in seeds:
            cond = f"rho{rho}_s{s}"
            runs = {}
            for pol in policies:
                d = Path(live_root) / f"rho{rho}_s{s}_{pol}_K{k}"
                if (d / "queries.jsonl").exists():
                    runs[pol] = str(d)
                else:
                    missing.append(str(d))
            if runs:
                conds[cond] = runs
    return conds, missing


def comparability(runs: dict[str, dict]) -> list[str]:
    warns = []
    shas = {lb: (r["meta"].get("workload") or {}).get("sha256") for lb, r in runs.items()}
    if len(set(shas.values())) > 1:
        warns.append(f"workload hash 불일치: {shas}")
    keysets = {lb: frozenset(r["rows"]) for lb, r in runs.items()}
    if len(set(keysets.values())) > 1:
        warns.append(
            "측정 요청 집합이 다름 (오류 요청 등): "
            + ", ".join(f"{lb}={len(v)}" for lb, v in keysets.items())
        )
    for lb, r in runs.items():
        if r["n_errors"]:
            warns.append(f"{lb}: 오류 요청 {r['n_errors']}건 (분석에서 제외)")
    return warns


def analyze_condition(cond: str, paths: dict[str, str], seq: dict, seq_rows: dict, a) -> dict:
    runs = {lb: load_run(d) for lb, d in paths.items()}
    warns = comparability(runs)
    model, model_id, w = cost_model_of(runs, a.cost_model)
    warns += w
    upper = None
    for r in runs.values():
        th = ((r["meta"].get("comparability") or {}).get("thresholds") or {}).get("upper")
        if th is not None:
            upper = th
            break
    upper = a.upper if upper is None else upper
    res = {"cond": cond, "runs": paths, "cost_model": model_id, "upper": upper}
    res.update(dispatch={}, choices={}, key_check={}, path_vs_seq={})
    est_rows, he_rows, bd_rows, lp_rows = [], [], [], []
    bd = {}
    for lb, run in runs.items():
        measured = set(run["rows"])
        arrivals = {k: r["arrival"] for k, r in run["rows"].items()}
        if seq:
            same = [
                seq[r["query_id"]]["corrected"] == corrected(r)
                for r in run["rows"].values()
                if r["query_id"] in seq
            ]
            res["path_vs_seq"][lb] = {"n": len(same), "agree": mean(same)}
        overtaken: Counter = Counter()
        if run["log"]:
            res["dispatch"][lb] = dispatch_metrics(run["log"], measured, arrivals)
            tickets = build_tickets(run, model, seq)
            res["key_check"][lb] = key_check(run, tickets)
            res["choices"][lb] = choice_metrics(run, tickets, overtaken)
            est_rows += estimate_rows(cond, lb, run, tickets)
            if run["policy"] == "srpt_est" and seq:
                he_rows += high_error_rows(cond, run, tickets, seq, upper)
        bd[lb] = breakdown(cond, lb, run, overtaken)
        bd_rows += bd[lb]
        lp_rows += latency_by_path(cond, lb, run)
    dec = []
    for pair in a.compare:
        base, target = pair.split(":", 1)
        if base in bd and target in bd:
            dec += decomposition(cond, base, target, bd)
        else:
            warns.append(f"비교 생략 (run 없음): {pair}")
    for lb, kc in res["key_check"].items():
        if kc.get("n_mismatch_gt_1e-6"):
            warns.append(
                f"{lb}: key 재계산 불일치 {kc['n_mismatch_gt_1e-6']}건 "
                f"(max {kc['max_abs_diff']:.3g}) -> 반사실 선택은 근사"
            )
    for lb, d in res["dispatch"].items():
        if d.get("queue_reconstruction_mismatch"):
            warns.append(f"{lb}: 대기열 복원 불일치 {d['queue_reconstruction_mismatch']}건")
    res["warnings"] = warns
    res["tables"] = {
        "estimates": est_rows,
        "high_error": he_rows,
        "breakdown": bd_rows,
        "tail": tail_summary(bd_rows),
        "decomposition": dec,
        "latency_by_path": lp_rows,
        "grade_decision": grade_decision_rows(cond, seq_rows, upper) if seq_rows else [],
    }
    return res


# ---------------------------------------------------------------- report
def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "-" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return lines


def high_error_summary(rows: list[dict]) -> list[dict]:
    cols = ("corrected", "est0", "true_total_seq", "err", "seq_wall_rewrite", "seq_wall_generate",
            "seq_gen_out_tokens", "gen_prompt_chars", "retrieval_chars", "top1_score",
            "n_web_docs", "latency")  # fmt: skip
    out = []
    for name, sub in (
        ("top10_abs_err", [r for r in rows if r.get("top10_abs_err")]),
        ("rest", [r for r in rows if not r.get("top10_abs_err")]),
    ):
        if sub:
            out.append(
                {"group": name, "n": len(sub),
                 **{c: mean([float(r[c]) for r in sub if r[c] is not None]) for c in cols}}
            )  # fmt: skip
    return out


LIMITS = [
    "요청별 vLLM 내부 대기 시간: /metrics 집계값만 있음."
    " calls.wall 은 vLLM 대기 + batching 간섭 포함",
    "dispatch 시점의 대기열 스냅샷은 기록하지 않음: scheduler_log 의 enqueue (= t - wait) 로 복원"
    " (복원 길이와 n_waiting 의 일치를 검사)",
    "반사실 선택은 같은 대기열에서의 즉시 선택만 비교. 다른 선택을 했을 때의 이후 전개"
    " (대기열 변화, batching 구성 변화) 는 로그로 알 수 없음"
    " -> 시뮬레이션 없이 지연 개선량으로 해석하지 않는다",
    "실제 남은 작업량 (oracle) 은 부하 없는 sequential run 의 같은 query wall."
    " live 와 경로가 다른 요청은 None 으로 두고 제외 (추정하지 않음)."
    " 같은 경로여도 출력 길이가 다를 수 있음",
    "조건당 run 1개: 같은 정책·workload 반복 run 이 없어 run 간 변동 (vLLM 비결정성) 은 분리 불가",
    "grading 도중의 문서별 점수가 언제 나왔는지는 calls 로 알 수 있지만,"
    " 정책 입력 (QueryState) 에는 grading 종료 후의 결과만 들어감 (정책 동작 확인용 정보)",
]


def report_md(results: list[dict], missing: list[str], name: str) -> str:
    L = [f"# SV 정책 상세 진단 ({name})", ""]
    if missing:
        L += ["## 없는 run", ""] + [f"- {m}" for m in missing] + [""]
    L += ["## 1. 재정렬 / 2. 선택 차이", ""]
    rows = []
    for res in results:
        for lb, d in res["dispatch"].items():
            c = res["choices"].get(lb, {})
            ag = c.get("agree_with_actual", {})
            rg = c.get("regret_mean_s", {})
            pc = c.get("pair_concordance", {})
            rows.append(
                {
                    "cond": res["cond"], "run": lb, "n_dispatch": d.get("n_dispatch"),
                    "choice_ge2": d.get("choice_ge2"), "queue_mean": d.get("queue_len_mean"),
                    "reorder_call": d.get("reorder_vs_call_rate"),
                    "reorder_arrival": d.get("reorder_vs_arrival_rate"),
                    "agree_fq": ag.get("fcfs_query"), "agree_srpt": ag.get("srpt_est"),
                    "agree_nopath": ag.get("srpt_nopath"), "agree_oracle": ag.get("oracle"),
                    "srpt_ne_fq": c.get("srpt_vs_fq_differ_rate"),
                    "regret_actual": rg.get("actual"), "regret_fq": rg.get("fcfs_query"),
                    "regret_srpt": rg.get("srpt_est"), "regret_oracle": rg.get("oracle"),
                    "conc_srpt": pc.get("srpt_est"), "conc_nopath": pc.get("srpt_nopath"),
                    "conc_sjf": pc.get("sjf_pred"),
                }
            )  # fmt: skip
    L += md_table(rows, list(rows[0]) if rows else ["cond"])
    L += [
        "",
        "- agree_X: 실제 dispatch 가 규칙 X 의 선택과 같은 비율 (대기열 2 이상인 dispatch 중)."
        " 자기 정책 열은 1 이어야 함",
        "- srpt_ne_fq: 같은 후보에서 srpt_est 와 fcfs_query 의 선택이 다른 비율"
        " (정책이 차이를 낼 수 있는 상한)",
        "- regret_X: 규칙 X 가 고른 호출의 실제 남은 작업 (sequential wall) - 후보 최소 (초)."
        " actual = 실제 선택",
        "- conc_X: 동시에 대기한 후보 쌍에서 추정 순서와 실제 남은 작업 순서가 같은 비율"
        " (0.5 = 무작위)",
        "",
    ]
    for res in results:
        for lb, c in res["choices"].items():
            if c.get("srpt_vs_fq_differ"):
                L.append(
                    f"- {res['cond']} {lb}: srpt_est≠fcfs_query 선택 {c['srpt_vs_fq_differ']}건, "
                    f"(srpt 선택>fq 선택 종류) {json.dumps(c['srpt_vs_fq_differ_kinds'])}, "
                    f"oracle 판정 {json.dumps(c['srpt_vs_fq_differ_oracle'])}"
                )
    L += ["", "## 3. 추정 오차 (기록된 key vs sequential 실제 남은 작업, 초)", ""]
    est = [r for res in results for r in res["tables"]["estimates"] if r["path"] != "all"
           or r["kind"] == "all"]  # fmt: skip
    L += md_table(
        est,
        [
            "cond",
            "run",
            "kind",
            "path",
            "n",
            "n_with_truth",
            "est_mean",
            "true_mean",
            "err_mean",
            "abs_err_mean",
            "rel_err_median",
            "spearman",
        ],
    )
    L += ["", "## 4. 추정 오차 상위 10% 요청 특징 (srpt_est run, 첫 grade 호출 시점)", ""]
    for res in results:
        he = res["tables"]["high_error"]
        if he:
            L += [f"### {res['cond']}", ""]
            L += md_table(high_error_summary(he), list(high_error_summary(he)[0]))
            L += [""]
    L += ["## 5. 경로별 지연", ""]
    lp = [r for res in results for r in res["tables"]["latency_by_path"]]
    L += md_table(lp, ["cond", "run", "path", "n", "mean", "p50", "p95", "gate_wait_mean"])
    L += ["", "## 6. 지연 분해 (전체 / 지연 상위 5%)", ""]
    tl = [r for res in results for r in res["tables"]["tail"]]
    L += md_table(tl, ["cond", "run", "subset", "n", "corrected_share", "latency_mean",
                       "overtaken_mean", *COMPONENTS])  # fmt: skip
    L += [
        "",
        "- overtaken: 그 요청의 호출이 대기하는 동안 더 늦게 도착한 요청의 호출이 먼저 나간 횟수",
        "",
    ]
    L += ["## 7. 악화 원인 분해 (같은 요청, target - base, 초)", ""]
    dc = [r for res in results for r in res["tables"]["decomposition"]]
    L += md_table(dc, ["cond", "base", "target", "subset", "n", "d_latency", "d_overtaken",
                       *[f"d_{c}" for c in COMPONENTS]])  # fmt: skip
    L += ["", "- 평균 차이의 신뢰구간은 paired_bootstrap_sv.py", ""]
    L += ["## 8. 정책이 쓰지 않는 정보: correct 경로가 확정되는 grade 순번 (sequential run)", ""]
    gd = results[0]["tables"]["grade_decision"] if results else []
    L += (
        md_table(
            gd,
            [
                "decided_at_grade",
                "n_correct",
                "share_of_correct",
                "cum_share_of_correct",
                "n_corrected_path",
            ],
        )
        if gd
        else ["- --seq 없음"]
    )
    L += ["", "## 조건 확인", ""]
    for res in results:
        L += [f"- {res['cond']}: key 재계산 {json.dumps(res['key_check'])}"]
        L += [f"- {res['cond']}: sequential 과 경로 일치 {json.dumps(res['path_vs_seq'])}"]
        L += [f"- {res['cond']}: {w}" for w in res["warnings"]]
    L += ["", "## 로그 한계 (추정하지 않은 항목)", ""] + [f"- {x}" for x in LIMITS] + [""]
    return "\n".join(L)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-root", required=True)
    ap.add_argument("--seq", default=None, help="부하 없는 sequential eval run (oracle 작업량)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rhos", nargs="+", default=["0.8", "0.95"])
    ap.add_argument("--seeds", nargs="+", default=["0", "1"])
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--compare", nargs="+", default=list(DEFAULT_COMPARE), metavar="BASE:TARGET")
    ap.add_argument("--cost-model", default=None, help="run_meta 대신 쓸 cost model json")
    ap.add_argument("--upper", type=float, default=0.7, help="run_meta 에 threshold 가 없을 때")
    ap.add_argument("--name", default="sv_policy_diag")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    conds, missing = discover(a.live_root, a.rhos, a.seeds, a.k, a.policies)
    if not conds:
        raise SystemExit(f"run 이 없음: {a.live_root}")
    seq = load_seq(a.seq)
    seq_rows = load_run(a.seq)["rows"] if a.seq else {}
    results = [analyze_condition(c, p, seq, seq_rows, a) for c, p in conds.items()]
    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    slim = [{k: v for k, v in r.items() if k != "tables"} for r in results]
    (out / "analysis.json").write_text(
        json.dumps(
            {"created": datetime.now().isoformat(), "args": vars(a), "missing": missing,
             "conditions": slim},
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )  # fmt: skip
    choices = [
        {"cond": r["cond"], "run": lb, **{k: json.dumps(v) if isinstance(v, dict) else v
                                          for k, v in c.items()}}
        for r in results
        for lb, c in r["choices"].items()
    ]  # fmt: skip
    write_csv(out / "choices.csv", choices)
    for fname, key in (
        ("estimates.csv", "estimates"),
        ("high_error_requests.csv", "high_error"),
        ("request_breakdown.csv", "breakdown"),
        ("decomposition.csv", "decomposition"),
        ("latency_by_path.csv", "latency_by_path"),
        ("grade_decision.csv", "grade_decision"),
    ):
        rows = [x for r in results for x in r["tables"][key]]
        if key == "grade_decision":
            rows = results[0]["tables"][key]
        write_csv(out / fname, rows)
    md = report_md(results, missing, a.name)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not a.quiet:
        print(md)
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
