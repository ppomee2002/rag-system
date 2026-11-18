"""상태 기반 SRPT-est (M2) GPU scheduling: 사전 예상 (expect) 과 live 최종 비교 (report).
읽기 전용 (추론·API 호출 없음). 결과는 새 폴더에만 쓴다.

expect (Part A, live 전에 실행 — 사전 예상과 검증 목표를 고정)
    uv run python experiments/crag_system/state_sched_analysis.py expect \
        --base-root results/crag_system/pre/PRE-EQL1-LIVE \
        --train results/crag_system/pre/PRE-SV1-TRAIN/train \
        --seq results/crag_system/pre/PRE-SV1-A/eval_seq \
        --cost-model results/crag_system/cost_models/SV1_train_wall.json \
        --state-model results/crag_system/cost_models/SV1_train_state_m2.json \
        [--evc-seq <PRE-EVC-SV1-SEQ/ts>] [--speedup 1:1,2:1.7,4:2.6] --k 4 --name SST1-expect

  1) 결정 차이 [관측 로그 기반]: 기존 fcfs_query live 의 gate 대기열을 복원해, 같은 시각 함께
     대기한 호출 사이에서 fcfs_query / M0 / M2 가 고를 호출, 무부하 잔여 작업 [관측·측정만] 과의
     순위 일치율, regret. 실제 dispatch 는 fcfs_query 결과이며 순서를 바꾼 뒤의 지연은 알 수 없다.
  2) trace replay 시뮬레이션 [시뮬레이션]: 같은 live run 의 요청 (도착 시각·경로·prompt 길이·
     retrieval / web 지연) 을 K slot gate 로 재생. 호출 서비스 시간 = 무부하 시간 x 감속률
     slow(호출 종류, 동시 실행 수). 감속률은 기준 fcfs_query live run 에서 추정 [추정, in-sample].
     --speedup 을 주면 균일 처리율 모델 g(n) 시나리오를 추가한다.
     보정 오차: 재생한 fcfs_query 를 실제 fcfs_query live 와 비교 (평균·p95·요청별 상관).
  3) 시나리오 (live 수치 환산, 모두 [추정])
     A  sim 상대 개선율을 live fcfs_query 평균에 적용
     B  live E2E 중 gate wait 만 sim 비율로 줄어든다고 가정 (실행시간 변화 무시)
  4) 검증 목표 (사전 등록): report 단계가 같은 기준으로 판정한다.
  재현할 수 없는 것: vLLM 내부 batching·prefix cache·prefill / decode 혼합 효과, 호출 순서 변화에
  따른 실행시간 변화, 같은 요청의 grading 결과 변동 (vLLM 비결정성). 시뮬레이션 수치는 GPU 개선율이
  아니며 보장된 성능이 아니다.

report (Part D, live 후)
    uv run python experiments/crag_system/state_sched_analysis.py report \
        --base-root results/crag_system/pre/PRE-EQL1-LIVE \
        --m0-root results/crag_system/pre/PRE-SST1-LIVE-srpt_est \
        --m2-root results/crag_system/pre/PRE-SST1-LIVE-srpt_state \
        --seq results/crag_system/pre/PRE-SV1-A/eval_seq [--evc-seq ...] \
        [--expect <expect 결과 analysis.json>] --k 4 --name SST1-report

  정책별 Accuracy, E2E 평균 / p50 / p95 / p99, gate wait, throughput, GPU util, 결정 변경률
  (M2 vs M0, 각 run 의 state_estimates 로그), scheduler overhead, query 단위 paired bootstrap 95% CI
  (iid / block, seed 층화), 개선율 (Base - M2) / Base x 100, 환경 차이 (기준 run 재측정 필요 여부),
  사전 예상 vs 실제.
결과: results/crag_system/analysis/state_sched/<name>/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cost_state_ablation as csa  # noqa: E402
import state_sched as ss  # noqa: E402
from diag_evc_live import (  # noqa: E402
    call_noload,
    decompose,
    discover,
    fit_noload,
    load_evc,
    load_seq_rows,
)
from diag_sched import mean, pct, spearman  # noqa: E402
from diag_sv_policy import load_run, md_table, qfeats, replay, write_csv  # noqa: E402
from equal_load import run_summary  # noqa: E402
from evidence_cost import _cfg_policy  # noqa: E402
from grade_info_ablation import grade_scores, parse_speedup  # noqa: E402
from paired_bootstrap_sv import boot, pair_arrays  # noqa: E402

from rag.scheduling.cost_model import GRADE_KINDS, cost_model_from_dict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SIM_POLICIES = ("fcfs_query", "srpt_est", "srpt_state", "oracle")
EST_OF = {"srpt_est": "M0", "srpt_state": "M2"}
PAIRS = (("fcfs_query", "srpt_state"), ("srpt_est", "srpt_state"), ("fcfs_query", "srpt_est"))
EPS = 1e-9


def kgroup(kind: str) -> str:
    return "grade" if kind in GRADE_KINDS else kind


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_jsonl(p: Path) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_ctx(a) -> dict:
    seq_rows, seq_calls, _ = load_seq_rows(Path(a.seq))
    evc = load_evc(Path(a.evc_seq)) if a.evc_seq else {}
    return {"seq_rows": seq_rows, "seq_calls": seq_calls, "evc": evc,
            "noload": fit_noload(seq_calls, evc)}  # fmt: skip


def base_runs(root: str, k: int, sched: str | None) -> tuple[dict, list[str]]:
    """root -> {seed: (run, dir, evidence policy)}. 같은 K, (선택) 같은 scheduler 만."""
    out, warns = {}, []
    for key, d in discover(Path(root)).items():
        if key[3] != k:
            continue
        run = load_run(d)
        if sched and run["policy"] != sched:
            warns.append(f"{d}: scheduler={run['policy']} (기대 {sched}) -> 제외")
            continue
        pol = _cfg_policy(d) or key[2]
        if key[1] in out:
            warns.append(f"{root}: seed {key[1]} run 이 여러 개 -> {d.name} 사용")
        out[key[1]] = (run, d, pol, key[0])
    return out, warns


# ================================================================ 관측 / 추정 (요청 단위)
def obs_seq(row: dict, pol: str, calls: list[dict]) -> list[csa.Obs]:
    """순차 grading 가정: grade i enqueue 때 관측 = grade 0..i-1 점수. 이후 호출은 모든 점수."""
    feats, n_docs = qfeats(row)
    scores = grade_scores(row)
    act = row.get("corrective_action")
    out = []
    for c in calls:
        grade = c["kind"] in GRADE_KINDS
        seen = (tuple((i, scores[i]) for i in range(min(c["idx"], len(scores)))) if grade
                else tuple(enumerate(scores)))  # fmt: skip
        out.append(csa.Obs(qid=row["query_id"], policy=pol, kind=c["kind"], idx=c["idx"],
                           chars=c["prompt_chars"], n_docs=n_docs,
                           feats=tuple(float(x) for x in feats),
                           rchars=tuple(row.get("retrieval_chars") or []),
                           rscores=tuple(csa._rscores(row)), q_len=len(row["question"]),
                           s_pop=row.get("s_pop"), seen=seen,
                           action=None if grade else act))  # fmt: skip
    return out


def estimates(model, st, o: csa.Obs) -> dict:
    m = model.for_query(o.qid)
    m0 = m.est_remaining(o.kind, o.idx, o.chars, o.n_docs, np.array(o.feats))
    return {"M0": m0, "M2": ss.m2_estimate(m, st, o)[0]}


# ================================================================ 1) 결정 차이 (live 로그)
def decision_analysis(label: str, run: dict, pol: str, model, st, ctx: dict) -> dict:
    tickets = {}
    for e in run["log"]:
        key = e.get("request_key")
        row = run["rows"].get(key)
        if row is None or row.get("question") is None:
            continue
        cs = run["calls"].get(key, [])
        call = next((c for c in cs if c["kind"] == e["kind"] and c["idx"] == e["idx"]), None)
        if call is None:
            continue
        enq = e["t"] - e["wait"]
        scores = grade_scores(row)
        grade = e["kind"] in GRADE_KINDS
        seen = (tuple(sorted((c["idx"], scores[c["idx"]]) for c in cs if c["kind"] in GRADE_KINDS
                             and c["idx"] != e["idx"] and c["t_end"] <= enq + 1e-6
                             and c["idx"] < len(scores))) if grade
                else tuple(enumerate(scores)))  # fmt: skip
        feats, n_docs = qfeats(row)
        o = csa.Obs(qid=row["query_id"], policy=pol, kind=e["kind"], idx=e["idx"],
                    chars=call["prompt_chars"], n_docs=n_docs,
                    feats=tuple(float(x) for x in feats),
                    rchars=tuple(row.get("retrieval_chars") or []),
                    rscores=tuple(csa._rscores(row)), q_len=len(row["question"]),
                    s_pop=row.get("s_pop"), seen=seen,
                    action=None if grade else row.get("corrective_action"))  # fmt: skip
        tru, exact = 0.0, True
        for c in cs:
            if c["t_start"] >= call["t_start"] - 1e-9:
                v, s = call_noload(c, row, pol, ctx["seq_calls"], ctx["seq_rows"], ctx["evc"],
                                   ctx["noload"])  # fmt: skip
                exact &= s in ("관측", "측정")
                tru += v or 0.0
        tickets[e["seq"]] = {"seq": e["seq"], "rk": key, "enq": enq, "grant": e["t"],
                             "kind": e["kind"], "est": {**estimates(model, st, o),
                                                        "fcfs_query": row["arrival"]},
                             "true_rem": tru if exact else None}  # fmt: skip
    return decisions_from_tickets(label, run, tickets, ("fcfs_query", "M0", "M2"))


def decisions_from_tickets(label: str, run: dict, tickets: dict, rules: tuple) -> dict:
    n = 0
    top = Counter()
    pair_chg, pair_n = Counter(), Counter()
    conc, conc_n, regret = Counter(), Counter(), defaultdict(list)
    by_kind = Counter()
    for s, others in replay(run, tickets):
        if not others:
            continue
        n += 1
        cands = [tickets[s], *others]
        pick = {r: min(cands, key=lambda c, r=r: (c["est"][r], c["seq"]))["seq"] for r in rules}
        if pick["M2"] != pick["M0"]:
            by_kind[tickets[pick["M2"]]["kind"]] += 1
        for r in rules:
            top[(r, "M0")] += pick[r] != pick["M0"]
            top[(r, "actual")] += pick[r] != s
        trues = [c["true_rem"] for c in cands]
        if all(x is not None for x in trues):
            best = min(trues)
            for r in rules:
                regret[r].append(tickets[pick[r]]["true_rem"] - best)
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                x, y = cands[i], cands[j]
                dm = x["est"]["M0"] - y["est"]["M0"]
                for r in rules:
                    de = x["est"][r] - y["est"][r]
                    if abs(dm) > EPS:
                        pair_n[r] += 1
                        pair_chg[r] += np.sign(de) != np.sign(dm)
                    if x["true_rem"] is None or y["true_rem"] is None:
                        continue
                    dt = x["true_rem"] - y["true_rem"]
                    if abs(dt) > EPS and abs(de) > EPS:
                        conc_n[r] += 1
                        conc[r] += (dt > 0) == (de > 0)
    rows = [{"run": label, "scheduler": run["policy"], "rule": r, "n_decisions": n,
             "top_differs_vs_M0": top[(r, "M0")] / n if n else None,
             "top_differs_vs_actual": top[(r, "actual")] / n if n else None,
             "pair_order_change_vs_M0": pair_chg[r] / pair_n[r] if pair_n[r] else None,
             "truth_pair_concordance": conc[r] / conc_n[r] if conc_n[r] else None,
             "truth_pairs": conc_n[r], "regret_mean": mean(regret[r]),
             "regret_n": len(regret[r])} for r in rules]  # fmt: skip
    exact = sum(t["true_rem"] is not None for t in tickets.values())
    return {"rows": rows, "n_tickets": len(tickets),
            "exact_truth_share": exact / len(tickets) if tickets else None,
            "m2_pick_differs_by_kind": dict(by_kind)}  # fmt: skip


# ================================================================ 2) replay 시뮬레이션
def concurrency(calls: list[dict]) -> np.ndarray:
    """각 호출 실행 구간의 시간 평균 동시 실행 수 (자신 포함)."""
    s = np.array([c["t_start"] for c in calls])
    e = np.array([c["t_end"] for c in calls])
    d = np.maximum(e - s, EPS)
    ov = np.clip(np.minimum(e[:, None], e[None, :]) - np.maximum(s[:, None], s[None, :]), 0, None)
    np.fill_diagonal(ov, 0.0)
    return 1.0 + ov.sum(axis=1) / d


def fit_slowdown(run: dict, pol: str, ctx: dict, k: int) -> dict:
    """기준 live run 의 호출: wall / 무부하 [관측·측정만] 를 (종류, 동시 실행 수) 별 중앙값으로."""
    calls, ratio = [], []
    for key, row in run["rows"].items():
        for c in run["calls"].get(key, []):
            v, s = call_noload(c, row, pol, ctx["seq_calls"], ctx["seq_rows"], ctx["evc"],
                               ctx["noload"])  # fmt: skip
            calls.append(c)
            ratio.append(c["wall"] / v if (v and s in ("관측", "측정") and v > EPS) else None)
    conc = concurrency(calls) if calls else np.array([])
    cell = defaultdict(list)
    for c, r, n in zip(calls, ratio, conc, strict=True):
        if r is not None:
            nn = int(min(max(round(n), 1), k))
            cell[(kgroup(c["kind"]), nn)].append(r)
            cell[("all", nn)].append(r)
    table = {f"{g}:{n}": {"median": float(np.median(v)), "n": len(v)}
             for (g, n), v in sorted(cell.items())}  # fmt: skip
    return {"table": table, "n_calls": len(calls),
            "n_ratio": sum(r is not None for r in ratio)}  # fmt: skip


def slow_fn(fit: dict, k: int, min_n: int = 5):
    t = fit["table"]

    def f(kind: str, n: int) -> float:
        n = int(min(max(n, 1), k))
        for g in (kgroup(kind), "all"):
            for m in [n, *sorted(range(1, k + 1), key=lambda x: abs(x - n))]:
                e = t.get(f"{g}:{m}")
                if e and e["n"] >= min_n:
                    return max(e["median"], EPS)
        return 1.0

    return f


def build_jobs(run: dict, pol: str, ctx: dict, model, st) -> tuple[list[dict], Counter]:
    jobs, drop = [], Counter()
    for key, row in run["rows"].items():
        cs = sorted(run["calls"].get(key, []), key=lambda c: c["t_start"])
        if not cs or row.get("question") is None:
            drop["no_calls"] += 1
            continue
        svc, srcs = [], Counter()
        for c in cs:
            v, s = call_noload(c, row, pol, ctx["seq_calls"], ctx["seq_rows"], ctx["evc"],
                               ctx["noload"])  # fmt: skip
            svc.append(v)
            srcs[s] += 1
        if any(v is None for v in svc):
            drop["noload_missing"] += 1
            continue
        obs = obs_seq(row, pol, cs)
        est = [estimates(model, st, o) for o in obs]
        st_ = row.get("stage_latencies") or {}
        phases = [("delay", float(st_.get("retrieval", 0.0)))]
        for j, c in enumerate(cs):
            phases.append(("call", j))
            if c["kind"] == "rewrite":
                phases.append(("delay", float(st_.get("web_search", 0.0))))
        keys = []
        for j in range(len(cs)):
            keys.append({"fcfs_query": row["arrival"], "srpt_est": est[j]["M0"],
                         "srpt_state": est[j]["M2"], "oracle": float(sum(svc[j:]))})  # fmt: skip
        jobs.append({"key": key, "qid": row["query_id"], "arrival": float(row["arrival"]),
                     "action": row.get("corrective_action"), "kinds": [c["kind"] for c in cs],
                     "svc": svc, "keys": keys, "phases": phases,
                     "exact": (srcs["관측"] + srcs["측정"]) / sum(srcs.values())})  # fmt: skip
    return jobs, drop


def simulate(jobs: list[dict], policy: str, k: int, rate) -> dict[str, dict]:
    """K slot gate, 정적 key (enqueue 때 한 번). rate(kind, n) = 동시 n 개일 때 호출 진행 속도."""
    import heapq

    st = [{"p": -1, "wait": 0.0, "finish": None} for _ in jobs]
    events = [(j["arrival"], i, i) for i, j in enumerate(jobs)]
    heapq.heapify(events)
    order = len(jobs)
    waiting: list = []
    active: dict[int, list] = {}  # seq -> [남은 무부하 작업, job, call]
    seq = 0
    t = 0.0

    def advance(i: int):
        nonlocal seq, order
        s = st[i]
        s["p"] += 1
        ph = jobs[i]["phases"]
        if s["p"] >= len(ph):
            s["finish"] = t
            return
        kind, v = ph[s["p"]]
        if kind == "delay":
            order += 1
            heapq.heappush(events, (t + v, order, i))
        else:
            heapq.heappush(waiting, (jobs[i]["keys"][v][policy], seq, i, v, t))
            seq += 1

    while events or waiting or active:
        while waiting and len(active) < k:
            _, s_, i, j, te = heapq.heappop(waiting)
            st[i]["wait"] += t - te
            active[s_] = [jobs[i]["svc"][j], i, j]
        n = len(active)
        rates = {s_: 1.0 / rate(jobs[v[1]]["kinds"][v[2]], n) for s_, v in active.items()}
        tc = min((v[0] / rates[s_] for s_, v in active.items()), default=math.inf) + t
        te = events[0][0] if events else math.inf
        tn = min(tc, te)
        if tn == math.inf:
            raise RuntimeError("시뮬레이션 교착")
        for s_, v in active.items():
            v[0] -= rates[s_] * (tn - t)
        t = tn
        if tc <= te:
            for s_ in [s_ for s_, v in active.items() if v[0] <= 1e-9]:
                advance(active.pop(s_)[1])
        else:
            _, _, i = heapq.heappop(events)
            advance(i)
    return {j["key"]: {"latency": st[i]["finish"] - j["arrival"], "gate_wait": st[i]["wait"],
                       "arrival": j["arrival"], "action": j["action"]}
            for i, j in enumerate(jobs)}  # fmt: skip


def rel_boot(base: np.ndarray, new: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    """paired (같은 요청) bootstrap: 개선율 (base - new) / base x 100 (평균·p95)."""
    rng = np.random.default_rng(seed)
    n = len(base)
    out = {"n": n}
    if n < 2:
        return out
    idx = rng.integers(0, n, (n_boot, n))
    for name, f in (
        ("mean", np.mean),
        ("p95", lambda x, axis=None: np.percentile(x, 95, axis=axis)),
    ):
        b, w = f(base), f(new)
        bb, ww = f(base[idx], axis=1), f(new[idx], axis=1)
        imp = (bb - ww) / bb * 100
        out[f"{name}_impr_pct"] = float((b - w) / b * 100)
        out[f"{name}_impr_lo"] = float(np.percentile(imp, 2.5))
        out[f"{name}_impr_hi"] = float(np.percentile(imp, 97.5))
    return out


def sim_block(label: str, run: dict, jobs: list[dict], k: int, rate, scen: str,
              n_boot: int) -> tuple[list, list, dict]:  # fmt: skip
    res = {p: simulate(jobs, p, k, rate) for p in SIM_POLICIES}
    keys = [j["key"] for j in jobs]
    rows = []
    for p, r in res.items():
        lat = [r[x]["latency"] for x in keys]
        rows.append({"run": label, "scenario": scen, "policy": p, "n": len(lat),
                     "e2e_mean": mean(lat), "e2e_p50": pct(lat, 50), "e2e_p95": pct(lat, 95),
                     "e2e_p99": pct(lat, 99),
                     "gate_wait_mean": mean([r[x]["gate_wait"] for x in keys])})  # fmt: skip
    live_lat = np.array([run["rows"][x]["latency"] for x in keys])
    sim_fq = np.array([res["fcfs_query"][x]["latency"] for x in keys])
    calib = {"run": label, "scenario": scen, "n": len(keys),
             "live_mean": float(live_lat.mean()), "sim_mean": float(sim_fq.mean()),
             "sim_over_live_mean": float(sim_fq.mean() / live_lat.mean()),
             "live_p95": pct(live_lat, 95), "sim_p95": pct(sim_fq, 95),
             "sim_over_live_p95": pct(sim_fq, 95) / pct(live_lat, 95),
             "per_request_spearman": spearman(sim_fq, live_lat),
             "per_request_mae": float(np.abs(sim_fq - live_lat).mean()),
             "live_gate_wait_mean": mean([run["rows"][x]["query_state"]["cum_wait"]
                                          for x in keys])}  # fmt: skip
    pairs = []
    for b, n in PAIRS + (("fcfs_query", "oracle"),):
        x = np.array([res[b][q]["latency"] for q in keys])
        y = np.array([res[n][q]["latency"] for q in keys])
        pairs.append({"run": label, "scenario": scen, "base": b, "new": n,
                      **rel_boot(x, y, n_boot)})  # fmt: skip
    return rows, pairs, calib


def scenarios(sim_rows: list[dict], calib: list[dict]) -> list[dict]:
    """live fcfs_query 수치로 환산 [추정]. A = 상대 개선율 적용, B = gate wait 만 변화."""
    out = []
    by = {(r["run"], r["scenario"], r["policy"]): r for r in sim_rows}
    for c in calib:
        fq = by[(c["run"], c["scenario"], "fcfs_query")]
        for p in ("srpt_est", "srpt_state"):
            r = by[(c["run"], c["scenario"], p)]
            ra = r["e2e_mean"] / fq["e2e_mean"]
            gw_ratio = r["gate_wait_mean"] / fq["gate_wait_mean"] if fq["gate_wait_mean"] else 1
            b_mean = c["live_mean"] - c["live_gate_wait_mean"] * (1 - gw_ratio)
            out.append({"run": c["run"], "scenario": c["scenario"], "policy": p,
                        "A_live_mean_est": c["live_mean"] * ra,
                        "A_impr_vs_fcfs_pct": (1 - ra) * 100,
                        "A_live_p95_est": c["live_p95"] * r["e2e_p95"] / fq["e2e_p95"],
                        "A_p95_impr_vs_fcfs_pct": (1 - r["e2e_p95"] / fq["e2e_p95"]) * 100,
                        "B_live_mean_est": b_mean,
                        "B_impr_vs_fcfs_pct": (1 - b_mean / c["live_mean"]) * 100,
                        "sim_gate_wait_ratio": gw_ratio})  # fmt: skip
    return out


def cmd_expect(a) -> Path:
    warns = []
    ctx = load_ctx(a)
    model = cost_model_from_dict(read_json(Path(a.cost_model)))
    st, smeta = ss.load_state_model(a.state_model)
    if smeta["cost_model_sha256"] != ss.file_sha(a.cost_model):
        raise SystemExit("state model 을 만든 cost model 과 --cost-model 이 다름")
    runs, w = base_runs(a.base_root, a.k, "fcfs_query")
    warns += w
    if not runs:
        raise SystemExit(f"fcfs_query 기준 run 없음: {a.base_root}")
    dec_rows, dec_info, sim_rows, pairs, calib, slow, drops = [], {}, [], [], [], {}, {}
    for seed, (run, _d, pol, _rho) in sorted(runs.items()):
        label = f"s{seed}"
        leak = set(smeta.get("train_query_ids") or []) & {r["query_id"] for r in
                                                          run["rows"].values()}  # fmt: skip
        if leak:
            warns.append(f"{label}: state model 학습 질문과 {len(leak)}개 겹침")
        res = decision_analysis(label, run, pol, model, st, ctx)
        dec_rows += res["rows"]
        dec_info[label] = {k: v for k, v in res.items() if k != "rows"}
        jobs, drop = build_jobs(run, pol, ctx, model, st)
        drops[label] = {"n_jobs": len(jobs), "n_rows": len(run["rows"]), **drop,
                        "exact_noload_share_mean": mean([j["exact"] for j in jobs])}  # fmt: skip
        fit = fit_slowdown(run, pol, ctx, a.k)
        slow[label] = fit
        rate_models = {"empirical": slow_fn(fit, a.k)}
        if a.speedup:
            sp = parse_speedup(a.speedup)
            ks, vs = sorted(sp), [sp[x] for x in sorted(sp)]
            # 균일 처리율 g(n) (K=1 대비) -> 호출당 감속률 n / g(n)
            rate_models["speedup"] = lambda kind, n, ks=ks, vs=vs: (
                max(n, 1) / float(np.interp(max(n, 1), ks, vs))
            )
        for scen, rate in rate_models.items():
            r, p, c = sim_block(label, run, jobs, a.k, rate, scen, a.n_boot)
            sim_rows += r
            pairs += p
            calib.append(c)
    scen = scenarios(sim_rows, calib)
    res = {"inputs": vars(a), "warnings": warns, "state_model": {k: v for k, v in smeta.items()
                                                                if k != "train_query_ids"},
           "decisions": dec_rows, "decision_info": dec_info, "jobs": drops,
           "slowdown_fit": slow, "sim": sim_rows, "sim_pairs": pairs, "calibration": calib,
           "scenarios": scen, "targets": targets(pairs, calib, scen)}  # fmt: skip
    return write_out(
        a,
        res,
        expect_md,
        {
            "decisions.csv": dec_rows,
            "sim.csv": sim_rows,
            "sim_pairs.csv": pairs,
            "scenarios.csv": scen,
        },
    )


def targets(pairs: list[dict], calib: list[dict], scen: list[dict]) -> dict:
    """검증 목표 (사전 등록) 와 예상 범위. 예상 범위는 [추정]."""

    def rng(xs):
        xs = [x for x in xs if x is not None]
        return [min(xs), max(xs)] if xs else None

    fq = [p for p in pairs if (p["base"], p["new"]) == ("fcfs_query", "srpt_state")]
    m0 = [p for p in pairs if (p["base"], p["new"]) == ("srpt_est", "srpt_state")]
    worst_cal = max((abs(c["sim_over_live_mean"] - 1) for c in calib), default=None)
    reliable = worst_cal is not None and worst_cal <= 0.2
    return {
        "calibration_worst_mean_err": worst_cal,
        "numeric_expectation_reliable": reliable,
        "expected_mean_impr_vs_fcfs_pct_range": rng(
            [p.get("mean_impr_pct") for p in fq]
            + [s["B_impr_vs_fcfs_pct"] for s in scen if s["policy"] == "srpt_state"]),
        "expected_mean_impr_vs_m0_pct_range": rng([p.get("mean_impr_pct") for p in m0]),
        "expected_p95_impr_vs_fcfs_pct_range": rng([p.get("p95_impr_pct") for p in fq]),
        "rules": [
            "H1 (주): srpt_state 평균 E2E < fcfs_query — seed 층화 paired bootstrap (block) "
            "95% CI 상한 < 0 이면 '개선', 하한 > 0 '악화', 그 외 '구분 불가'",
            "H2: srpt_state 평균 E2E < srpt_est (같은 판정)",
            "H3: p95 E2E 에 대해 H1 / H2 (같은 판정)",
            "Q (품질 불변): 정확도 차이의 95% CI 가 0 을 포함하고 경로 불일치가 소수",
            "D (메커니즘): live srpt_state run 의 M2 != M0 선택 비율이 사전 결정 분석"
            " (top_differs_vs_M0) 과 같은 크기 수준. "
            "0 에 가까우면 지연 차이를 M2 로 설명할 수 없음",
            "조건별 (seed) 부호가 서로 다르면 결론을 '구분 불가' 로 낮춘다",
        ],
        "note": ("numeric_expectation_reliable=false 이면 예상 범위는 방향성 참고용이며 "
                 "개선율 목표로 쓰지 않는다. 어떤 경우에도 GPU 개선 보장이 아님"),
    }  # fmt: skip


def expect_md(res: dict, name: str) -> str:
    t = res["targets"]
    L = [
        f"# 상태 기반 SRPT-est (M2) 사전 예상 ({name})",
        "",
        "**시뮬레이션·로그 재구성 결과이며 GPU 개선율이 아니다. 보장된 성능이 아니다.**",
        "",
        "```",
        json.dumps(res["inputs"], ensure_ascii=False, indent=1, default=str),
        "```",
        "",
    ]
    if res["warnings"]:
        L += ["## 경고", ""] + [f"- {w}" for w in res["warnings"]] + [""]
    L += ["## 1. 결정 차이 (기존 fcfs_query live 대기열, 실제 dispatch 는 fcfs_query)", "",
          "- top_differs_vs_M0: 같은 대기열에서 그 규칙이 고를 호출이 M0 와 다른 비율",
          "- truth_pair_concordance: 경쟁 호출 쌍의 무부하 잔여 작업 [관측·측정] 순위와 일치율",
          "- regret_mean: 고른 호출의 잔여 작업 - 최소 잔여 작업 (초)", ""]  # fmt: skip
    L += md_table(res["decisions"], ["run", "rule", "n_decisions", "top_differs_vs_M0",
                                     "top_differs_vs_actual", "pair_order_change_vs_M0",
                                     "truth_pair_concordance", "truth_pairs",
                                     "regret_mean"])  # fmt: skip
    L += ["", "```", json.dumps(res["decision_info"], ensure_ascii=False, indent=1), "```", "",
          "## 2. trace replay 시뮬레이션 [시뮬레이션]", "",
          f"- 요청: {json.dumps(res['jobs'], ensure_ascii=False)}",
          "- empirical: 감속률 slow(종류, 동시 수) = 기준 live 의 wall / 무부하 중앙값 (in-sample)",
          "", "### 보정 (재생 fcfs_query vs 실제 fcfs_query live)", ""]  # fmt: skip
    L += md_table(res["calibration"], ["run", "scenario", "n", "live_mean", "sim_mean",
                                       "sim_over_live_mean", "live_p95", "sim_p95",
                                       "sim_over_live_p95", "per_request_spearman",
                                       "per_request_mae", "live_gate_wait_mean"])  # fmt: skip
    L += ["", "### 정책별", ""]
    L += md_table(res["sim"], ["run", "scenario", "policy", "n", "e2e_mean", "e2e_p50",
                               "e2e_p95", "e2e_p99", "gate_wait_mean"])  # fmt: skip
    L += ["", "### 개선율 (base - new) / base x 100, paired bootstrap 95% CI (요청 재표본만)", ""]
    L += md_table(res["sim_pairs"], ["run", "scenario", "base", "new", "n", "mean_impr_pct",
                                     "mean_impr_lo", "mean_impr_hi", "p95_impr_pct",
                                     "p95_impr_lo", "p95_impr_hi"])  # fmt: skip
    L += ["", "## 3. live 수치 환산 시나리오 [추정]", "",
          "- A: sim 상대 비율을 live fcfs_query 평균 / p95 에 적용",
          "- B: live 의 실행시간은 그대로, gate wait 만 sim 비율로 변화", ""]  # fmt: skip
    L += md_table(res["scenarios"], ["run", "scenario", "policy", "A_live_mean_est",
                                     "A_impr_vs_fcfs_pct", "A_live_p95_est",
                                     "A_p95_impr_vs_fcfs_pct", "B_live_mean_est",
                                     "B_impr_vs_fcfs_pct", "sim_gate_wait_ratio"])  # fmt: skip
    L += ["", "## 4. 검증 목표 (사전 등록)", "",
          "```", json.dumps({k: v for k, v in t.items() if k != "rules"}, ensure_ascii=False,
                            indent=1), "```", ""] + [f"- {r}" for r in t["rules"]]  # fmt: skip
    L += ["", "## 재현하지 못하는 것 (한계)", "",
          "- vLLM continuous batching·prefix cache·prefill/decode 혼합: 감속률은 (종류, 동시 수) "
          "만의 함수로 근사. 순서가 바뀌면 같은 동시 수에서도 실행시간이 달라질 수 있음",
          "- 감속률은 fcfs_query run 에서 추정 (in-sample): 정책을 바꾸면 동시 실행 구성이 바뀜",
          "- 무부하 시간 일부는 토큰 모델 [추정] (exact_noload_share), 일부 요청은 제외",
          "- 같은 요청의 grading 결과·경로가 run 마다 달라질 수 있음 (vLLM 비결정성) — 미반영",
          "- CI 는 요청 재표본만 반영. 모델 오차·run 간 변동·seed 간 변동은 들어가지 않음",
          "- M2 예측력 (MAE) 개선은 오프라인 결과. 지연 개선과 같은 것이 아님", ""]  # fmt: skip
    return "\n".join(L)


# ================================================================ report (Part D)
IGNORE_CFG = ("exp_id", "scheduler.policy", "scheduler.state_model", "scheduler.state_shadow",
              "run.session", "output_dir")  # fmt: skip


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
    else:
        out[p] = d
    return out


def env_diff(base_dir: Path, new_dir: Path) -> dict:
    def load(d):
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) if (
            d / "config.yaml").exists() else {}  # fmt: skip
        meta = read_json(d / "run_meta.json") if (d / "run_meta.json").exists() else {}
        return flat(cfg), meta

    cb, mb = load(base_dir)
    cn, mn = load(new_dir)
    cfg_diff = {k: [cb.get(k), cn.get(k)] for k in sorted(set(cb) | set(cn))
                if cb.get(k) != cn.get(k) and not k.startswith(IGNORE_CFG)}  # fmt: skip
    pick = {"git.commit": ("git", "commit"), "git.dirty": ("git", "dirty"),
            "engines": ("engines",), "packages": ("packages",), "gpu": ("gpu",),
            "host": ("host", "hostname"),
            "workload_sha256": ("workload", "sha256")}  # fmt: skip

    def get(m, path):
        for p in path:
            m = (m or {}).get(p) if isinstance(m, dict) else None
        return m

    meta_diff = {}
    for name, path in pick.items():
        x, y = get(mb, path), get(mn, path)
        if name == "gpu":  # 실시간 사용량은 제외, 이름·드라이버만
            x, y = _gpu_ident(x), _gpu_ident(y)
        if x != y:
            meta_diff[name] = [x, y] if name not in ("engines", "packages") else _dict_diff(x, y)
    critical = [k for k in cfg_diff if not k.startswith(("run.sample", "run.purpose"))]
    critical += [k for k in meta_diff if k in ("engines", "packages", "gpu", "workload_sha256")]
    return {"config_diff": cfg_diff, "meta_diff": meta_diff, "critical": critical}


def _gpu_ident(g):
    if not isinstance(g, list | dict):
        return g
    items = g if isinstance(g, list) else [g]
    return [{k: v for k, v in (x or {}).items() if k in ("name", "driver_version", "memory_total",
                                                         "index")}
            for x in items if isinstance(x, dict)]  # fmt: skip


def _dict_diff(x, y):
    fx, fy = flat(x or {}), flat(y or {})
    return {
        k: [fx.get(k), fy.get(k)]
        for k in sorted(set(fx) | set(fy))
        if fx.get(k) != fy.get(k) and not any(s in k for s in ("time", "uptime", "pid"))
    }


def run_decisions(label: str, run: dict, d: Path) -> dict | None:
    """새 run 의 state_estimates (호출별 M0 / M2) + gate 로그 -> 결정 변경률 (M2 vs M0)."""
    log = {e["seq"]: e for e in read_jsonl(d / "state_estimates.jsonl")}
    if not log:
        return None
    tickets = {}
    for e in run["log"]:
        x = log.get(e["seq"])
        if x is None or x.get("m2") is None:
            continue
        tickets[e["seq"]] = {"seq": e["seq"], "enq": e["t"] - e["wait"], "grant": e["t"],
                             "kind": e["kind"], "true_rem": None,
                             "est": {"M0": x["m0"], "M2": x["m2"],
                                     "fcfs_query": x["arrival"]}}  # fmt: skip
    res = decisions_from_tickets(label, run, tickets, ("fcfs_query", "M0", "M2"))
    diff = [abs(x["m2"] - x["m0"]) for x in log.values() if x.get("m2") is not None]
    res["est_abs_diff_mean"] = mean(diff)
    res["est_differs_share"] = mean([v > 1e-9 for v in diff])
    return res


def cmd_report(a) -> Path:
    warns = []
    ctx = load_ctx(a)
    roots = {"fcfs_query": a.base_root, "srpt_est": a.m0_root, "srpt_state": a.m2_root}
    runs = {}
    for sched, root in roots.items():
        r, w = base_runs(root, a.k, sched)
        warns += w
        for seed, v in r.items():
            runs[(sched, seed)] = v
    seeds = sorted({s for _, s in runs})
    summ, dec_rows, envs, over, checks = [], [], [], [], []
    for (sched, seed), (run, d, pol, rho) in sorted(runs.items()):
        dec = decompose(run, pol, ctx["seq_calls"], ctx["seq_rows"], ctx["evc"], ctx["noload"])
        rs = run_summary(sched, "live", (rho, seed, pol, a.k), run, d, dec, a.k)
        rs["scheduler"] = sched
        rs["accuracy"] = rs.pop("acc")
        summ.append(rs)
        rd = run_decisions(f"{sched}/s{seed}", run, d)
        if rd:
            dec_rows += [
                {
                    **x,
                    "est_differs_share": rd["est_differs_share"],
                    "est_abs_diff_mean": rd["est_abs_diff_mean"],
                }
                for x in rd["rows"]
            ]
        ov = d / "scheduler_overhead.json"
        if ov.exists():
            o = read_json(ov)
            over.append(
                {
                    "run": f"{sched}/s{seed}",
                    "n": o["used"].get("n"),
                    "used_mean_us": o["used"].get("mean_us"),
                    "used_p99_us": o["used"].get("p99_us"),
                    "used_max_us": o["used"].get("max_us"),
                    "m2_mean_us": o["m2"].get("mean_us"),
                    "used_total_over_llm_wall": o.get("used_total_over_llm_wall"),
                }
            )
        ck = d / "state_check.json"
        if ck.exists():
            checks.append({"run": f"{sched}/s{seed}", **read_json(ck)})
            if read_json(ck).get("score_mismatch_rows"):
                warns.append(f"{sched}/s{seed}: tap 점수와 결과 점수 불일치 행 있음")
        if sched != "fcfs_query" and ("fcfs_query", seed) in runs:
            e = env_diff(runs[("fcfs_query", seed)][1], d)
            envs.append({"run": f"{sched}/s{seed}", **e})
            if e["critical"]:
                warns.append(
                    f"{sched}/s{seed}: 기준 fcfs_query 와 환경·설정 차이 {e['critical']}"
                    " -> fcfs_query 재측정 권장 (SCHEDS 에 fcfs_query 추가)"
                )
    pairs = []
    for b, n in PAIRS:
        per = {s: pair_arrays(runs[(b, s)][0], runs[(n, s)][0]) for s in seeds
               if (b, s) in runs and (n, s) in runs}  # fmt: skip
        if not per:
            continue
        groups = {f"s{s}": [s] for s in per}
        if len(per) > 1:
            groups["pooled"] = list(per)
        for g, ss_ in groups.items():
            for method in ("iid", "block"):
                st = [per[s]["subsets"]["all"] for s in ss_]
                r = boot(st, a.n_boot, 0, method, a.block)
                pairs.append({"base": b, "new": n, "group": g, "method": method,
                              "path_mismatch": sum(per[s]["path_mismatch"] for s in ss_),
                              **{k: r[k] for k in ("n_pairs", "base_mean", "target_mean",
                                                   "mean_diff", "mean_lo", "mean_hi",
                                                   "p95_diff", "p95_lo", "p95_hi", "p99_diff",
                                                   "p99_lo", "p99_hi", "mean_rel",
                                                   "mean_rel_lo", "mean_rel_hi")},
                              "impr_pct": -r["mean_rel"] * 100,
                              "impr_lo": -r["mean_rel_hi"] * 100,
                              "impr_hi": -r["mean_rel_lo"] * 100,
                              "verdict_mean": verdict(r["mean_lo"], r["mean_hi"]),
                              "verdict_p95": verdict(r["p95_lo"], r["p95_hi"])})  # fmt: skip
        for s in per:  # 정확도 (같은 요청 쌍)
            rb, rn = runs[(b, s)][0]["rows"], runs[(n, s)][0]["rows"]
            ks = sorted(set(rb) & set(rn))
            x = np.array([rb[k].get("acc") or 0.0 for k in ks])
            y = np.array([rn[k].get("acc") or 0.0 for k in ks])
            lo, hi = _boot_diff(x, y, a.n_boot)
            pairs.append({"base": b, "new": n, "group": f"s{s}", "method": "accuracy",
                          "n_pairs": len(ks), "mean_diff": float(y.mean() - x.mean()),
                          "mean_lo": lo, "mean_hi": hi,
                          "path_mismatch": per[s]["path_mismatch"]})  # fmt: skip
    exp = read_json(Path(a.expect)) if a.expect else None
    res = {"inputs": vars(a), "warnings": warns, "runs": summ, "pairs": pairs,
           "decisions": dec_rows, "overhead": over, "state_check": checks, "env": envs,
           "expect_vs_actual": expect_vs_actual(exp, pairs) if exp else None}  # fmt: skip
    return write_out(
        a,
        res,
        report_md,
        {"runs.csv": summ, "pairs.csv": pairs, "decisions.csv": dec_rows, "overhead.csv": over},
    )


def verdict(lo, hi) -> str:
    if lo is None or hi is None:
        return "-"
    if hi < 0:
        return "개선"
    if lo > 0:
        return "악화"
    return "구분 불가"


def _boot_diff(x, y, n_boot, seed=0):
    if len(x) < 2:
        return None, None
    idx = np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))
    d = y[idx].mean(axis=1) - x[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def expect_vs_actual(exp: dict, pairs: list[dict]) -> dict:
    t = exp.get("targets") or {}
    act = {
        f"{p['base']}->{p['new']}": {
            "impr_pct": p["impr_pct"],
            "ci": [p["impr_lo"], p["impr_hi"]],
            "verdict": p["verdict_mean"],
        }
        for p in pairs
        if p.get("group") == "pooled" and p.get("method") == "block"
    }
    dec = {(r["run"], r["rule"]): r["top_differs_vs_M0"] for r in exp.get("decisions") or []}
    return {"expected_mean_impr_vs_fcfs_pct_range": t.get("expected_mean_impr_vs_fcfs_pct_range"),
            "expected_mean_impr_vs_m0_pct_range": t.get("expected_mean_impr_vs_m0_pct_range"),
            "numeric_expectation_reliable": t.get("numeric_expectation_reliable"),
            "expected_top_differs_M2_vs_M0": {k[0]: v for k, v in dec.items() if k[1] == "M2"},
            "actual_pooled_block": act}  # fmt: skip


def report_md(res: dict, name: str) -> str:
    L = [f"# 상태 기반 SRPT-est (M2) live 비교 ({name})", "",
         "개선율 = (Base - New) / Base x 100 (양수 = New 가 빠름). 판정은 seed 층화 paired "
         "bootstrap 95% CI (block 을 보수적 값으로)", ""]  # fmt: skip
    if res["warnings"]:
        L += ["## 경고", ""] + [f"- {w}" for w in res["warnings"]] + [""]
    L += ["## 정책·run 별", ""]
    L += md_table(res["runs"], ["scheduler", "seed", "n", "n_errors", "accuracy", "e2e_mean",
                                "e2e_p50", "e2e_p95", "e2e_p99", "gate_wait_mean", "exec_mean",
                                "throughput", "thr_over_lambda", "gpu_util_mean",
                                "slots_full_share", "drain_after_last_arrival_s",
                                "workload_sha256"])  # fmt: skip
    L += ["", "## 정책 쌍 (paired, 같은 요청)", ""]
    lat = [p for p in res["pairs"] if p["method"] != "accuracy"]
    L += md_table(lat, ["base", "new", "group", "method", "n_pairs", "base_mean", "target_mean",
                        "impr_pct", "impr_lo", "impr_hi", "mean_diff", "mean_lo", "mean_hi",
                        "verdict_mean", "p95_diff", "p95_lo", "p95_hi", "verdict_p95",
                        "p99_diff", "path_mismatch"])  # fmt: skip
    L += ["", "### 정확도 차이 (new - base, 같은 요청)", ""]
    L += md_table([p for p in res["pairs"] if p["method"] == "accuracy"],
                  ["base", "new", "group", "n_pairs", "mean_diff", "mean_lo", "mean_hi",
                   "path_mismatch"])  # fmt: skip
    if res["decisions"]:
        L += ["", "## 결정 변경률 (각 run 의 실제 대기열, 기록된 M0 / M2)", ""]
        L += md_table(res["decisions"], ["run", "rule", "n_decisions", "top_differs_vs_M0",
                                         "top_differs_vs_actual", "pair_order_change_vs_M0",
                                         "est_differs_share",
                                         "est_abs_diff_mean"])  # fmt: skip
    if res["overhead"]:
        L += ["", "## Scheduler overhead (정책 base 계산, gate lock 안)", ""]
        L += md_table(res["overhead"], ["run", "n", "used_mean_us", "used_p99_us",
                                        "used_max_us", "m2_mean_us",
                                        "used_total_over_llm_wall"])  # fmt: skip
    if res["state_check"]:
        L += ["", "## 관측 정합성 (tap 점수 = 결과 점수)", ""]
        L += md_table(res["state_check"], ["run", "rows_checked", "score_mismatch_rows",
                                           "grade_idx_mismatch_rows", "estimates_logged",
                                           "gate_dispatches",
                                           "m0_eq_m2_share_generate"])  # fmt: skip
    L += [
        "",
        "## 환경 차이 (기준 fcfs_query run 대비)",
        "",
        "```",
        json.dumps(res["env"], ensure_ascii=False, indent=1, default=str),
        "```",
        "",
    ]
    if res["expect_vs_actual"]:
        L += ["## 사전 예상 vs 실제", "", "```",
              json.dumps(res["expect_vs_actual"], ensure_ascii=False, indent=1), "```", "",
              "- 실제가 예상 범위 밖이면: 보정 오차 (expect calibration), 결정 변경률 차이, "
              "실행시간 (exec_mean) 변화, 경로 불일치 순서로 원인을 본다", ""]  # fmt: skip
    L += ["## 해석 주의", "",
          "- fcfs_query 는 기존 EQL1 run 재사용 "
          "(다른 시각·다른 코드 커밋일 수 있음, 환경 차이 참고)",
          "- seed 2개, 조건당 run 1개: run 간 변동 (vLLM 비결정성) 은 CI 에 없음",
          "- 결정 변경률이 낮으면 지연 차이를 M2 효과로 설명하기 어렵다", ""]  # fmt: skip
    return "\n".join(L)


# ================================================================ 공통
def write_out(a, res: dict, md_fn, csvs: dict) -> Path:
    base = Path(a.out_root) / "state_sched" / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out, n = base, 1
    while out.exists():
        n += 1
        out = base.with_name(f"{base.name}_{n}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str),
                                       encoding="utf-8")  # fmt: skip
    for fn, rows in csvs.items():
        write_csv(out / fn, rows)
    md = md_fn(res, a.name)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not a.quiet:
        print(md)
    print(f"saved -> {out}")
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("expect", "report"):
        p = sub.add_parser(name)
        p.add_argument("--base-root", required=True, help="fcfs_query 기준 live root (EQL1)")
        p.add_argument("--seq", required=True)
        p.add_argument("--evc-seq", default=None)
        p.add_argument("--k", type=int, default=4)
        p.add_argument("--n-boot", type=int, default=2000)
        p.add_argument("--name", default=f"state-sched-{name}")
        p.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
        p.add_argument("--quiet", action="store_true")
        if name == "expect":
            p.add_argument("--cost-model", required=True)
            p.add_argument("--state-model", required=True)
            p.add_argument("--speedup", default=None, help="예: 1:1,2:1.7,4:2.6 (추가 시나리오)")
        else:
            p.add_argument("--m0-root", required=True)
            p.add_argument("--m2-root", required=True)
            p.add_argument("--expect", default=None, help="expect 결과 analysis.json")
            p.add_argument("--block", type=int, default=20)
    a = ap.parse_args(argv)
    return cmd_expect(a) if a.cmd == "expect" else cmd_report(a)


if __name__ == "__main__":
    main()
