"""Grading 중간 정보의 스케줄링 유용성 오프라인 ablation (읽기 전용, 추가 추론·API 호출 없음).

질문: grading 도중에 관측되는 문서별 점수가 남은 작업량 예측과 우선순위 결정에 도움이 되는가.
기존 정책 (SRPT-est) 은 바꾸지 않고, 같은 입력 위치에서 추정식만 바꿔 비교한다.

    uv run python experiments/crag_system/grade_info_ablation.py \
        --train results/crag_system/pre/PRE-SV1-TRAIN/train \
        --seq results/crag_system/pre/PRE-SV1-A/eval_seq \
        --cost-model results/crag_system/cost_models/SV1_train_wall.json \
        --live-root results/crag_system/pre/PRE-SV1-LIVE --k 4 --rhos 0.8 0.95 --seeds 0 1 \
        --workloads results/crag_system/workloads --tag SV1 \
        --calibration <PRE-K/..._calibration/calibration.json> --name SV1-grade-ablation

추정 조건 (모두 "호출이 gate 에 들어가는 시점" 에 계산, 작을수록 먼저)
  A  A_srpt_est      기존 SRPT-est 추정식 그대로 (cost_model.est_remaining)
  B  B_stage         진행 단계만: train trace 의 E[남은 작업 | 호출 종류, grade 순번].
                     retrieval 특징·prompt 길이·점수 미사용.
                     A 와 겹치는 정보: A 도 남은 grade 수 (n_docs - idx) 와 호출 종류를 쓴다.
                     rewrite 단계에 있다는 사실 자체가 correction 경로를 뜻하므로
                     (이 파이프라인 구조), rewrite 호출에서는 B 도 경로 정보를 갖는다 (A 와 같음).
  C  C1_grade_hit    A 와 같되 grade 호출 idx>=1 에서 grading 이후 작업량 항 pred_post(feats) 를
                     이미 끝난 grade 호출 (0..idx-1) 의 점수로 바꾼다.
                     점수 >= upper 가 나왔으면 correct 확정 -> E[post | not corrected],
                     아니면 E[post] 를 P(corrected | idx, 미확정) 로 섞음
     C2_grade_score  C1 + 미확정일 때 관측 점수 구간 (모두 < lower / 그 외) 으로 P 를 나눔
                     idx=0 (관측 없음) 과 rewrite / generate 호출은 A 와 같다 -> A 와의 차이는
                     grading 중간 점수에서만 온다.
  D  D_oracle        실제 남은 작업 (eval sequential run 의 같은 query 호출 wall 합). 참고용 상한

정보 누수 방지
  - 학습 (B 표, C 확률, A cost model) 은 train trace (--train) 만. eval 질문과 겹치면 중단
  - C 는 idx 번째 grade 호출의 추정에 scores[:idx] 만 넘긴다 (함수 인자로 강제, 길이 검사).
    최종 corrective_action / 이후 점수는 C 에 들어가지 않는다
  - eval sequential run 은 정답 (D, 오차 기준) 과 시뮬레이션 환경 (서비스 시간·경로) 으로만 쓴다
  - live 대기열 분석에서 C 의 점수는 그 live run 에서 관측된 점수. 직전 grade 호출이
    enqueue 전에 끝났는지 calls 시각으로 확인하고, 아니면 그 ticket 의 C 는 계산하지 않는다

평가
  1 점추정 (eval sequential trace 의 모든 호출 시점): MAE, 편향, Spearman. 단계 x 경로별
  2 live 대기열 (SV1 run 의 복원 대기열, diag_sv_policy 와 같은 방법): 동시 대기 쌍 순위 일치율,
    실제 선택 / fcfs_query / A 대비 선택 차이, regret. 같은 대기열의 즉시 선택만 비교한다
  3 오프라인 스케줄링 시뮬레이션 (--workloads 가 있을 때): SV1 workload 의 도착 시각 +
    eval sequential trace 의 경로·호출 wall·retrieval·web 지연, K slot gate
    (호출 단위 비선점, key 는 enqueue 때 한 번 계산 = 실제 gate 와 같음). GPU 처리 모델 (가정):
      ps          동시에 n 개 실행 중이면 각 호출이 g(n)/n 속도로 진행. g(n) = K calibration 의
                  K 별 LLM 호출 처리율 / K=1 처리율 (선형 보간). --speedup 으로 직접 지정 가능
      independent 호출마다 sequential wall 그대로 (batching 간섭 없음, 낙관적)
    시뮬레이션 지연은 모델 값이다. --live-root 가 있으면 같은 workload 의 live 결과
    (fcfs / fcfs_query / srpt_nopath / srpt_est) 와 나란히 두어 모델 타당성을 먼저 확인한다.

결과: results/crag_system/analysis/<name>/<timestamp>/ (덮어쓰지 않음)
  report.md, analysis.json, point_estimates.csv, queue_metrics.csv,
  sim_summary.csv, sim_paired.csv, sim_requests.csv
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_sv_policy import (  # noqa: E402
    ROOT,
    build_tickets,
    corrected,
    discover,
    load_run,
    load_seq,
    md_table,
    mean,
    pct,
    qfeats,
    read_json,
    replay,
    spearman,
    write_csv,
)
from paired_bootstrap_sv import boot  # noqa: E402

from rag.scheduling.cost_model import GRADE_KINDS, cost_model_from_dict  # noqa: E402

ESTIMATORS = ("A_srpt_est", "B_stage", "C1_grade_hit", "C2_grade_score", "D_oracle")
SIM_POLICIES = ("fcfs", "fcfs_query", "srpt_nopath", *ESTIMATORS)
LIVE_FOR_SIM = {"fcfs": "fcfs", "fcfs_query": "fcfs_query", "srpt_nopath": "srpt_nopath",
                "A_srpt_est": "srpt_est"}  # fmt: skip
SIM_PAIRS = (
    ("A_srpt_est", "B_stage"), ("A_srpt_est", "C1_grade_hit"), ("A_srpt_est", "C2_grade_score"),
    ("fcfs_query", "C2_grade_score"), ("fcfs_query", "A_srpt_est"), ("A_srpt_est", "D_oracle"),
)  # fmt: skip
EPS = 1e-9


# ---------------------------------------------------------------- trace
@dataclass
class TQ:
    """trace 의 요청 하나 (부하 없는 sequential run)."""

    key: str
    qid: str
    corrected: bool
    feats: np.ndarray
    n_docs: int
    scores: list
    calls: list  # [{kind, idx, chars, wall}] 실행 순서
    retrieval: float
    web: float

    def remaining(self, j: int) -> float:
        return float(sum(c["wall"] for c in self.calls[j:]))

    @property
    def post_work(self) -> float:
        return float(sum(c["wall"] for c in self.calls if c["kind"] not in GRADE_KINDS))


def grade_scores(row: dict) -> list:
    return list(((row.get("extra") or {}).get("grading") or {}).get("scores_full") or [])


def trace_queries(run: dict) -> list[TQ]:
    out = []
    for key, row in run["rows"].items():
        feats, n_docs = qfeats(row)
        st = row.get("stage_latencies") or {}
        out.append(
            TQ(
                key=key, qid=row["query_id"], corrected=corrected(row), feats=feats,
                n_docs=n_docs, scores=grade_scores(row),
                calls=[{"kind": c["kind"], "idx": c["idx"], "chars": c["prompt_chars"],
                        "wall": float(c["wall"])} for c in run["calls"].get(key, [])],
                retrieval=float(st.get("retrieval", 0.0)), web=float(st.get("web_search", 0.0)),
            )
        )  # fmt: skip
    return out


def grading_check(run: dict) -> dict:
    """grade 호출 idx i 가 문서 i 이고, 직전 grade 호출이 끝난 뒤 enqueue 됐는지
    (= 점수 관측 가능)."""
    n_req = bad_order = bad_count = 0
    strategies = {run["meta"].get("grading")}
    for key, row in run["rows"].items():
        gs = [c for c in run["calls"].get(key, []) if c["kind"] == "grade"]
        if not gs:
            continue
        n_req += 1
        gs.sort(key=lambda c: c["idx"])
        if len(gs) != len([s for s in grade_scores(row) if s is not None]):
            bad_count += 1
        for prev, cur in zip(gs, gs[1:], strict=False):
            if cur["t_enqueue"] < prev["t_end"] - 1e-6:
                bad_order += 1
                break
    return {
        "grading_strategy": sorted(map(str, strategies)),
        "requests_with_grade": n_req,
        "not_sequential": bad_order,
        "score_count_mismatch": bad_count,
    }


def observable_prefix_ok(run: dict, key: str, idx: int) -> bool:
    """live ticket (grade idx) 의 enqueue 시점에 grade 0..idx-1 이 모두 끝났는가."""
    cs = {c["idx"]: c for c in run["calls"].get(key, []) if c["kind"] == "grade"}
    cur = cs.get(idx)
    if cur is None:
        return False
    return all(j in cs and cs[j]["t_end"] <= cur["t_enqueue"] + 1e-6 for j in range(idx))


# ---------------------------------------------------------------- estimators
def stage_key(kind: str, idx: int) -> str:
    return f"grade{idx}" if kind in GRADE_KINDS else kind


def score_bucket(prefix: list, upper: float, lower: float, fine: bool) -> str:
    obs = [s for s in prefix if s is not None]
    if any(s >= upper for s in obs):
        return "hit"
    if not fine:
        return "nohit"
    return "all_low" if all(s < lower for s in obs) else "mid"


class Tables:
    """train trace 만으로 학습한 B / C 의 조건부 평균·확률."""

    def __init__(self, train: list[TQ], upper: float, lower: float, min_cell: int = 10):
        self.upper, self.lower, self.min_cell = upper, lower, min_cell
        rem = defaultdict(list)
        for q in train:
            for j, c in enumerate(q.calls):
                rem[stage_key(c["kind"], c["idx"])].append(q.remaining(j))
        self.stage = {k: float(np.mean(v)) for k, v in rem.items()}
        self.stage_n = {k: len(v) for k, v in rem.items()}
        allrem = [x for v in rem.values() for x in v]
        self.stage_default = float(np.mean(allrem)) if allrem else 0.0
        corr = [q.post_work for q in train if q.corrected]
        notc = [q.post_work for q in train if not q.corrected]
        self.post_corr = float(np.mean(corr)) if corr else 0.0
        self.post_not = float(np.mean(notc)) if notc else 0.0
        self.p_global = len(corr) / len(train) if train else 0.0
        self.cells = {"c1": defaultdict(lambda: [0, 0]), "c2": defaultdict(lambda: [0, 0])}
        max_docs = max((len(q.scores) for q in train), default=0)
        for q in train:
            for i in range(1, max_docs):
                if i > len(q.scores):
                    break
                pre = q.scores[:i]
                for name, fine in (("c1", False), ("c2", True)):
                    for key in ((i, score_bucket(pre, upper, lower, fine)), ("any", score_bucket(
                            pre, upper, lower, fine))):  # fmt: skip
                        cell = self.cells[name][key]
                        cell[0] += q.corrected
                        cell[1] += 1

    def stage_est(self, kind: str, idx: int) -> float:
        return self.stage.get(stage_key(kind, idx), self.stage_default)

    def p_corrected(self, name: str, idx: int, prefix: list) -> tuple[float, str]:
        b = score_bucket(prefix, self.upper, self.lower, name == "c2")
        if b == "hit":
            return 0.0, "hit"  # full grading 규칙: score >= upper 가 하나라도 있으면 correct
        for key in ((idx, b), ("any", b)):
            k, n = self.cells[name].get(key, (0, 0))
            if n >= self.min_cell:
                return k / n, f"{key[0]}:{b}"
        return self.p_global, "global"

    def post_hat(self, name: str, idx: int, prefix: list) -> float:
        p, _ = self.p_corrected(name, idx, prefix)
        return p * self.post_corr + (1 - p) * self.post_not

    def to_dict(self) -> dict:
        return {
            "stage_mean": self.stage, "stage_n": self.stage_n,
            "post_corrected_mean": self.post_corr, "post_not_corrected_mean": self.post_not,
            "p_corrected_global": self.p_global,
            "cells": {name: {f"{k[0]}:{k[1]}": {"n_corrected": v[0], "n": v[1]}
                             for k, v in cells.items()} for name, cells in self.cells.items()},
        }  # fmt: skip


def estimates(model, T: Tables, qid, kind, idx, chars, n_docs, feats, prefix, truth=None) -> dict:
    """한 호출 시점의 추정치. prefix = 이 호출 전에 끝난 grade 호출의 점수 (scores[:idx]).

    누수 방지: grade 호출이면 len(prefix) == idx 여야 한다 (이후 점수를 넘기면 오류).
    prefix=None 이면 관측 불가 -> C 는 None.
    """
    m = model.for_query(qid)
    a = m.est_remaining(kind, idx, chars, n_docs, feats)
    out = {"A_srpt_est": a, "B_stage": T.stage_est(kind, idx), "D_oracle": truth}
    if kind == "grade" and idx >= 1:
        if prefix is None:
            out["C1_grade_hit"] = out["C2_grade_score"] = None
        else:
            if len(prefix) != idx:
                raise ValueError(f"prefix 길이 {len(prefix)} != idx {idx} (미래 점수 누수 방지)")
            grade_work = a - m.pred_post(feats)
            out["C1_grade_hit"] = grade_work + T.post_hat("c1", idx, prefix)
            out["C2_grade_score"] = grade_work + T.post_hat("c2", idx, prefix)
    else:
        out["C1_grade_hit"] = out["C2_grade_score"] = a
    return out


def point_estimate(q: TQ, j: int, model, T: Tables) -> dict:
    """trace 요청 q 의 j 번째 호출 시점 추정치. 관측 가능한 점수 (scores[:idx]) 만 넘긴다.
    q.corrected (최종 경로) 와 이후 점수는 C 에 들어가지 않는다 (D 의 정답만 미래 정보)."""
    c = q.calls[j]
    prefix = q.scores[: c["idx"]] if c["kind"] == "grade" else []
    return estimates(model, T, q.qid, c["kind"], c["idx"], c["chars"], q.n_docs, q.feats,
                     prefix, q.remaining(j))  # fmt: skip


def nopath_est(model, qid, kind, idx, chars, n_docs, feats) -> float:
    m = model.for_query(qid)
    if kind == "rewrite":
        return m.pred_post(feats)
    return m.est_remaining(kind, idx, chars, n_docs, feats)


# ---------------------------------------------------------------- 1 point estimates
def point_rows(eval_q: list[TQ], model, T: Tables) -> list[dict]:
    pts = defaultdict(list)  # (stage, path) -> [(est dict, truth)]
    for q in eval_q:
        for j, c in enumerate(q.calls):
            truth = q.remaining(j)
            e = point_estimate(q, j, model, T)
            path = "corrected" if q.corrected else "not_corrected"
            st = stage_key(c["kind"], c["idx"])
            groups = [st, "all"] + (["grade_all"] if c["kind"] in GRADE_KINDS else [])
            for g in groups:
                for p in (path, "all"):
                    pts[(g, p)].append((e, truth))
    out = []
    for (g, p), xs in sorted(pts.items()):
        for name in ESTIMATORS:
            pairs = [(e[name], t) for e, t in xs if e[name] is not None]
            if not pairs:
                continue
            est = np.array([x for x, _ in pairs])
            tru = np.array([t for _, t in pairs])
            out.append(
                {
                    "stage": g, "path": p, "estimator": name, "n": len(pairs),
                    "est_mean": float(est.mean()), "true_mean": float(tru.mean()),
                    "bias": float((est - tru).mean()), "mae": float(np.abs(est - tru).mean()),
                    "spearman": spearman(est, tru),
                }
            )  # fmt: skip
    return out


# ---------------------------------------------------------------- 2 live queues
def queue_metrics(cond: str, label: str, run: dict, model, T: Tables, seq: dict) -> dict:
    tickets = build_tickets(run, model, seq)
    n_unobs = 0
    for t in tickets.values():
        row = run["rows"][t["rk"]]
        feats, n_docs = qfeats(row)
        prefix = None
        if t["kind"] == "grade":
            if observable_prefix_ok(run, t["rk"], t["idx"]):
                prefix = grade_scores(row)[: t["idx"]]
            else:
                n_unobs += t["idx"] >= 1
        e = estimates(model, T, t["qid"], t["kind"], t["idx"], t["chars"], n_docs, feats,
                      prefix if t["kind"] == "grade" else [], t["true_rem"])  # fmt: skip
        t["est"].update({k: v for k, v in e.items()})
    rules = ("fcfs_query", *ESTIMATORS)
    agree = defaultdict(int)
    avail = defaultdict(int)
    vs_fq = defaultdict(int)
    vs_a = defaultdict(int)
    regret = defaultdict(list)
    conc, conc_n = defaultdict(int), defaultdict(int)
    n_choice = 0

    def pick(cands, r):
        if any(c["est"].get(r) is None for c in cands):
            return None
        return min(cands, key=lambda c: (c["est"][r], c["seq"]))

    for s, others in replay(run, tickets):
        if not others:
            continue
        n_choice += 1
        cands = [tickets[s], *others]
        picks = {r: pick(cands, r) for r in rules}
        trues = [c["true_rem"] for c in cands]
        best = min(trues) if all(x is not None for x in trues) else None
        for r, p in picks.items():
            if p is None:
                continue
            avail[r] += 1
            agree[r] += p["seq"] == s
            if picks["fcfs_query"] is not None:
                vs_fq[r] += p["seq"] != picks["fcfs_query"]["seq"]
            if picks["A_srpt_est"] is not None:
                vs_a[r] += p["seq"] != picks["A_srpt_est"]["seq"]
            if best is not None:
                regret[r].append(p["true_rem"] - best)
        known = [c for c in cands if c["true_rem"] is not None]
        for r in rules:
            for i in range(len(known)):
                for j in range(i + 1, len(known)):
                    x, y = known[i], known[j]
                    ex, ey = x["est"].get(r), y["est"].get(r)
                    if ex is None or ey is None:
                        continue
                    dt, de = x["true_rem"] - y["true_rem"], ex - ey
                    if abs(dt) < EPS or abs(de) < EPS:
                        continue
                    conc_n[r] += 1
                    conc[r] += (dt > 0) == (de > 0)
    rows = []
    for r in rules:
        rows.append(
            {
                "cond": cond, "run": label, "rule": r, "n_with_choice": n_choice,
                "n_available": avail[r],
                "agree_actual": agree[r] / avail[r] if avail[r] else None,
                "differ_vs_fcfs_query": vs_fq[r] / avail[r] if avail[r] else None,
                "differ_vs_A": vs_a[r] / avail[r] if avail[r] else None,
                "regret_mean": mean(regret[r]), "pair_concordance":
                conc[r] / conc_n[r] if conc_n[r] else None, "pairs": conc_n[r],
            }
        )  # fmt: skip
    return {"rows": rows, "grade_tickets_unobservable": n_unobs}


# ---------------------------------------------------------------- 3 simulation
def parse_speedup(s: str) -> dict[int, float]:
    out = {}
    for part in s.split(","):
        k, v = part.split(":")
        out[int(k)] = float(v)
    if out.get(1) != 1.0:
        raise SystemExit("--speedup 은 1:1 을 포함해야 함 (K=1 기준 상대 처리율)")
    return out


def speedup_from_calibration(path: str) -> dict[int, float]:
    rows = read_json(path).get("rows") or []
    key = "llm_calls_per_s" if all(r.get("llm_calls_per_s") for r in rows) else "throughput_qps"
    base = next((r[key] for r in rows if r["K"] == 1 and r.get(key)), None)
    if not base:
        raise SystemExit(f"calibration 에 K=1 처리율이 없음: {path}")
    return {int(r["K"]): float(r[key]) / base for r in rows if r.get(key)}


def make_g(speedup: dict[int, float] | None, model: str):
    if model == "independent":
        return lambda n: float(n)
    ks = sorted(speedup)

    def g(n: int) -> float:
        if n <= 0:
            return 0.0
        if n in speedup:
            return speedup[n]
        if n > ks[-1]:
            return speedup[ks[-1]]
        lo = max(k for k in ks if k < n)
        hi = min(k for k in ks if k > n)
        return speedup[lo] + (speedup[hi] - speedup[lo]) * (n - lo) / (hi - lo)

    return g


def simulate(requests: list[dict], tq: dict[str, TQ], policy: str, K: int, g, model,
             T: Tables) -> list[dict]:  # fmt: skip
    """K slot gate + 처리율 모델 g(n). key 는 호출 enqueue 때 한 번 계산 (실제 gate 와 같음)."""
    jobs = []
    for r in requests:
        q = tq[r["query_id"]]
        phases = [("delay", q.retrieval)]
        for j, c in enumerate(q.calls):
            phases.append(("call", j))
            if c["kind"] == "rewrite":
                phases.append(("delay", q.web))
        jobs.append({"req": r, "q": q, "phases": phases, "p": -1, "finish": None, "wait": 0.0})
    events = [(r["req"]["arrival"], n, "arrive", n) for n, r in enumerate(jobs)]
    heapq.heapify(events)
    order = len(jobs)
    waiting: list[tuple] = []
    active: dict[int, list] = {}  # ticket seq -> [remaining work, job index]
    seq = 0
    t = 0.0

    def key_of(jn: int, j: int) -> float:
        job, c = jobs[jn], jobs[jn]["q"].calls[j]
        q = job["q"]
        if policy == "fcfs":
            return 0.0
        if policy == "fcfs_query":
            return job["req"]["arrival"]
        if policy == "srpt_nopath":
            return nopath_est(model, q.qid, c["kind"], c["idx"], c["chars"], q.n_docs, q.feats)
        return point_estimate(q, j, model, T)[policy]

    def advance(jn: int):
        nonlocal seq, order
        job = jobs[jn]
        job["p"] += 1
        if job["p"] >= len(job["phases"]):
            job["finish"] = t
            return
        kind, v = job["phases"][job["p"]]
        if kind == "delay":
            order += 1
            heapq.heappush(events, (t + v, order, "delay", jn))
        else:
            heapq.heappush(waiting, (key_of(jn, v), seq, jn, v, t))
            seq += 1

    def dispatch():
        while waiting and len(active) < K:
            _, s, jn, j, t_enq = heapq.heappop(waiting)
            jobs[jn]["wait"] += t - t_enq
            active[s] = [jobs[jn]["q"].calls[j]["wall"], jn]

    while events or waiting or active:
        n = len(active)
        rate = g(n) / n if n else 0.0
        tc = t + min(v[0] for v in active.values()) / rate if n else math.inf
        te = events[0][0] if events else math.inf
        tn = min(tc, te)
        if tn == math.inf:
            raise RuntimeError("시뮬레이션 교착 (대기 호출이 있으나 실행 불가)")
        for v in active.values():
            v[0] -= rate * (tn - t)
        t = tn
        if tc <= te:
            for s in [s for s, v in active.items() if v[0] <= EPS]:
                jn = active.pop(s)[1]
                advance(jn)
        else:
            _, _, _, jn = heapq.heappop(events)
            advance(jn)
        dispatch()
    return [
        {"request_key": f"{j['req']['i']}:{j['req']['query_id']}", "query_id": j["q"].qid,
         "corrected": j["q"].corrected, "arrival": j["req"]["arrival"],
         "latency": j["finish"] - j["req"]["arrival"], "gate_wait": j["wait"]}
        for j in jobs
    ]  # fmt: skip


def lat_stats(xs: list[float]) -> dict:
    return {"mean": mean(xs), "p50": pct(xs, 50), "p95": pct(xs, 95)}


def run_sim(conds: dict, a, tq: dict, model, T: Tables, g) -> tuple[list, list, list, list]:
    summary, paired, req_rows, warns = [], [], [], []
    per_policy = defaultdict(dict)  # policy -> cond -> {request_key: row}
    for cond, live in conds.items():
        rho, s = cond[len("rho") :].split("_s")
        wl_path = Path(a.workloads) / f"{a.tag}_rho{rho}_s{s}.json"
        if not wl_path.exists():
            warns.append(f"workload 없음: {wl_path}")
            continue
        wl = json.loads(wl_path.read_text(encoding="utf-8"))
        reqs = [r for r in wl["requests"] if r["query_id"] in tq]
        if len(reqs) != len(wl["requests"]):
            warns.append(
                f"{cond}: eval trace 에 없는 query {len(wl['requests']) - len(reqs)}건 제외"
            )
        live_runs = {p: load_run(d) for p, d in live.items()} if live else {}
        for pol in SIM_POLICIES:
            res = simulate(reqs, tq, pol, a.k, g, model, T)
            per_policy[pol][cond] = {r["request_key"]: r for r in res}
            lat = [r["latency"] for r in res]
            row = {"cond": cond, "policy": pol, "n": len(res), **lat_stats(lat),
                   "corr_mean": mean([r["latency"] for r in res if r["corrected"]]),
                   "notcorr_mean": mean([r["latency"] for r in res if not r["corrected"]]),
                   "gate_wait_mean": mean([r["gate_wait"] for r in res])}  # fmt: skip
            lp = LIVE_FOR_SIM.get(pol)
            if lp and lp in live_runs:
                ll = [r["latency"] for r in live_runs[lp]["rows"].values()]
                row.update(live_mean=mean(ll), live_p95=pct(ll, 95))
                row["sim_live_mean_ratio"] = row["mean"] / row["live_mean"] if ll else None
            summary.append(row)
            for r in res:
                req_rows.append({"cond": cond, "policy": pol, **r})
    for base, target in SIM_PAIRS:
        strata_by = {}
        for cond in per_policy[base]:
            a_, b_ = per_policy[base][cond], per_policy[target][cond]
            keys = sorted(a_, key=lambda k: a_[k]["arrival"])
            strata_by[cond] = (np.array([a_[k]["latency"] for k in keys]),
                               np.array([b_[k]["latency"] for k in keys]))  # fmt: skip
        groups = {c: [c] for c in strata_by}
        if len(strata_by) > 1:
            groups["all_pooled"] = list(strata_by)
        for gname, cs in groups.items():
            r = boot([strata_by[c] for c in cs], a.n_boot, 0, "iid", 20)
            paired.append(
                {
                    "base": base,
                    "target": target,
                    "group": gname,
                    **{
                        k: r[k]
                        for k in (
                            "n_pairs",
                            "mean_diff",
                            "mean_lo",
                            "mean_hi",
                            "p95_diff",
                            "p95_lo",
                            "p95_hi",
                            "mean_rel",
                        )
                    },
                }
            )
    return summary, paired, req_rows, warns


# ---------------------------------------------------------------- driver
def leakage_checks(train: list[TQ], eval_q: list[TQ], model) -> tuple[dict, list[str]]:
    tr, ev = {q.qid for q in train}, {q.qid for q in eval_q}
    model_ids = set(model.train_query_ids or [])
    info = {
        "train_queries": len(tr), "eval_queries": len(ev), "train_eval_overlap": len(tr & ev),
        "cost_model_train_ids": len(model_ids), "cost_model_eval_overlap": len(model_ids & ev),
        "cost_model_ids_not_in_train_trace": len(model_ids - tr),
    }  # fmt: skip
    errs = []
    if tr & ev:
        errs.append(f"train trace 와 eval 질문이 {len(tr & ev)}개 겹침")
    if model_ids & ev:
        errs.append(f"cost model 학습 질문과 eval 질문이 {len(model_ids & ev)}개 겹침")
    return info, errs


def thresholds(run: dict, a) -> tuple[float, float]:
    th = (run["meta"].get("comparability") or {}).get("thresholds") or {}
    return th.get("upper", a.upper), th.get("lower", a.lower)


def report_md(res: dict, a) -> str:
    L = [f"# Grading 중간 정보 ablation ({a.name})", ""]
    L += ["## 0. 데이터·누수 점검", ""]
    L += [f"- {k}: {v}" for k, v in res["leakage"].items()]
    L += [f"- train grading: {json.dumps(res['grading_check']['train'], ensure_ascii=False)}"]
    L += [f"- eval grading: {json.dumps(res['grading_check']['eval'], ensure_ascii=False)}"]
    L += [f"- thresholds upper/lower: {res['thresholds']}"]
    L += [f"- {w}" for w in res["warnings"]]
    L += ["", "## 1. 점추정 (eval sequential trace, 초)", ""]
    pr = [r for r in res["point"] if r["stage"] in ("all", "grade_all", "grade0", "grade1",
                                                     "grade2", "rewrite", "generate")]  # fmt: skip
    L += md_table(pr, ["stage", "path", "estimator", "n", "est_mean", "true_mean", "bias", "mae",
                       "spearman"])  # fmt: skip
    L += ["", "- 전체 단계별 표는 point_estimates.csv. bias = 추정 - 실제", ""]
    L += ["## 2. live 대기열 (같은 대기열의 즉시 선택만, E2E 개선 아님)", ""]
    L += md_table(res["queue"], ["cond", "run", "rule", "n_with_choice", "n_available",
                                 "agree_actual", "differ_vs_fcfs_query", "differ_vs_A",
                                 "regret_mean", "pair_concordance", "pairs"])  # fmt: skip
    L += [f"- {k}: 관측 불가 grade ticket {v}" for k, v in res["queue_unobservable"].items()]
    L += ["", "## 3. 오프라인 스케줄링 시뮬레이션 (모델 값)", ""]
    if res["sim"]:
        L += [
            f"- GPU 처리 모델: {a.rate_model}, g(n): {json.dumps(res['speedup'])}, K={a.k}",
            "- 먼저 live_* 와 비교해 모델이 live 의 정책 간 순서를 재현하는지 확인할 것",
            "",
        ]
        L += md_table(res["sim"], ["cond", "policy", "n", "mean", "p50", "p95", "corr_mean",
                                   "notcorr_mean", "gate_wait_mean", "live_mean", "live_p95",
                                   "sim_live_mean_ratio"])  # fmt: skip
        L += ["", "### 시뮬레이션 paired 차이 (target - base, iid bootstrap 95% CI)",
              ""]  # fmt: skip
        L += md_table(res["sim_paired"], ["base", "target", "group", "n_pairs", "mean_diff",
                                          "mean_lo", "mean_hi", "mean_rel", "p95_diff", "p95_lo",
                                          "p95_hi"])  # fmt: skip
    else:
        L += [
            "- 실행 안 함 (--workloads 와 --calibration/--speedup 필요,"
            " 또는 rate-model independent)"
        ]
    L += ["", "## 학습한 표 (train trace)", "", "```json",
          json.dumps(res["tables"], indent=1, ensure_ascii=False)[:6000], "```", ""]  # fmt: skip
    return "\n".join(L)


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train split sequential run (학습 전용)")
    ap.add_argument("--seq", required=True, help="eval split sequential run (정답·시뮬레이션 환경)")
    ap.add_argument("--cost-model", required=True)
    ap.add_argument("--live-root", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rhos", nargs="+", default=["0.8", "0.95"])
    ap.add_argument("--seeds", nargs="+", default=["0", "1"])
    ap.add_argument("--queue-policies", nargs="+", default=["fcfs_query", "srpt_est"])
    ap.add_argument("--workloads", default=None, help="<dir>/<tag>_rho<r>_s<s>.json")
    ap.add_argument("--tag", default="SV1")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--speedup", default=None, help="예: 1:1,2:1.7,4:2.6 (calibration 대신)")
    ap.add_argument("--rate-model", choices=["ps", "independent"], default="ps")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--upper", type=float, default=0.7)
    ap.add_argument("--lower", type=float, default=0.3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--name", default="grade_info_ablation")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    train_run, eval_run = load_run(a.train), load_run(a.seq)
    train, eval_q = trace_queries(train_run), trace_queries(eval_run)
    model = cost_model_from_dict(json.loads(Path(a.cost_model).read_text(encoding="utf-8")))
    leak, errs = leakage_checks(train, eval_q, model)
    if errs:
        raise SystemExit("정보 누수: " + "; ".join(errs))
    upper, lower = thresholds(train_run, a)
    if thresholds(eval_run, a) != (upper, lower):
        raise SystemExit("train / eval threshold 가 다름")
    T = Tables(train, upper, lower, a.min_cell)
    gc = {"train": grading_check(train_run), "eval": grading_check(eval_run)}
    warns = []
    for k, v in gc.items():
        if v["grading_strategy"] != ["full"]:
            warns.append(
                f"{k}: grading 전략 {v['grading_strategy']} (순차 full 이 아니면 C 해석 불가)"
            )
        if v["not_sequential"] or v["score_count_mismatch"]:
            warns.append(f"{k}: grade 순서/점수 수 불일치 {v}")
    res = {"leakage": leak, "grading_check": gc, "thresholds": [upper, lower],
           "tables": T.to_dict(), "warnings": warns}  # fmt: skip
    res["point"] = point_rows(eval_q, model, T)

    conds = {}
    res["queue"], res["queue_unobservable"] = [], {}
    if a.live_root:
        conds, missing = discover(a.live_root, a.rhos, a.seeds, a.k,
                                  ["fcfs", "fcfs_query", "srpt_nopath", "srpt_est"])  # fmt: skip
        warns += [f"없는 run: {m}" for m in missing]
        seq = load_seq(a.seq)
        for cond, runs in conds.items():
            for pol in a.queue_policies:
                if pol in runs:
                    qm = queue_metrics(cond, pol, load_run(runs[pol]), model, T, seq)
                    res["queue"] += qm["rows"]
                    res["queue_unobservable"][f"{cond}:{pol}"] = qm["grade_tickets_unobservable"]
    elif a.workloads:
        conds = {f"rho{r}_s{s}": {} for r in a.rhos for s in a.seeds}

    res["sim"], res["sim_paired"], sim_req, res["speedup"] = [], [], [], None
    if a.workloads and (a.calibration or a.speedup or a.rate_model == "independent"):
        sp = None
        if a.rate_model == "ps":
            sp = parse_speedup(a.speedup) if a.speedup else speedup_from_calibration(a.calibration)
        res["speedup"] = sp
        tq = {q.qid: q for q in eval_q}
        res["sim"], res["sim_paired"], sim_req, w = run_sim(
            conds, a, tq, model, T, make_g(sp, a.rate_model)
        )
        warns += w
    elif a.workloads:
        warns.append("시뮬레이션 생략: --calibration 또는 --speedup 필요 (rate-model ps)")

    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    res["created"] = datetime.now().isoformat()
    res["args"] = vars(a)
    (out / "analysis.json").write_text(
        json.dumps({k: v for k, v in res.items() if k != "point"}, indent=2, ensure_ascii=False,
                   default=str), encoding="utf-8")  # fmt: skip
    write_csv(out / "point_estimates.csv", res["point"])
    write_csv(out / "queue_metrics.csv", res["queue"])
    write_csv(out / "sim_summary.csv", res["sim"])
    write_csv(out / "sim_paired.csv", res["sim_paired"])
    write_csv(out / "sim_requests.csv", sim_req)
    md = report_md(res, a)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not a.quiet:
        print(md)
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
