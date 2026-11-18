"""상태 기반 SRPT-est (M2) live 정책 — 실험 전용 (기존 정책·cost model·CRAGGraph 는 바꾸지 않음).

cost_state_ablation.py 의 M2 (State + Evidence-aware 잔여 LLM 작업량 예측) 를 live gate 의
우선순위로 쓴다. Scheduler 구조 (SchedulingGate, K slots, 4-tuple key, 실행 중 호출 비선점) 는
기존 srpt_est 와 같고 base (= 호출이 gate 에 들어갈 때 한 번 계산하는 남은 작업량 추정치) 만 바뀐다.

    # 1) M2 표 학습 (train sequential trace 만. 기존 cost model 파일은 읽기만)
    uv run python experiments/crag_system/state_sched.py fit \
        --train results/crag_system/pre/PRE-SV1-TRAIN/train \
        --cost-model results/crag_system/cost_models/SV1_train_wall.json \
        --out results/crag_system/cost_models/SV1_train_state_m2.json
    # 2) live 는 run_state_sched.py (run_state_sched.sh live)

정책 (scheduler.policy)
  srpt_est    기존 M0 와 값이 같다 (rag.scheduling.policies.SRPTEst.base 그대로 반환).
              scheduler.state_shadow=true (기본) 면 같은 시점의 M2 값을 그림자로 계산해 기록만 한다.
  srpt_state  M2. 같은 시점의 M0 도 계산해 기록한다 (우선순위 차이 분석용).

추정치 갱신 시점 (= gate enqueue, SchedulingGate.acquire 안에서 한 번. 대기 중에는 바뀌지 않음)
  grade i     관측: 검색 점수·문서 길이 (retrieval 직후), 질문 길이 (도착 시), 현재 prompt 길이,
              grade 0..i-1 의 점수 (이 호출 enqueue 전에 끝난 호출만). action = 미관측
              -> M0 의 grading 항 + P(action | 관측 점수 prefix) x action 별 post 작업량 (M2)
  rewrite     관측: 모든 grade 점수 -> action 확정 (decide_action, 그래프와 같은 규칙)
              -> rewrite 평균 + 그 action·evidence 정책의 generation prompt 길이 기대값으로 gen 비용
  generate    실제 prompt 길이 관측 -> M0 = M2 (gen_coef @ chars)
  미래 action·최종 점수·현재 호출의 점수는 Obs (cost_state_ablation.Obs) 검사로 막는다.

관측 경로: QueryState / PolicyView 에는 문서별 grading 점수가 없다 (코어 인터페이스는 바꾸지 않음).
  GradeTap 이 grader LLM 을 감싸 grade 호출 응답을 grade_one 과 같은 파서 (parse_score) 로 읽어
  ScoreBoard 에 (request_key, idx, score) 로 남긴다. 호출이 끝난 뒤에만 기록하므로 그 요청의 다음
  호출 enqueue 때만 보인다. 파이프라인 결과 (호출 수·점수·action·contexts) 에는 영향이 없다.
  full sequential grading 에서만 허용 (호출 순서 = 문서 순서 = recorder idx).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cost_state_ablation as csa  # noqa: E402
from diag_sv_policy import load_run  # noqa: E402

from rag.grading.base import decide_action, parse_score  # noqa: E402
from rag.scheduling.cost_model import GRADE_KINDS, cost_model_from_dict  # noqa: E402
from rag.scheduling.policies import SRPTEst, _feats, _model_meta  # noqa: E402
from rag.telemetry.calls import infer_call_kind  # noqa: E402

ACTIONS = csa.ACTIONS
STATE_POLICIES = ("srpt_est", "srpt_state")
FORMAT = "state_m2/v1"


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ================================================================ M2 표 (직렬화)
def tables_to_dict(st: csa.StateTables) -> dict:
    return {
        "upper": st.upper, "lower": st.lower, "min_cell": st.min_cell,
        "cells": [[k, b, dict(c)] for (k, b), c in st.cells.items()],
        "global_action": dict(st.global_a), "post_mean": st.post_mean, "gen_mean": st.gen_mean,
        "sel": [[a, i, v] for (a, i), v in st.sel.items()], "web_n": st.web_n,
        "web_total": st.web_total, "n_web_rows": st.n_web_rows,
        "gen_chars_range": list(st.gen_chars_range),
    }  # fmt: skip


def tables_from_dict(d: dict) -> csa.StateTables:
    """학습 없이 저장된 표로 StateTables 를 복원 (p_action / exp_gen_chars 는 원래 메서드 사용)."""
    st = object.__new__(csa.StateTables)
    st.upper, st.lower, st.min_cell = float(d["upper"]), float(d["lower"]), int(d["min_cell"])
    st.cells = {(k, b): Counter(c) for k, b, c in d["cells"]}
    st.global_a = Counter(d["global_action"])
    st.post_mean, st.gen_mean = dict(d["post_mean"]), dict(d["gen_mean"])
    st.sel = {(a, i): float(v) for a, i, v in d["sel"]}
    st.sel_n = {}
    st.web_n, st.web_total = float(d["web_n"]), float(d["web_total"])
    st.n_web_rows = d.get("n_web_rows")
    st.gen_chars_range = tuple(d.get("gen_chars_range") or (None, None))
    return st


def fit_state_model(train_dir: str, cost_model_path: str, min_cell: int = 10) -> dict:
    run = load_run(train_dir)
    upper, lower = csa.thresholds_of(run)
    trajs = csa.trajectories(run, upper, lower)
    if not trajs:
        raise SystemExit(f"train trace 가 비어 있음: {train_dir}")
    st = csa.StateTables(trajs, upper, lower, min_cell=min_cell)
    model = cost_model_from_dict(json.loads(Path(cost_model_path).read_text(encoding="utf-8")))
    tr = sorted({t.qid for t in trajs})
    mids = set(model.train_query_ids or [])
    return {
        "format": FORMAT,
        "tables": tables_to_dict(st),
        "meta": {
            "train_dir": str(train_dir), "train_query_ids": tr, "n_train_queries": len(tr),
            "train_actions": dict(Counter(t.action for t in trajs)),
            "cost_model_path": str(cost_model_path), "cost_model_sha256": file_sha(cost_model_path),
            "cost_model_train_ids_not_in_state_train": len(mids - set(tr)) if mids else None,
            "state_train_ids_not_in_cost_model": len(set(tr) - mids) if mids else None,
            "min_cell": min_cell, "created": datetime.now().isoformat(timespec="seconds"),
            "source": "cost_state_ablation.StateTables (M2)",
        },
    }  # fmt: skip


def load_state_model(path: str | Path) -> tuple[csa.StateTables, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("format") != FORMAT:
        raise ValueError(f"state model 형식이 아님 ({d.get('format')!r} != {FORMAT}): {path}")
    meta = {**d["meta"], "path": str(path), "sha256": file_sha(path)}
    return tables_from_dict(d["tables"]), meta


# ================================================================ 관측 (live)
class ScoreBoard:
    """request_key -> {q_len, scores {grade idx: score}}. grade 호출이 끝난 뒤에만 기록된다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._d: dict[str, dict] = {}

    def start(self, request_key: str, q_len: int) -> None:
        with self._lock:
            self._d[request_key] = {"q_len": int(q_len), "scores": {}}

    def add(self, request_key: str, idx: int, score: float) -> None:
        with self._lock:
            e = self._d.setdefault(request_key, {"q_len": None, "scores": {}})
            if idx in e["scores"]:
                raise ValueError(f"{request_key}: grade idx {idx} 중복 기록")
            e["scores"][idx] = float(score)

    def get(self, request_key: str) -> tuple[int | None, dict[int, float]]:
        with self._lock:
            e = self._d.get(request_key)
            return (None, {}) if e is None else (e["q_len"], dict(e["scores"]))

    def items(self) -> dict:
        with self._lock:
            return {
                k: {"q_len": v["q_len"], "scores": dict(v["scores"])} for k, v in self._d.items()
            }


class GradeTap:
    """grader LLM 래퍼: 응답은 그대로 돌려주고 grade 호출 점수만 ScoreBoard 에 남긴다."""

    def __init__(self, inner, board: ScoreBoard, request_key: str):
        self.inner, self.board, self.request_key = inner, board, request_key
        self._n = 0
        self._lock = threading.Lock()

    def generate(self, prompt: str, **kwargs) -> str:
        kind = kwargs.get("kind") or infer_call_kind(prompt)
        if kind != "grade":
            return self.inner.generate(prompt, **kwargs)
        with self._lock:
            i, self._n = self._n, self._n + 1
        out = self.inner.generate(prompt, **kwargs)
        self.board.add(self.request_key, i, parse_score(out))  # grade_one 과 같은 파서
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


def observe(t, board: ScoreBoard, evidence_policy: str, upper: float, lower: float) -> csa.Obs:
    """CallTicket -> 그 enqueue 시점에 관측 가능한 값만 담은 Obs."""
    q = t.query
    q_len, scores = board.get(q.request_key)
    if q_len is None:
        raise RuntimeError(f"{q.request_key}: ScoreBoard 에 질문 길이가 없음 (GradeTap 미연결)")
    grade = t.kind in GRADE_KINDS
    if grade:
        seen = tuple(sorted((i, s) for i, s in scores.items() if i < t.idx))
        action = None
    else:
        seen = tuple(sorted(scores.items()))
        action = decide_action([s for _, s in seen], upper, lower)
    return csa.Obs(qid=q.query_id, policy=evidence_policy, kind=t.kind, idx=t.idx,
                   chars=t.prompt_chars, n_docs=q.n_docs,
                   feats=tuple(float(x) for x in _feats(q)),
                   rchars=tuple(q.retrieval_chars), rscores=tuple(q.retrieval_scores),
                   q_len=q_len, s_pop=q.s_pop, seen=seen, action=action)  # fmt: skip


def m2_estimate(m, st: csa.StateTables, o: csa.Obs) -> tuple[float, dict]:
    """cost_state_ablation.estimate 의 M2 와 같은 식 (테스트로 대조). (값, 기록용 정보)."""
    feats = np.array(o.feats)
    if o.kind == "generate":
        return m.call_est("generate", o.chars), {"basis": "gen_chars"}
    if o.kind == "rewrite":
        gc = st.exp_gen_chars(o, o.action)
        return m.rewrite_mean + m.call_est("generate", gc), {"basis": "action", "gen_chars": gc}
    if o.kind not in GRADE_KINDS:
        return m.est_remaining(o.kind, o.idx, o.chars, o.n_docs, feats), {"basis": "m0"}
    m0 = m.est_remaining(o.kind, o.idx, o.chars, o.n_docs, feats)
    grade_work = m0 - m.pred_post(feats) if o.kind == "grade" else 0.0
    p, cell = st.p_action(o)
    v = grade_work + sum(
        p[a] * ((m.rewrite_mean if a != "correct" else 0.0)
                + m.call_est("generate", st.exp_gen_chars(o, a)))
        for a in ACTIONS if p[a] > 0)  # fmt: skip
    return v, {"basis": "p_action", "cell": cell, "p": {a: round(p[a], 4) for a in ACTIONS}}


# ================================================================ 정책
class StateSRPT(SRPTEst):
    """use="M2" -> srpt_state, use="M0" -> srpt_est (값은 SRPTEst.base 그대로, M2 는 그림자 기록).

    gate lock 안에서 enqueue 때 한 번 호출된다 (dynamic=False: dispatch 때 다시 계산하지 않음).
    """

    def __init__(self, model, tables: csa.StateTables, board: ScoreBoard, evidence_policy: str,
                 use: str = "M2", shadow: bool = True, state_meta: dict | None = None):  # fmt: skip
        if use not in ("M0", "M2"):
            raise ValueError("use: M0 | M2")
        super().__init__(model)
        self.tables, self.board, self.evidence_policy = tables, board, evidence_policy
        self.use, self.shadow = use, shadow or use == "M2"
        self.name = "srpt_state" if use == "M2" else "srpt_est"
        self.state_meta = state_meta or {}
        self.log: list[dict] = []

    def base(self, t):
        c0 = time.perf_counter()
        m0 = SRPTEst.base(self, t)
        c1 = time.perf_counter()
        m2 = info = o = None
        if self.shadow:
            o = observe(t, self.board, self.evidence_policy, self.tables.upper, self.tables.lower)
            m2, info = m2_estimate(self.model.for_query(o.qid), self.tables, o)
        c2 = time.perf_counter()
        q = t.query
        self.log.append({
            "seq": t.seq, "request_key": q.request_key, "query_id": q.query_id, "kind": t.kind,
            "idx": t.idx, "arrival": q.arrival, "m0": m0, "m2": m2, "used": self.use,
            "n_seen": None if o is None else len(o.seen),
            "action_obs": None if o is None else o.action, "info": info,
            "t_m0_us": (c1 - c0) * 1e6, "t_m2_us": (c2 - c1) * 1e6 if self.shadow else None,
            "t_used_us": ((c1 - c0) if self.use == "M0" else (c2 - c0)) * 1e6,
        })  # fmt: skip
        return m2 if self.use == "M2" else m0

    def describe(self):
        return {"name": self.name, "cost_model": _model_meta(self.model),
                "estimator": self.use, "shadow_other": self.shadow,
                "state_model": {k: v for k, v in self.state_meta.items()
                                if k != "train_query_ids"},
                "evidence_policy": self.evidence_policy,
                "update_points": "gate enqueue (grade i / rewrite / generate)",
                "key": "static (dynamic=False)"}  # fmt: skip


def overhead_summary(log: list[dict], total_llm_wall: float | None = None) -> dict:
    def stats(xs):
        if not xs:
            return {"n": 0}
        a = np.array(xs, float)
        return {"n": len(a), "mean_us": float(a.mean()), "p50_us": float(np.median(a)),
                "p95_us": float(np.percentile(a, 95)), "p99_us": float(np.percentile(a, 99)),
                "max_us": float(a.max()), "total_s": float(a.sum() / 1e6)}  # fmt: skip

    out = {"used": stats([e["t_used_us"] for e in log]),
           "m0": stats([e["t_m0_us"] for e in log]),
           "m2": stats([e["t_m2_us"] for e in log if e["t_m2_us"] is not None]),
           "by_kind_used": {k: stats([e["t_used_us"] for e in log if e["kind"] == k])
                            for k in sorted({e["kind"] for e in log})}}  # fmt: skip
    if total_llm_wall:
        out["used_total_over_llm_wall"] = out["used"].get("total_s", 0.0) / total_llm_wall
    out["note"] = (
        "정책 base() 계산 시간 (gate lock 안, enqueue 당 1회). key() 는 정적 tuple 이라 "
        "따로 재지 않음. 그림자 계산 시간은 used 에 넣지 않았으나 같은 lock 안에서 쓴다"
    )
    return out


# ================================================================ check (GPU 실행 전)
def check(a) -> int:
    """기준 fcfs_query run 과 같은 조건인지, 누수·덮어쓰기가 없는지 확인. 0 = 통과."""
    from equal_load import run_id as eq_run_id
    from workload import load_workload

    bad, info = [], []
    st, meta = load_state_model(a.state_model)
    if meta["cost_model_sha256"] != file_sha(a.cost_model):
        bad.append(f"state model 의 cost model sha 가 {a.cost_model} 과 다름")
    train = set(meta.get("train_query_ids") or [])
    for wp in a.workload:
        wl = load_workload(wp)
        rid = eq_run_id(wl, wl["equal_load"]["policy"], a.k)
        qids = {r["query_id"] for r in wl["requests"]}
        if train & qids:
            bad.append(f"{wp}: state model 학습 질문과 {len(train & qids)}개 겹침")
        base = Path(a.base_root) / rid
        if not (base / "summary.json").exists():
            bad.append(f"기준 fcfs_query run 없음: {base}")
        else:
            bm = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
            sched = ((bm.get("scheduler") or {}).get("policy") or {}).get("name")
            slots = (bm.get("scheduler") or {}).get("slots")
            bsha = (bm.get("workload") or {}).get("sha256")
            if sched != "fcfs_query":
                bad.append(f"{base}: scheduler={sched} (fcfs_query 기대)")
            if slots != a.k:
                bad.append(f"{base}: K={slots} (기대 {a.k})")
            if bsha != wl["sha256"]:
                bad.append(f"{base}: workload sha 다름 (도착 trace 가 다름)")
            if wl["equal_load"]["policy"] != "keep_all":
                bad.append(f"{wp}: evidence policy {wl['equal_load']['policy']} (keep_all 기대)")
            info.append(
                f"기준 OK: {base.name} (sha {bsha[:12]}, git "
                f"{(bm.get('git') or {}).get('commit', '?')[:10]})"
            )
        for root in a.new_root:
            d = Path(root) / rid
            if d.exists() and not (d / "summary.json").exists():
                bad.append(f"중단된 run 폴더 (덮어쓰지 않음, 이름을 바꿀 것): {d}")
            elif d.exists():
                info.append(f"완료됨 (live 에서 건너뜀): {d}")
    info.append(
        f"state model: n_train={meta['n_train_queries']} min_cell={meta['min_cell']} "
        f"threshold=({st.upper}, {st.lower})"
    )
    for x in info:
        print("  " + x)
    for x in bad:
        print("FAIL " + x)
    print("check: " + ("OK" if not bad else f"{len(bad)} 문제"))
    return 1 if bad else 0


# ================================================================ CLI
def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="train trace -> M2 표 (json). 기존 파일은 덮어쓰지 않음")
    f.add_argument("--train", required=True)
    f.add_argument("--cost-model", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--min-cell", type=int, default=10)
    c = sub.add_parser("check", help="live 전 점검 (기준 run·workload·누수·폴더 충돌)")
    c.add_argument("--state-model", required=True)
    c.add_argument("--cost-model", required=True)
    c.add_argument("--base-root", required=True)
    c.add_argument("--new-root", action="append", default=[])
    c.add_argument("--workload", action="append", required=True)
    c.add_argument("--k", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        raise SystemExit(check(a))
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"이미 있음 (덮어쓰지 않음): {out}")
    d = fit_state_model(a.train, a.cost_model, a.min_cell)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    m = d["meta"]
    print(json.dumps({k: v for k, v in m.items() if k != "train_query_ids"}, ensure_ascii=False,
                     indent=1))  # fmt: skip
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
