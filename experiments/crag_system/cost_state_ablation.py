"""C-RAG 실행 상태 기반 잔여 LLM 작업량 예측 ablation (읽기 전용, 추가 추론·API 호출 없음).

질문: 실행 단계·grading 결과·evidence 정보를 쓰면 기존 SRPT-est 보다 요청의 남은 LLM 작업량을
더 정확히 예측하는가. 예측 정확도만 본다.
실제 GPU 지연 개선은 별개의 결과이며 여기서 주장하지 않는다.
기존 정책·cost model·결과 파일은 바꾸지 않는다 (결과는 새 폴더에만).

    uv run python experiments/crag_system/cost_state_ablation.py \
        --train results/crag_system/pre/PRE-SV1-TRAIN/train \
        --seq results/crag_system/pre/PRE-SV1-A/eval_seq \
        --cost-model results/crag_system/cost_models/SV1_train_wall.json \
        --evc-seq results/crag_system/pre/PRE-EVC-SV1-SEQ/<ts> \
        [--live-root results/crag_system/pre/PRE-EVC-SV1-LIVE] \
        [--live-root results/crag_system/pre/PRE-SV1-LIVE] --name SV1-cost-state

모델 (모두 "호출이 gate 에 들어가는 시점" 의 남은 LLM 작업량
      = 이 호출 포함 이후 호출의 무부하 시간 합)
  M0   기존 SRPT-est. cost_model.est_remaining 그대로 (실제 정책 클래스 SRPTEst.base 와 대조)
  M1   Stage-aware. train trace 의 E[남은 작업 | 호출 종류, grade 순번].
       질문 내용·점수·evidence 미사용
       (grade_info_ablation B_stage 와 같은 표)
  M2s  State-aware. M0 의 grading 항 + 이미 관측된 grading 점수로 P(action) 을 갱신하고
       action 별 train 평균 post 작업량을 섞음. action 확정 뒤 (rewrite) 에는 그 action 의 평균
  M2   State + Evidence-aware. M2s 와 같은 P(action) 에, action x evidence 정책별 generation prompt
       길이 기대값 (FAISS 문서 길이는 retrieval 직후 관측, 미관측 문서는 train 선택 확률,
       web 문서는 train 평균 길이) 을 넣어 generation 비용을 M0 의 gen 계수로 계산
  generate 호출 시점에는 실제 prompt 길이가 관측되므로 M0 = M2s = M2 (gen_coef @ chars).
  이 시점은 초기 시점과 반드시 따로 보고한다.

정보 누수 방지
  - 모든 추정은 Obs (그 시점에 관측 가능한 값만 담은 객체) 만 받는다. grade 호출에서는
    enqueue 전에 끝난 grade 호출의 점수만 담고, action 은 grade 단계에서 None.
  - 학습 (M0 cost model, M1/M2 표) 은 train trace 만. eval 질문과 겹치면 중단.
  - EVC / live 데이터는 정답 (무부하 시간) 과 평가 대상으로만 쓰고 학습에 쓰지 않는다.
  - evidence 정책 (crag_rule / keep_all ...) 은 시스템 설정이므로 관측 가능한 값으로 본다.

평가 집합
  seq_crag_rule  : eval sequential run 전체 (정답 = seq 호출 wall [관측])
  evc_<정책>     : EVC 측정이 네 정책 모두 있는 질문. grade / rewrite = seq wall [관측],
                   generate = 그 정책 context 의 EVC 측정 wall [측정] (crag_rule 포함 같은 조건)
  live (선택)    : live run 의 gate 대기열 복원. 같은 대기열 안의 우선순위 결정 차이만 본다
                   (정답 = diag_evc_live.call_noload, 관측·측정이 아닌 것은 정답에서 제외)

지표: MAE, RMSE, Bias, Spearman, p50/p95 |오차|, 쌍별 우선순위 역전율 (정답 대비),
      M0 대비 우선순위 변화율, query 단위 paired bootstrap 95% CI (ΔMAE, ΔBias vs M0)
결과: results/crag_system/analysis/cost_state/<name>/<timestamp>/ (덮어쓰지 않음)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence_filter_ablation as efa  # noqa: E402
from diag_evc_live import call_noload, fit_noload, load_evc, load_seq_rows  # noqa: E402
from diag_sched import mean, spearman  # noqa: E402
from diag_sv_policy import load_run, md_table, qfeats, replay, write_csv  # noqa: E402
from evidence_cost import LIVE_DIR, _cfg_policy  # noqa: E402
from grade_info_ablation import Tables, grade_scores, score_bucket  # noqa: E402

from rag.pipelines import prompts  # noqa: E402
from rag.scheduling.cost_model import GRADE_KINDS, cost_model_from_dict  # noqa: E402
from rag.scheduling.policies import CallTicket, SRPTEst  # noqa: E402
from rag.scheduling.state import QueryState  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0", "M1", "M2s", "M2")
POLICIES = ("crag_rule", "no_correct_filter", "no_incorrect_discard", "keep_all")
ACTIONS = efa.ACTIONS
T0_GEN = len(prompts.GENERATE.format(context="", question=""))
DOC_OVH = 6  # 기대값 계산용 문서당 형식 문자 수 ("[k] " + 구분 "\n\n"). 정확 계산은 ctx_chars
EPS = 1e-9


# ================================================================ prompt 길이 (문자 수) 산술
def ctx_chars(lengths: list[int], n_web: int = 0, web_total: float = 0.0) -> float:
    """prompts.format_context 결과 길이. FAISS 문서 길이 목록 + web 문서 n_web 개 (합 web_total)."""
    n = len(lengths) + n_web
    if n == 0:
        return 0.0
    return sum(len(f"[{i + 1}] ") for i in range(n)) + 2 * (n - 1) + sum(lengths) + web_total


def gen_chars(q_len: int, lengths: list[int], n_web: int = 0, web_total: float = 0.0) -> float:
    return T0_GEN + q_len + ctx_chars(lengths, n_web, web_total)


def faiss_kept(policy: str, action: str, scores: list, upper: float, lower: float) -> list[int]:
    """정책의 generation context 에 남는 FAISS 문서 index (efa.build_context 와 같은 규칙)."""
    rule = efa.CONDITIONS[policy]
    n = len(scores)
    if rule[action]:
        return list(range(n))
    if action == "incorrect":
        return []
    cut = upper if action == "correct" else lower
    return [i for i, s in enumerate(scores) if s is not None and s >= cut]


# ================================================================ trajectory (무부하 호출 열)
@dataclass
class Traj:
    key: str
    qid: str
    policy: str
    action: str
    scores: list
    feats: np.ndarray
    n_docs: int
    rchars: list
    rscores: list
    q_len: int
    s_pop: float | None
    calls: list  # [{kind, idx, chars, wall, t_enq, t_end, src}] 실행 순서
    web_n: int = 0
    web_total: float | None = None  # 이 질문 web 문서 문자 합 (seq gen prompt 에서 역산)
    gen_chars_check: dict = field(default_factory=dict)

    def remaining(self, j: int) -> float:
        return float(sum(c["wall"] for c in self.calls[j:]))


def web_total_from_row(row: dict, gen_prompt_chars: int, upper: float, lower: float):
    """crag_rule run 의 generate prompt 길이에서 web 문서 문자 합을 역산. correct 면 0."""
    ids = list(row.get("retrieval_doc_ids") or [])
    rch = list(row.get("retrieval_chars") or [])
    ctx = list(row.get("contexts") or [])
    web_ids = list((row.get("extra") or {}).get("web_doc_ids") or [])
    pos = {d: i for i, d in enumerate(ids)}
    kept = [rch[pos[d]] for d in ctx if d in pos and pos[d] < len(rch)]
    n_web = len(ctx) - len(kept)
    if n_web != len(web_ids) and row.get("corrective_action") != "correct":
        return None, None
    q_len = len(row.get("question") or "")
    base = gen_chars(q_len, kept, n_web, 0.0)
    return n_web, float(gen_prompt_chars - base)


def trajectories(run: dict, upper: float, lower: float) -> list[Traj]:
    """crag_rule run (train / eval seq) -> Traj. 호출 wall 은 [관측]."""
    out = []
    for key, row in run["rows"].items():
        cs = run["calls"].get(key, [])
        if not cs or row.get("question") is None:
            continue
        feats, n_docs = qfeats(row)
        calls = [{"kind": c["kind"], "idx": c["idx"], "chars": c["prompt_chars"],
                  "wall": float(c["wall"]), "t_enq": c.get("t_enqueue", c["t_start"]),
                  "t_end": c["t_end"], "src": "관측"} for c in cs]  # fmt: skip
        gen = [c for c in calls if c["kind"] == "generate"]
        t = Traj(key=key, qid=row["query_id"], policy="crag_rule",
                 action=row.get("corrective_action"), scores=grade_scores(row), feats=feats,
                 n_docs=n_docs, rchars=list(row.get("retrieval_chars") or []),
                 rscores=_rscores(row),
                 q_len=len(row["question"]), s_pop=row.get("s_pop"), calls=calls)  # fmt: skip
        if gen:
            n_web, w = web_total_from_row(row, gen[-1]["chars"], upper, lower)
            if n_web is not None:
                t.web_n, t.web_total = n_web, w
                if t.action == "correct":
                    t.gen_chars_check = {"exact": abs(w) < 0.5}
        out.append(t)
    return out


def policy_traj(t: Traj, policy: str, gen_wall: float, upper: float, lower: float) -> Traj | None:
    """crag_rule eval trajectory -> 같은 grading·web 으로 evidence 정책만 바꾼 trajectory.
    generate 호출의 prompt 길이 = 그 정책 context 의 길이 (산술), wall = EVC 측정."""
    if t.web_total is None or t.action not in ACTIONS:
        return None
    kept = [t.rchars[i] for i in faiss_kept(policy, t.action, t.scores, upper, lower)]
    web = t.action != "correct"
    chars = gen_chars(t.q_len, kept, t.web_n if web else 0, t.web_total if web else 0.0)
    calls = [dict(c) for c in t.calls]
    for c in calls:
        if c["kind"] == "generate":
            c.update(chars=int(round(chars)), wall=float(gen_wall), src="측정")
    return Traj(**{**t.__dict__, "policy": policy, "calls": calls})


# ================================================================ 관측 가능한 값
@dataclass(frozen=True)
class Obs:
    """한 결정 시점에 관측 가능한 값만. 추정 함수는 이 객체만 받는다."""

    qid: str
    policy: str  # evidence 정책 (시스템 설정)
    kind: str
    idx: int
    chars: int  # 현재 호출 prompt 길이 (호출 시점에 관측)
    n_docs: int
    feats: tuple
    rchars: tuple  # FAISS 문서 길이 (retrieval 직후)
    rscores: tuple  # FAISS 검색 점수 (retrieval 직후, feats 의 원천)
    q_len: int
    s_pop: float | None
    seen: tuple  # ((doc idx, score), ...) enqueue 전에 끝난 grade 호출
    action: str | None  # grade 단계에서는 None

    def __post_init__(self):
        if self.kind in GRADE_KINDS and self.action is not None:
            raise ValueError("grade 단계에서 action 은 관측 불가 (미래 정보)")
        if self.kind in GRADE_KINDS and any(i == self.idx for i, _ in self.seen):
            raise ValueError("현재 grade 호출의 점수는 관측 불가")


def obs_at(t: Traj, j: int, seen_idx: set[int] | None = None) -> Obs:
    """trajectory t 의 j 번째 호출 시점 Obs. grade 호출이면 enqueue 전에 끝난 grade 의 점수만."""
    c = t.calls[j]
    grade = c["kind"] in GRADE_KINDS
    if grade:
        if seen_idx is None:
            seen_idx = {
                p["idx"] for p in t.calls[:j]
                if p["kind"] in GRADE_KINDS and p["t_end"] <= c["t_enq"] + 1e-6
            }  # fmt: skip
        seen = tuple((i, t.scores[i]) for i in sorted(seen_idx) if i < len(t.scores))
    else:
        seen = tuple(enumerate(t.scores))
    return Obs(qid=t.qid, policy=t.policy, kind=c["kind"], idx=c["idx"], chars=c["chars"],
               n_docs=t.n_docs, feats=tuple(float(x) for x in t.feats), rchars=tuple(t.rchars),
               rscores=tuple(t.rscores),
               q_len=t.q_len, s_pop=t.s_pop, seen=seen,
               action=None if grade else t.action)  # fmt: skip


def stage_of(kind: str, idx: int) -> str:
    if kind in GRADE_KINDS:
        return "grade0" if idx == 0 else "grade_mid"
    return kind


# ================================================================ 학습 (train trace 만)
class StateTables:
    """M2s / M2 용 표. train (crag_rule sequential) trace 만으로 만든다."""

    def __init__(self, train: list[Traj], upper: float, lower: float, min_cell: int = 10):
        self.upper, self.lower, self.min_cell = upper, lower, min_cell
        self.cells: dict = defaultdict(Counter)
        self.global_a = Counter(t.action for t in train if t.action in ACTIONS)
        for t in train:
            if t.action not in ACTIONS:
                continue
            for k in range(len(t.scores)):
                pre = t.scores[:k]
                b = score_bucket(pre, upper, lower, True)
                self.cells[(k, b)][t.action] += 1
                self.cells[("any", b)][t.action] += 1
        post = defaultdict(list)
        gen = defaultdict(list)
        for t in train:
            post[t.action].append(sum(c["wall"] for c in t.calls if c["kind"] not in GRADE_KINDS))
            gen[t.action] += [c["wall"] for c in t.calls if c["kind"] == "generate"]
        allpost = [x for v in post.values() for x in v]
        self.post_mean = {a: float(np.mean(post[a])) if post[a] else float(np.mean(allpost or [0]))
                          for a in ACTIONS}  # fmt: skip
        allgen = [x for v in gen.values() for x in v]
        self.gen_mean = {a: float(np.mean(gen[a])) if gen[a] else float(np.mean(allgen or [0]))
                         for a in ACTIONS}  # fmt: skip
        # 문서 i 가 action a 의 필터 (correct: >= upper, ambiguous: >= lower) 를 통과할 확률
        sel = defaultdict(lambda: [0, 0])
        for t in train:
            if t.action not in ("correct", "ambiguous"):
                continue
            cut = upper if t.action == "correct" else lower
            for i, s in enumerate(t.scores):
                for key in ((t.action, i), (t.action, "any")):
                    sel[key][0] += s is not None and s >= cut
                    sel[key][1] += 1
        self.sel = {k: v[0] / v[1] for k, v in sel.items() if v[1]}
        self.sel_n = {f"{k[0]}:{k[1]}": v[1] for k, v in sel.items()}
        webs = [(t.web_n, t.web_total) for t in train
                if t.action in ("ambiguous", "incorrect") and t.web_total is not None
                and t.web_total >= 0]  # fmt: skip
        self.web_n = float(np.mean([n for n, _ in webs])) if webs else 0.0
        self.web_total = float(np.mean([w for _, w in webs])) if webs else 0.0
        self.n_web_rows = len(webs)
        gch = [c["chars"] for t in train for c in t.calls if c["kind"] == "generate"]
        self.gen_chars_range = (min(gch), max(gch)) if gch else (None, None)

    def p_action(self, obs: Obs) -> tuple[dict, str]:
        pre = [s for _, s in obs.seen]
        b = score_bucket(pre, self.upper, self.lower, True)
        if b == "hit":  # full grading 규칙: upper 이상이 하나라도 있으면 correct 확정
            return {"correct": 1.0, "ambiguous": 0.0, "incorrect": 0.0}, "hit"
        for key in ((len(pre), b), ("any", b)):
            c = self.cells.get(key)
            if c and sum(c.values()) >= self.min_cell:
                n = sum(c.values())
                return {a: c[a] / n for a in ACTIONS}, f"{key[0]}:{b}"
        n = sum(self.global_a.values()) or 1
        return {a: self.global_a[a] / n for a in ACTIONS}, "global"

    def p_sel(self, action: str, i: int) -> float:
        return self.sel.get((action, i), self.sel.get((action, "any"), 0.0))

    def exp_gen_chars(self, obs: Obs, action: str) -> float:
        """action 이 a 일 때 정책 obs.policy 의 generation prompt 길이 기대값."""
        rule = efa.CONDITIONS[obs.policy]
        seen = dict(obs.seen)
        cut = self.upper if action == "correct" else self.lower
        known_all = len(seen) >= obs.n_docs
        if rule[action]:
            w = [1.0] * obs.n_docs
        elif action == "incorrect":
            w = [0.0] * obs.n_docs
        else:
            w = [(1.0 if (seen[i] is not None and seen[i] >= cut) else 0.0) if i in seen
                 else self.p_sel(action, i) for i in range(obs.n_docs)]  # fmt: skip
        lens = list(obs.rchars[: obs.n_docs])
        web = action != "correct"
        if known_all and all(x in (0.0, 1.0) for x in w):  # FAISS 부분은 정확히 계산
            kept = [L for L, x in zip(lens, w, strict=False) if x == 1.0]
            n_web = int(round(self.web_n)) if web else 0
            return gen_chars(obs.q_len, kept, n_web, self.web_total if web else 0.0)
        faiss = sum(x * (L + DOC_OVH) for x, L in zip(w, lens, strict=False))
        return T0_GEN + obs.q_len + faiss + ((self.web_total + DOC_OVH * self.web_n) if web else 0)

    def to_dict(self) -> dict:
        return {"cells": {f"{k[0]}:{k[1]}": dict(v) for k, v in self.cells.items()},
                "global_action": dict(self.global_a), "post_mean": self.post_mean,
                "gen_mean": self.gen_mean, "p_sel": {f"{k[0]}:{k[1]}": v
                                                     for k, v in self.sel.items()},
                "p_sel_n": self.sel_n, "web_n_mean": self.web_n,
                "web_total_chars_mean": self.web_total, "n_web_rows": self.n_web_rows,
                "train_gen_chars_range": self.gen_chars_range,
                "min_cell": self.min_cell}  # fmt: skip


@dataclass
class Bundle:
    model: object  # CostModel | OOFCostModel (기존, 바꾸지 않음)
    stage: Tables  # M1
    st: StateTables  # M2s / M2


def estimate(b: Bundle, o: Obs) -> dict[str, float]:
    m = b.model.for_query(o.qid)
    feats = np.array(o.feats)
    m0 = m.est_remaining(o.kind, o.idx, o.chars, o.n_docs, feats)
    out = {"M0": m0, "M1": b.stage.stage_est(o.kind, o.idx)}
    if o.kind == "generate":
        out["M2s"] = out["M2"] = m.call_est("generate", o.chars)
        return out
    if o.kind == "rewrite":
        a = o.action
        out["M2s"] = m.rewrite_mean + b.st.gen_mean[a]
        out["M2"] = m.rewrite_mean + m.call_est("generate", b.st.exp_gen_chars(o, a))
        return out
    grade_work = m0 - m.pred_post(feats) if o.kind == "grade" else 0.0
    p, _ = b.st.p_action(o)
    out["M2s"] = grade_work + sum(p[a] * b.st.post_mean[a] for a in ACTIONS)
    out["M2"] = grade_work + sum(
        p[a] * ((m.rewrite_mean if a != "correct" else 0.0)
                + m.call_est("generate", b.st.exp_gen_chars(o, a)))
        for a in ACTIONS if p[a] > 0)  # fmt: skip
    return out


def m0_policy_class(model, o: Obs) -> float:
    """실제 정책 클래스 (rag.scheduling.policies.SRPTEst) 로 계산한 M0 (재현 식과 대조용)."""
    s = QueryState(o.qid, s_pop=o.s_pop)
    s.retrieval_scores, s.retrieval_chars, s.n_docs = list(o.rscores), list(o.rchars), o.n_docs
    return SRPTEst(model).base(CallTicket(s, o.kind, o.idx, o.chars, 0.0, 0))


def _rscores(row: dict) -> list:
    return [s if s is not None else 0.0 for s in (row.get("extra") or {}).get(
        "retrieval_scores") or []]  # fmt: skip


# ================================================================ 지표
def pair_stats(
    est: np.ndarray,
    tru: np.ndarray,
    ref: np.ndarray,
    groups: np.ndarray,
    cap: int = 4000,
    seed: int = 0,
) -> tuple[float | None, float | None, int]:
    """쌍별 (다른 query) 역전율 (정답 대비) 과 M0 대비 순서 변화율. n > cap 이면 무작위 cap 개."""
    n = len(est)
    if n < 2:
        return None, None, 0
    idx = np.arange(n)
    if n > cap:
        idx = np.random.default_rng(seed).choice(n, cap, replace=False)
    e, t, r, g = est[idx], tru[idx], ref[idx], groups[idx]
    de = e[:, None] - e[None, :]
    dt = t[:, None] - t[None, :]
    dr = r[:, None] - r[None, :]
    iu = np.triu_indices(len(idx), 1)
    de, dt, dr = de[iu], dt[iu], dr[iu]
    diffq = (g[:, None] != g[None, :])[iu]
    m = diffq & (np.abs(dt) > EPS)
    inv = None
    if m.any():
        s = np.sign(de[m]) * np.sign(dt[m])
        inv = float(np.mean(np.where(s == 0, 0.5, s < 0)))
    mr = diffq & (np.abs(dr) > EPS)
    chg = float(np.mean(np.sign(de[mr]) != np.sign(dr[mr]))) if mr.any() else None
    return inv, chg, int(m.sum())


def boot_delta(err_m: np.ndarray, err_0: np.ndarray, groups: np.ndarray, n_boot: int,
               seed: int = 0) -> dict:  # fmt: skip
    """query 단위 paired bootstrap: ΔMAE, ΔBias (모델 - M0)."""
    qs, inv = np.unique(groups, return_inverse=True)
    if len(qs) < 2:
        return {}
    k = len(qs)
    s_abs = np.bincount(inv, np.abs(err_m) - np.abs(err_0), minlength=k)
    s_sig = np.bincount(inv, err_m - err_0, minlength=k)
    cnt = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(k, np.full(k, 1 / k), size=n_boot).astype(float)
    tot = w @ cnt
    d_mae = (w @ s_abs) / tot
    d_bias = (w @ s_sig) / tot
    return {"d_mae": float(s_abs.sum() / cnt.sum()),
            "d_mae_lo": float(np.percentile(d_mae, 2.5)),
            "d_mae_hi": float(np.percentile(d_mae, 97.5)),
            "d_bias": float(s_sig.sum() / cnt.sum()),
            "d_bias_lo": float(np.percentile(d_bias, 2.5)),
            "d_bias_hi": float(np.percentile(d_bias, 97.5))}  # fmt: skip


def metric_rows(points: list[dict], set_name: str, n_boot: int) -> list[dict]:
    """points: {qid, policy, stage, action, truth, est{model}} -> 그룹별 지표 행."""
    groups = defaultdict(list)
    for p in points:
        st = p["stage"]
        for sg in {st, "all", "pre_generate" if st != "generate" else None,
                   "grading" if st.startswith("grade") else None} - {None}:  # fmt: skip
            for ag in (p["action"], "all"):
                groups[(p["policy"], sg, ag)].append(p)
    out = []
    for (pol, sg, ag), ps in sorted(groups.items()):
        tru = np.array([p["truth"] for p in ps])
        g = np.array([p["qid"] for p in ps])
        m0 = np.array([p["est"]["M0"] for p in ps])
        for mname in MODELS:
            est = np.array([p["est"][mname] for p in ps])
            err = est - tru
            ae = np.abs(err)
            inv, chg, npairs = (pair_stats(est, tru, m0, g) if ag == "all"
                                else (None, None, 0))  # fmt: skip
            row = {"set": set_name, "policy": pol, "stage": sg, "action": ag, "model": mname,
                   "n_points": len(ps), "n_queries": len(set(g.tolist())),
                   "true_mean": float(tru.mean()), "est_mean": float(est.mean()),
                   "bias": float(err.mean()), "mae": float(ae.mean()),
                   "rmse": float(np.sqrt((err ** 2).mean())), "abs_p50": float(np.median(ae)),
                   "abs_p95": float(np.percentile(ae, 95)), "spearman": spearman(est, tru),
                   "pair_inversion": inv, "pair_change_vs_M0": chg if mname != "M0" else 0.0,
                   "pairs": npairs}  # fmt: skip
            if mname != "M0" and ag == "all" and n_boot:
                row.update(boot_delta(err, m0 - tru, g, n_boot))
            out.append(row)
    return out


def points_for(trajs: list[Traj], b: Bundle, check_m0: list) -> list[dict]:
    pts = []
    for t in trajs:
        for j, c in enumerate(t.calls):
            o = obs_at(t, j)
            e = estimate(b, o)
            if check_m0 is not None:
                check_m0.append(abs(e["M0"] - m0_policy_class(b.model, o)))
            pts.append({"qid": t.qid, "key": t.key, "policy": t.policy,
                        "stage": stage_of(c["kind"], c["idx"]), "kind": c["kind"],
                        "idx": c["idx"], "action": t.action, "truth": t.remaining(j),
                        "n_seen": len(o.seen), "est": e})  # fmt: skip
    return pts


# ================================================================ live 대기열 (결정 차이만)
def live_runs(roots: list[str]) -> dict[str, tuple[Path, str]]:
    """live root 들 -> {label: (dir, evidence policy)}. 폴더 이름이 evidence 정책이 아니면
    (SV1 의 scheduler 이름) crag_rule."""
    out = {}
    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            continue
        for d in sorted(p for p in rp.iterdir() if p.is_dir()):
            m = LIVE_DIR.match(d.name)
            if not m or not (d / "queries.jsonl").exists():
                continue
            pol = _cfg_policy(d) or (m["policy"] if m["policy"] in efa.CONDITIONS else "crag_rule")
            out[f"{rp.name}/{d.name}"] = (d, pol)
    return out


def live_tickets(run: dict, pol: str, b: Bundle, ctx: dict) -> dict[int, dict]:
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
        feats, n_docs = qfeats(row)
        scores = grade_scores(row)
        grade = e["kind"] in GRADE_KINDS
        seen = (tuple((c["idx"], scores[c["idx"]]) for c in cs if c["kind"] in GRADE_KINDS
                      and c["idx"] != e["idx"] and c["t_end"] <= enq + 1e-6
                      and c["idx"] < len(scores)) if grade
                else tuple(enumerate(scores)))  # fmt: skip
        o = Obs(qid=row["query_id"], policy=pol, kind=e["kind"], idx=e["idx"],
                chars=call["prompt_chars"], n_docs=n_docs, feats=tuple(float(x) for x in feats),
                rchars=tuple(row.get("retrieval_chars") or []), rscores=tuple(_rscores(row)),
                q_len=len(row["question"]), s_pop=row.get("s_pop"), seen=tuple(sorted(seen)),
                action=None if grade else row.get("corrective_action"))  # fmt: skip
        later = [c for c in cs if c["t_start"] >= call["t_start"] - 1e-9]
        tru, exact = 0.0, True
        for c in later:
            v, s = call_noload(c, row, pol, ctx["seq_calls"], ctx["seq_rows"], ctx["evc"],
                               ctx["noload"])  # fmt: skip
            exact &= s in ("관측", "측정")
            tru += v or 0.0
        tickets[e["seq"]] = {
            "seq": e["seq"], "rk": key, "enq": enq, "grant": e["t"], "kind": e["kind"],
            "arrival": row["arrival"], "logged_key": e.get("key"), "est": estimate(b, o),
            "true_rem": tru if exact else None,
        }  # fmt: skip
    return tickets


def live_decisions(label: str, run: dict, pol: str, tickets: dict) -> dict:
    sched = run["policy"]
    rules = (*MODELS, "fcfs_query")
    for t in tickets.values():
        t["est"]["fcfs_query"] = t["arrival"]
    n = 0
    top_vs_m0, top_vs_act, pair_chg, pair_n = Counter(), Counter(), Counter(), Counter()
    conc, conc_n, regret, rho_m0 = Counter(), Counter(), defaultdict(list), defaultdict(list)
    for s, others in replay(run, tickets):
        if not others:
            continue
        n += 1
        cands = [tickets[s], *others]
        pick = {r: min(cands, key=lambda c, r=r: (c["est"][r], c["seq"]))["seq"] for r in rules}
        trues = [c["true_rem"] for c in cands]
        best = min(trues) if all(x is not None for x in trues) else None
        for r in rules:
            top_vs_m0[r] += pick[r] != pick["M0"]
            top_vs_act[r] += pick[r] != s
            if best is not None:
                regret[r].append(tickets[pick[r]]["true_rem"] - best)
            if len(cands) >= 3:
                rho_m0[r].append(spearman([c["est"][r] for c in cands],
                                          [c["est"]["M0"] for c in cands]))  # fmt: skip
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    x, y = cands[i], cands[j]
                    dm = x["est"]["M0"] - y["est"]["M0"]
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
    out = []
    for r in rules:
        out.append({"run": label, "evidence_policy": pol, "scheduler": sched, "rule": r,
                    "n_decisions": n, "top_differs_vs_M0": top_vs_m0[r] / n if n else None,
                    "top_differs_vs_actual": top_vs_act[r] / n if n else None,
                    "pair_order_change_vs_M0": pair_chg[r] / pair_n[r] if pair_n[r] else None,
                    "spearman_vs_M0_mean": mean([x for x in rho_m0[r] if x is not None]),
                    "truth_pair_concordance": conc[r] / conc_n[r] if conc_n[r] else None,
                    "truth_pairs": conc_n[r], "regret_mean": mean(regret[r]),
                    "regret_n": len(regret[r])})  # fmt: skip
    exact = sum(t["true_rem"] is not None for t in tickets.values())
    kc = None
    if sched == "srpt_est":
        d = [abs(t["logged_key"][1] - t["est"]["M0"]) for t in tickets.values()
             if t.get("logged_key")]  # fmt: skip
        kc = {"checked": len(d), "max_abs_diff": max(d) if d else None,
              "n_mismatch_gt_1e-6": sum(x > 1e-6 for x in d)}  # fmt: skip
    return {"rows": out, "n_tickets": len(tickets), "exact_truth_share":
            exact / len(tickets) if tickets else None, "m0_logged_key_check": kc}  # fmt: skip


# ================================================================ main analysis
def leakage(train: list[Traj], evals: list[Traj], model) -> tuple[dict, list[str]]:
    tr, ev = {t.qid for t in train}, {t.qid for t in evals}
    mids = set(model.train_query_ids or [])
    info = {"train_queries": len(tr), "eval_queries": len(ev), "train_eval_overlap": len(tr & ev),
            "cost_model_train_ids": len(mids), "cost_model_eval_overlap": len(mids & ev),
            "cost_model_ids_not_in_train_trace": len(mids - tr),
            "train_trace_ids_not_in_cost_model": len(tr - mids) if mids else None}  # fmt: skip
    errs = []
    if tr & ev:
        errs.append(f"train trace 와 eval 질문이 {len(tr & ev)}개 겹침")
    if mids & ev:
        errs.append(f"cost model 학습 질문과 eval 질문이 {len(mids & ev)}개 겹침")
    return info, errs


def thresholds_of(run: dict) -> tuple[float, float]:
    th = (run["meta"].get("comparability") or {}).get("thresholds") or {}
    return float(th.get("upper", 0.7)), float(th.get("lower", 0.3))


def analyze(a) -> dict:
    warns = []
    train_run, seq_run = load_run(a.train), load_run(a.seq)
    upper, lower = thresholds_of(seq_run)
    if thresholds_of(train_run) != (upper, lower):
        warns.append(f"train / eval threshold 다름: {thresholds_of(train_run)} vs {(upper, lower)}")
    for name, run in (("train", train_run), ("seq", seq_run)):
        if run["policy"] not in (None, "none"):
            warns.append(f"{name} run 에 scheduler {run['policy']} (sequential 기대)")
    model = cost_model_from_dict(json.loads(Path(a.cost_model).read_text(encoding="utf-8")))
    train = trajectories(train_run, upper, lower)
    seq = trajectories(seq_run, upper, lower)
    info, errs = leakage(train, seq, model)
    if errs and not a.allow_overlap:
        raise SystemExit("정보 누수: " + "; ".join(errs))
    warns += errs
    b = Bundle(model, Tables(_as_tq(train), upper, lower, min_cell=a.min_cell),
               StateTables(train, upper, lower, min_cell=a.min_cell))  # fmt: skip
    m0chk: list = []
    pts_seq = points_for(seq, b, m0chk)
    # web 역산 검증: correct 질문은 web 없음 -> 산술 prompt 길이가 실제와 같아야 함
    chk = [t.gen_chars_check.get("exact") for t in seq if t.gen_chars_check]
    chars_check = {"correct_rows_checked": len(chk), "exact_share": mean(chk),
                   "corrected_rows_web_total_ok": sum(1 for t in seq if t.action != "correct"
                                                      and t.web_total is not None
                                                      and t.web_total >= 0)}  # fmt: skip
    # evc: 네 정책 모두 측정이 있는 질문 (generate wall = 그 정책 context 측정)
    evc = load_evc(Path(a.evc_seq)) if a.evc_seq else {}
    pts_evc, evc_info = [], {}
    if evc:
        by_q = {t.qid: t for t in seq}
        ok = [q for q, t in by_q.items() if all((q, p) in evc for p in a.policies)]
        evc_info = {"questions_with_all_policies": len(ok),
                    "seq_questions": len(by_q)}  # fmt: skip
        trajs = []
        for q in ok:
            for p in a.policies:
                pt = policy_traj(by_q[q], p, evc[(q, p)]["wall"], upper, lower)
                if pt is not None:
                    trajs.append(pt)
        pts_evc = points_for(trajs, b, None)
        ex = [t for t in trajs if t.policy == "crag_rule"]
        evc_info["crag_rule_gen_seq_vs_evc_wall"] = {
            "seq_mean": mean([c["wall"] for t in ex for c in by_q[t.qid].calls
                              if c["kind"] == "generate"]),
            "evc_mean": mean([c["wall"] for t in ex for c in t.calls if c["kind"] == "generate"]),
        }  # fmt: skip
        hi = b.st.gen_chars_range[1]
        evc_info["gen_chars_beyond_train_max"] = {
            p: mean([c["chars"] > hi for t in trajs if t.policy == p for c in t.calls
                     if c["kind"] == "generate"]) for p in a.policies} if hi else None  # fmt: skip
    rows = metric_rows(pts_seq, "seq_crag_rule", a.n_boot)
    rows += metric_rows(pts_evc, "evc", a.n_boot)
    # live
    live, live_info, chars_live = [], {}, []
    if a.live_root:
        seq_rows, seq_calls, _ = load_seq_rows(Path(a.seq))
        ctx = {"seq_rows": seq_rows, "seq_calls": seq_calls, "evc": evc,
               "noload": fit_noload(seq_calls, evc)}  # fmt: skip
        by_q = {t.qid: t for t in seq}
        for label, (d, pol) in live_runs(a.live_root).items():
            run = load_run(d)
            tk = live_tickets(run, pol, b, ctx)
            res = live_decisions(label, run, pol, tk)
            live += res["rows"]
            live_info[label] = {k: v for k, v in res.items() if k != "rows"}
            # 산술 prompt 길이 검증: live generate prompt_chars vs 같은 질문·action 의 재구성
            for key, row in run["rows"].items():
                t = by_q.get(row["query_id"])
                gen = [c for c in run["calls"].get(key, []) if c["kind"] == "generate"]
                if t is None or not gen or row.get("corrective_action") != t.action:
                    continue
                pt = policy_traj(t, pol, 0.0, upper, lower)
                if pt is not None:
                    pc = next(c["chars"] for c in pt.calls if c["kind"] == "generate")
                    chars_live.append(
                        {
                            "run": label,
                            "policy": pol,
                            "same": pc == gen[-1]["prompt_chars"],
                            "web_hit": (row.get("web_replay") or {}).get("hit"),
                        }
                    )  # noqa: E501
    chars_live_sum = [
        {"run": r, "n": len(v), "exact_share": mean([x["same"] for x in v])}
        for r, v in sorted(_group(chars_live, "run").items())
    ]
    return {
        "inputs": {"train": a.train, "seq": a.seq, "cost_model": a.cost_model,
                   "evc_seq": a.evc_seq, "live_root": a.live_root, "policies": a.policies,
                   "upper": upper, "lower": lower, "min_cell": a.min_cell, "n_boot": a.n_boot},
        "warnings": warns, "leakage": info,
        "m0_vs_policy_class": {"checked": len(m0chk), "max_abs_diff": max(m0chk) if m0chk
                               else None},
        "chars_check_seq": chars_check, "chars_check_live": chars_live_sum, "evc": evc_info,
        "tables": {"M1_stage": b.stage.to_dict(), "M2": b.st.to_dict()},
        "metrics": rows, "live": live, "live_info": live_info,
        "points": pts_seq + pts_evc,
    }  # fmt: skip


def _group(rows: list[dict], k: str) -> dict:
    out = defaultdict(list)
    for r in rows:
        out[r[k]].append(r)
    return out


def _as_tq(trajs: list[Traj]):
    """M1 (grade_info_ablation.Tables) 입력 형식."""
    from grade_info_ablation import TQ

    return [TQ(key=t.key, qid=t.qid, corrected=t.action != "correct", feats=t.feats,
               n_docs=t.n_docs, scores=t.scores,
               calls=[{"kind": c["kind"], "idx": c["idx"], "chars": c["chars"],
                       "wall": c["wall"]} for c in t.calls], retrieval=0.0, web=0.0)
            for t in trajs]  # fmt: skip


# ================================================================ report
def report_md(res: dict, name: str) -> str:
    M = res["metrics"]
    L = [f"# 상태 기반 잔여 LLM 작업량 예측 ablation ({name})", "",
         "예측 정확도 분석이다. 실제 GPU 요청 지연 개선은 별개이며 여기서 검증하지 않는다.", "",
         "```", json.dumps(res["inputs"], ensure_ascii=False, indent=1), "```", ""]  # fmt: skip
    if res["warnings"]:
        L += ["## 경고", ""] + [f"- {w}" for w in res["warnings"]] + [""]
    L += ["## 데이터 분리·누수 검사", "", "```", json.dumps(res["leakage"], indent=1), "```", "",
          f"- M0 (재현 식) vs 실제 정책 클래스 SRPTEst.base: "
          f"{json.dumps(res['m0_vs_policy_class'])}",
          f"- generation prompt 길이 산술 검증 (seq correct 질문, web 없음): "
          f"{json.dumps(res['chars_check_seq'], ensure_ascii=False)}",
          f"- EVC 평가 집합: {json.dumps(res['evc'], ensure_ascii=False)}", ""]  # fmt: skip
    if res["chars_check_live"]:
        L += ["- live generate prompt 길이 = 재구성 길이 (같은 질문·같은 action) 비율:", ""]
        L += md_table(res["chars_check_live"], ["run", "n", "exact_share"]) + [""]
    cols = ["policy", "stage", "model", "n_points", "n_queries", "true_mean", "est_mean", "bias",
            "mae", "rmse", "abs_p50", "abs_p95", "spearman", "pair_inversion",
            "pair_change_vs_M0", "d_mae", "d_mae_lo", "d_mae_hi"]  # fmt: skip
    for set_name, title in (
        ("seq_crag_rule", "seq (crag_rule, 정답 = seq wall [관측])"),
        ("evc", "evidence 정책별 (grade / rewrite [관측] + generate [측정])"),
    ):
        rs = [r for r in M if r["set"] == set_name and r["action"] == "all"]
        if not rs:
            continue
        L += [f"## {title}", "", "### 시점별 (action 전체)", ""]
        order = {"grade0": 0, "grade_mid": 1, "grading": 2, "rewrite": 3, "pre_generate": 4,
                 "generate": 5, "all": 6}  # fmt: skip
        rs.sort(key=lambda r: (r["policy"], order.get(r["stage"], 9), MODELS.index(r["model"])))
        L += md_table(rs, cols) + [""]
        ra = [r for r in M if r["set"] == set_name and r["action"] != "all"
              and r["stage"] in ("grade0", "pre_generate")]  # fmt: skip
        ra.sort(key=lambda r: (r["policy"], r["stage"], r["action"], MODELS.index(r["model"])))
        L += ["### action 별 (최종 action 으로 나눔, 추정에는 쓰지 않음)", ""]
        L += md_table(ra, ["policy", "stage", "action", "model", "n_points", "bias", "mae",
                           "spearman"]) + [""]  # fmt: skip
    pb = [r for r in M if r["set"] == "evc" and r["action"] == "all"
          and r["stage"] in ("grade0", "pre_generate", "generate")]  # fmt: skip
    if pb:
        L += ["### evidence 정책별 Bias 요약", ""]
        pb.sort(key=lambda r: (r["stage"], r["model"], r["policy"]))
        L += md_table(pb, ["stage", "model", "policy", "bias", "mae", "spearman"]) + [""]
    if res["live"]:
        L += ["## live 대기열 결정 차이 (지연 개선 아님)", "",
              "같은 시각 gate 에 함께 대기한 ticket 사이에서 각 모델이 고를 호출. "
              "실제 dispatch 는 run 의 scheduler 결과이며, 순서를 바꿨을 때의 GPU 지연은 "
              "이 로그로 알 수 없다.", ""]  # fmt: skip
        L += md_table(
            res["live"],
            [
                "run",
                "evidence_policy",
                "scheduler",
                "rule",
                "n_decisions",
                "top_differs_vs_M0",
                "top_differs_vs_actual",
                "pair_order_change_vs_M0",
                "spearman_vs_M0_mean",
                "truth_pair_concordance",
                "truth_pairs",
                "regret_mean",
            ],
        )
        L += ["", "```", json.dumps(res["live_info"], ensure_ascii=False, indent=1), "```", ""]
    L += ["## 해석 시 주의", "",
          "- 학습은 crag_rule sequential train trace 만. keep_all 등의 generation 비용은 "
          "M0 의 gen 계수 (prompt 문자 수 선형) 를 train 범위 밖으로 외삽한다 "
          "(gen_chars_beyond_train_max 참고)",
          "- evc 집합의 generate 정답은 prefix cache 초기화 측정, grade / rewrite 는 seq 기록 "
          "(prefix cache 켜짐). 두 조건이 섞여 있다",
          "- web 문서 길이는 미관측 -> train 평균. replay miss (empty) 는 반영하지 않음",
          "- 예측 오차 감소 ≠ 스케줄링 개선. 쌍별 역전율·결정 차이는 순서 변화의 가능성만 보여준다",
          ""]  # fmt: skip
    return "\n".join(L)


def new_out_dir(root: Path, name: str) -> Path:
    base = root / "cost_state" / name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out, n = base, 1
    while out.exists():
        n += 1
        out = base.with_name(f"{base.name}_{n}")
    out.mkdir(parents=True, exist_ok=False)
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="cost model 학습 sequential trace (train split)")
    ap.add_argument("--seq", required=True, help="eval sequential run (crag_rule)")
    ap.add_argument("--cost-model", required=True, help="기존 SRPT-est cost model json (M0)")
    ap.add_argument("--evc-seq", default=None, help="evidence_cost measure 결과 폴더")
    ap.add_argument("--live-root", action="append", default=[])
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-overlap", action="store_true", help="(테스트용) 누수 검사 실패 무시")
    ap.add_argument("--name", default="cost-state")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    bad = [p for p in a.policies if p not in efa.CONDITIONS or p == "vanilla_ctx"]
    if bad:
        raise SystemExit(f"지원하지 않는 정책: {bad}")
    res = analyze(a)
    out = new_out_dir(Path(a.out_root), a.name)
    (out / "analysis.json").write_text(json.dumps(
        {k: v for k, v in res.items() if k != "points"}, indent=2, ensure_ascii=False,
        default=str), encoding="utf-8")  # fmt: skip
    write_csv(out / "metrics.csv", res["metrics"])
    write_csv(out / "live_decisions.csv", res["live"])
    write_csv(out / "points.csv", [{**{k: v for k, v in p.items() if k != "est"},
                                    **{f"est_{m}": p["est"][m] for m in MODELS}}
                                   for p in res["points"]])  # fmt: skip
    md = report_md(res, a.name)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not a.quiet:
        print(md)
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
