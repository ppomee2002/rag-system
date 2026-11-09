"""query / 호출 단위 작업량 예측 (SRPT-est 용).

Source: exp/correction-scheduling:experiments/correction_scheduling/scheduling.py
(`features`, `CostModel`, `fit_cost_model`, `attach_oof_predictions`, `job_from_row`) 를 이식했다.
수식은 원본과 같고, 다음만 추가했다.
  - service_key: 호출의 서비스 시간으로 쓸 필드. Ollama trace 는 `server_total` (원본과 같음),
    vLLM trace 는 요청 단위 서버 시간이 없으므로 부하 없는 sequential run 의 `wall` 을 쓴다.
  - grade_batch 호출 (Batch / Adaptive grading) 의 비용 추정.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FEATURE_NAMES = ["intercept", "top1", "mean", "gap", "chars", "log_pop"]


def features(scores: list[float], chars: list[int], s_pop) -> np.ndarray:
    """retrieval 직후 (grading 전) 에 알 수 있는 값만 사용 (원본과 동일)."""
    s = sorted(scores, reverse=True) or [0.0]
    gap = s[0] - s[1] if len(s) > 1 else 0.0
    return np.array(
        [1.0, s[0], float(np.mean(s)), gap, sum(chars) / 1e4, math.log10((s_pop or 0) + 1)]
    )


@dataclass
class Call:
    kind: str
    idx: int
    chars: int
    service: float


@dataclass
class Job:
    qid: str
    action: str
    feats: np.ndarray
    phases: list  # ("delay", sec) | ("gpu", [Call, ...])
    work: float = 0.0
    solo: float = 0.0
    pred_total: float = 0.0
    pred_post: float = 0.0
    n_docs: int = 0
    acc: float = 0.0


GRADE_KINDS = ("grade", "grade_batch")


def job_from_row(row: dict, service_key: str = "server_total") -> Job:
    """trace 행 (query 결과 + calls) -> Job. 같은 grading phase 의 호출은 시작 순서대로 나열."""
    calls = sorted(row["calls"], key=lambda c: c["t_start"])

    def mk(c):
        service = c.get(service_key)
        if service is None:
            raise ValueError(f"call 에 {service_key} 가 없음 (vLLM trace 면 service_key='wall')")
        return Call(c["kind"], c["idx"], c["prompt_chars"], float(service))

    st = row["stage_latencies"]
    phases: list = [("delay", st.get("retrieval", 0.0))]
    grades = [mk(c) for c in calls if c["kind"] in GRADE_KINDS]
    phases += [("gpu", [g]) for g in grades]
    rewrites = [mk(c) for c in calls if c["kind"] == "rewrite"]
    if rewrites:
        phases.append(("gpu", rewrites[:1]))
        phases.append(("delay", st.get("web_search", 0.0)))
    phases.append(("gpu", [mk(c) for c in calls if c["kind"] == "generate"]))
    work = sum(c.service for k, cs in phases if k == "gpu" for c in cs)
    solo = sum(v if k == "delay" else sum(c.service for c in v) for k, v in phases)
    return Job(
        qid=row["query_id"],
        action=row["corrective_action"],
        feats=features(row["extra"]["retrieval_scores"], row["retrieval_chars"], row.get("s_pop")),
        phases=phases,
        work=work,
        solo=solo,
        n_docs=len(row["extra"]["retrieval_scores"]),
        acc=row.get("acc", 0.0),
    )


def post_work(job: Job) -> float:
    """grading 이후 작업량 (rewrite + generation)."""
    return sum(
        c.service for k, cs in job.phases if k == "gpu" for c in cs if c.kind not in GRADE_KINDS
    )


def _lstsq(X, y, ridge=1e-3):
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


@dataclass
class CostModel:
    grade_coef: np.ndarray  # service ~ a + b * prompt_chars
    gen_coef: np.ndarray
    rewrite_mean: float
    gen_corrected_mean: float
    total_beta: np.ndarray
    post_beta: np.ndarray
    grade_batch_coef: np.ndarray | None = None  # trace 에 grade_batch 가 있을 때만
    meta: dict = field(default_factory=dict)

    def call_est(self, kind: str, chars: int) -> float:
        if kind == "grade":
            return float(self.grade_coef @ [1.0, chars])
        if kind == "grade_batch":
            if self.grade_batch_coef is not None:
                return float(self.grade_batch_coef @ [1.0, chars])
            return float(self.grade_coef @ [1.0, chars])  # 근사: batch prompt 길이로 grade 식 적용
        if kind == "generate":
            return float(self.gen_coef @ [1.0, chars])
        return self.rewrite_mean

    def pred_total(self, feats) -> float:
        return float(feats @ self.total_beta)

    def pred_post(self, feats) -> float:
        return float(feats @ self.post_beta)

    def est_remaining(self, kind: str, idx: int, chars: int, n_docs: int, feats) -> float:
        """이 호출을 포함한 query 의 남은 작업량 추정 (원본 srpt_est).

        grade: 남은 문서 수 x 문서당 비용 + grading 이후 작업량 예측.
        Lazy / Adaptive 처럼 중간에 멈출 수 있는 전략에서는 상한 추정이 된다.
        """
        if kind == "grade":
            return (n_docs - idx) * self.call_est("grade", chars) + self.pred_post(feats)
        if kind == "grade_batch":
            return self.call_est("grade_batch", chars) + self.pred_post(feats)
        if kind == "rewrite":
            return self.rewrite_mean + self.gen_corrected_mean
        return self.call_est(kind, chars)

    def to_dict(self) -> dict:
        return {
            "grade_coef": self.grade_coef.tolist(),
            "gen_coef": self.gen_coef.tolist(),
            "grade_batch_coef": None
            if self.grade_batch_coef is None
            else self.grade_batch_coef.tolist(),
            "rewrite_mean": self.rewrite_mean,
            "gen_corrected_mean": self.gen_corrected_mean,
            "total_beta": dict(zip(FEATURE_NAMES, self.total_beta.tolist(), strict=True)),
            "post_beta": dict(zip(FEATURE_NAMES, self.post_beta.tolist(), strict=True)),
            **self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CostModel:
        return cls(
            grade_coef=np.array(d["grade_coef"]),
            gen_coef=np.array(d["gen_coef"]),
            rewrite_mean=d["rewrite_mean"],
            gen_corrected_mean=d["gen_corrected_mean"],
            total_beta=np.array([d["total_beta"][k] for k in FEATURE_NAMES]),
            post_beta=np.array([d["post_beta"][k] for k in FEATURE_NAMES]),
            grade_batch_coef=None
            if d.get("grade_batch_coef") is None
            else np.array(d["grade_batch_coef"]),
            meta={k: v for k, v in d.items() if k not in _DICT_KEYS},
        )


_DICT_KEYS = {
    "grade_coef",
    "gen_coef",
    "grade_batch_coef",
    "rewrite_mean",
    "gen_corrected_mean",
    "total_beta",
    "post_beta",
}


def fit_cost_model(jobs: list[Job]) -> CostModel:
    calls = [c for j in jobs for k, cs in j.phases if k == "gpu" for c in cs]

    def lin(kind):
        cs = [c for c in calls if c.kind == kind]
        if not cs:
            return None
        X = np.array([[1.0, c.chars] for c in cs])
        return _lstsq(X, np.array([c.service for c in cs]), ridge=1e-6)

    rw = [c.service for c in calls if c.kind == "rewrite"]
    gen_corr = [
        c.service
        for j in jobs
        if j.action != "correct"
        for k, cs in j.phases
        if k == "gpu"
        for c in cs
        if c.kind == "generate"
    ]
    F = np.array([j.feats for j in jobs])
    gen_all = [c.service for c in calls if c.kind == "generate"]
    grade_coef = lin("grade")
    batch_coef = lin("grade_batch")
    if grade_coef is None:  # batch 만 있는 trace
        grade_coef = batch_coef if batch_coef is not None else np.zeros(2)
    return CostModel(
        grade_coef=grade_coef,
        gen_coef=lin("generate") if gen_all else np.zeros(2),
        rewrite_mean=float(np.mean(rw)) if rw else 0.0,
        gen_corrected_mean=float(np.mean(gen_corr)) if gen_corr else float(np.mean(gen_all)),
        total_beta=_lstsq(F, np.array([j.work for j in jobs])),
        post_beta=_lstsq(F, np.array([post_work(j) for j in jobs])),
        grade_batch_coef=batch_coef,
        meta={"n_jobs": len(jobs)},
    )


def attach_oof_predictions(jobs: list[Job], k: int = 5, seed: int = 0) -> None:
    """query 단위 예측값은 k-fold out-of-fold (원본과 동일)."""
    idx = np.random.default_rng(seed).permutation(len(jobs))
    for fold in range(k):
        test = set(idx[fold::k].tolist())
        model = fit_cost_model([j for i, j in enumerate(jobs) if i not in test])
        for i in test:
            jobs[i].pred_total = model.pred_total(jobs[i].feats)
            jobs[i].pred_post = model.pred_post(jobs[i].feats)
