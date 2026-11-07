# Experiment-local extension. Source: correction_scheduling/scheduling.py
"""실험 1 trace -> job 모델, 비용 예측 모델, 스케줄링 정책, discrete-event simulator.

GPU 모델: Ollama 서버 = 단일 GPU. 실험 1 probe 에서 동시 호출이 처리량을 늘리지 않음을 확인했으므로
(4개 동시 8.7s vs 순차 ~7.6s) `slots` 개 호출이 처리 속도를 나눠 쓰는 processor sharing 으로
모델링한다.

- native     : slots=4 (OLLAMA_NUM_PARALLEL), 초과분은 호출 도착 순서 FIFO. client 는 바로 전송
- 그 외 정책 : client gate 가 slots=1 로 한 번에 하나씩 보내고 대기 호출은 priority 순
               (낮을수록 먼저)
    fifo_call   호출 도착 순서
    fifo_query  query 도착 순서 (query 단위 FCFS)
    sjf_pred    retrieval 직후 예측한 query 전체 GPU 작업량 (짧은 query 우선)
    srpt_est    correction-aware: stage 진행에 따라 갱신되는 남은 작업량 추정치 (grading 이 끝나면
                correction 여부가 확정되므로 추정이 정확해진다)
    srpt_oracle 실제 남은 작업량 (상한선, live 불가)
"""

import math
from dataclasses import dataclass, field

import numpy as np

GATED_POLICIES = ("fifo_call", "fifo_query", "sjf_pred", "srpt_est", "srpt_oracle")


@dataclass
class Call:
    kind: str
    idx: int
    chars: int
    service: float  # 부하 없을 때 서버 처리 시간 (Ollama total_duration)


@dataclass
class Job:
    qid: str
    action: str
    feats: np.ndarray
    phases: list  # ("delay", sec) | ("gpu", [Call, ...])
    work: float = 0.0  # GPU 작업량 합
    solo: float = 0.0  # 부하 없을 때 E2E
    pred_total: float = 0.0
    pred_post: float = 0.0
    n_docs: int = 0
    acc: float = 0.0


def features(scores: list[float], chars: list[int], s_pop) -> np.ndarray:
    """retrieval 직후 (grading 전) 에 알 수 있는 값만 사용."""
    s = sorted(scores, reverse=True) or [0.0]
    gap = s[0] - s[1] if len(s) > 1 else 0.0
    return np.array(
        [1.0, s[0], float(np.mean(s)), gap, sum(chars) / 1e4, math.log10((s_pop or 0) + 1)]
    )


FEATURE_NAMES = ["intercept", "top1", "mean", "gap", "chars", "log_pop"]


def job_from_row(row: dict, parallel_grading: bool = False) -> Job:
    """profile.jsonl 의 crag (순차 grading) 행 -> Job. parallel_grading: grade 를 한 번에 낸다."""
    calls = sorted(row["calls"], key=lambda c: c["t_start"])
    by_kind = {k: [c for c in calls if c["kind"] == k] for k in ("grade", "rewrite", "generate")}

    def mk(c):
        return Call(c["kind"], c["idx"], c["prompt_chars"], c["server_total"])

    st = row["stage_latencies"]
    phases: list = [("delay", st["retrieval"])]
    grades = [mk(c) for c in by_kind["grade"]]
    if parallel_grading:
        phases.append(("gpu", grades))
    else:
        phases += [("gpu", [g]) for g in grades]
    if by_kind["rewrite"]:
        phases.append(("gpu", [mk(by_kind["rewrite"][0])]))
        phases.append(("delay", st["web_search"]))
    phases.append(("gpu", [mk(c) for c in by_kind["generate"]]))
    work = sum(c.service for _, cs in phases if _ == "gpu" for c in cs)
    solo = sum(v if k == "delay" else sum(c.service for c in v) for k, v in phases)
    return Job(
        qid=row["query_id"],
        action=row["corrective_action"],
        feats=features(row["extra"]["retrieval_scores"], row["retrieval_chars"], row.get("s_pop")),
        phases=phases,
        work=work,
        solo=solo,
        n_docs=len(grades),
        acc=row.get("acc", 0.0),
    )


def post_work(job: Job) -> float:
    """grading 이후 GPU 작업량 (rewrite + generation)."""
    return sum(c.service for k, cs in job.phases if k == "gpu" for c in cs if c.kind != "grade")


def _lstsq(X, y, ridge=1e-3):
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


@dataclass
class CostModel:
    """호출 / query 단위 GPU 작업량 예측. 모두 실험 1 trace 에서 선형회귀로 fit."""

    grade_coef: np.ndarray  # [a, b]: service ~ a + b * prompt_chars
    gen_coef: np.ndarray
    rewrite_mean: float
    gen_corrected_mean: float  # correction 된 query 의 generation 작업량 평균
    total_beta: np.ndarray  # features -> query 전체 작업량
    post_beta: np.ndarray  # features -> grading 이후 작업량
    meta: dict = field(default_factory=dict)

    def call_est(self, kind: str, chars: int) -> float:
        if kind == "grade":
            return float(self.grade_coef @ [1.0, chars])
        if kind == "generate":
            return float(self.gen_coef @ [1.0, chars])
        return self.rewrite_mean

    def pred_total(self, feats) -> float:
        return float(feats @ self.total_beta)

    def pred_post(self, feats) -> float:
        return float(feats @ self.post_beta)

    def est_remaining(self, kind: str, idx: int, chars: int, n_docs: int, feats) -> float:
        """srpt_est: 이 호출을 포함한 query 의 남은 GPU 작업량 추정."""
        if kind == "grade":
            return (n_docs - idx) * self.call_est("grade", chars) + self.pred_post(feats)
        if kind == "rewrite":
            return self.rewrite_mean + self.gen_corrected_mean
        return self.call_est(kind, chars)

    def to_dict(self) -> dict:
        return {
            "grade_coef": self.grade_coef.tolist(),
            "gen_coef": self.gen_coef.tolist(),
            "rewrite_mean": self.rewrite_mean,
            "gen_corrected_mean": self.gen_corrected_mean,
            "total_beta": dict(zip(FEATURE_NAMES, self.total_beta.tolist(), strict=True)),
            "post_beta": dict(zip(FEATURE_NAMES, self.post_beta.tolist(), strict=True)),
            **self.meta,
        }


def fit_cost_model(jobs: list[Job]) -> CostModel:
    calls = [c for j in jobs for k, cs in j.phases if k == "gpu" for c in cs]

    def lin(kind):
        cs = [c for c in calls if c.kind == kind]
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
    return CostModel(
        grade_coef=lin("grade"),
        gen_coef=lin("generate"),
        rewrite_mean=float(np.mean(rw)) if rw else 0.0,
        gen_corrected_mean=float(np.mean(gen_corr)) if gen_corr else float(np.mean(gen_all)),
        total_beta=_lstsq(F, np.array([j.work for j in jobs])),
        post_beta=_lstsq(F, np.array([post_work(j) for j in jobs])),
        meta={"n_jobs": len(jobs)},
    )


def attach_oof_predictions(jobs: list[Job], k: int = 5, seed: int = 0) -> None:
    """query 단위 예측값은 k-fold out-of-fold 로 넣는다 (자기 자신을 보고 예측하지 않도록)."""
    idx = np.random.default_rng(seed).permutation(len(jobs))
    for fold in range(k):
        test = set(idx[fold::k].tolist())
        model = fit_cost_model([j for i, j in enumerate(jobs) if i not in test])
        for i in test:
            jobs[i].pred_total = model.pred_total(jobs[i].feats)
            jobs[i].pred_post = model.pred_post(jobs[i].feats)


# ---------------------------------------------------------------- simulator


@dataclass
class _Run:
    job: Job
    arrival: float
    phase: int = -1
    pending: int = 0
    finish: float = math.nan
    gpu_wait: float = 0.0
    last_rescue_wait: float = 0.0
    rescue_count: int = 0


@dataclass
class _Active:
    run: _Run
    call: Call
    remaining: float
    enqueued: float
    base_priority: float = 0.0


def _priority(policy: str, run: _Run, call: Call, now: float, model: CostModel) -> float:
    job = run.job
    if policy in ("native", "fifo_call"):
        return now
    if policy == "fifo_query":
        return run.arrival
    if policy == "sjf_pred":
        return job.pred_total
    if policy in ("srpt_est", "srpt_threshold", "srpt_rescue"):
        if call.kind == "grade":
            return (job.n_docs - call.idx) * model.call_est("grade", call.chars) + job.pred_post
        return model.est_remaining(call.kind, call.idx, call.chars, job.n_docs, job.feats)
    if policy == "srpt_oracle":
        # 이 호출이 속한 phase 부터 끝까지의 실제 GPU 작업량
        return sum(c.service for k, cs in job.phases[run.phase :] if k == "gpu" for c in cs)
    raise ValueError(policy)


def simulate(
    jobs: list[Job],
    arrivals: list[float],
    policy: str,
    model: CostModel,
    native_slots: int = 4,
    efficiency: float = 1.0,
    aging_alpha: float = 0.0,
    wait_threshold: float | None = None,
    dispatch_log: list | None = None,
    dispatch_stats: dict | None = None,
) -> list[_Run]:
    """jobs[i] 가 arrivals[i] 에 도착. efficiency: 동시 처리 2개 이상일 때 GPU 처리율 배수."""
    import heapq

    if policy in ("srpt_threshold", "srpt_rescue"):
        if wait_threshold is None or not math.isfinite(wait_threshold) or wait_threshold < 0:
            raise ValueError("srpt_threshold requires a finite nonnegative wait_threshold")
        if aging_alpha != 0:
            raise ValueError("Threshold policy replaces Aging; combining them is not supported")
    elif wait_threshold is not None:
        raise ValueError("wait_threshold is only a dispatch parameter for srpt_threshold")
    if aging_alpha < 0 or not math.isfinite(aging_alpha):
        raise ValueError("aging_alpha must be finite and nonnegative")

    if dispatch_stats is not None:
        dispatch_stats.update(total_dispatches=0, rescue_dispatches=0, srpt_override_dispatches=0,
                              shorter_bypass_events=0, noncorrection_shorter_bypass_events=0,
                              max_consecutive_overrides=0, current_consecutive_overrides=0)

    slots = native_slots if policy == "native" else 1
    runs = [_Run(j, a) for j, a in zip(jobs, arrivals, strict=True)]
    events = [(a, i, "arrive", r) for i, (a, r) in enumerate(zip(arrivals, runs, strict=True))]
    heapq.heapify(events)
    seq = len(events)
    queue: list = []  # heap of (priority, seq, _Active)
    active: list[_Active] = []
    now = 0.0

    def rate():
        n = len(active)
        return (efficiency if n > 1 else 1.0) / n if n else 0.0

    def advance(run: _Run, t: float):
        nonlocal seq
        run.phase += 1
        if run.phase >= len(run.job.phases):
            run.finish = t
            return
        kind, val = run.job.phases[run.phase]
        if kind == "delay":
            seq += 1
            heapq.heappush(events, (t + val, seq, "delay", run))
        else:
            run.pending = len(val)
            for c in val:
                seq += 1
                heapq.heappush(
                    queue, (_priority(policy, run, c, t, model), seq, _Active(run, c, c.service, t, _priority(policy, run, c, t, model)))
                )

    def record_dispatch(a, queued, rescued):
        if dispatch_stats is None:
            return
        shorter = [other for _, _, other in queued if other.base_priority < a.base_priority]
        dispatch_stats['total_dispatches'] += 1
        dispatch_stats['rescue_dispatches'] += int(rescued)
        dispatch_stats['srpt_override_dispatches'] += int(bool(shorter))
        dispatch_stats['shorter_bypass_events'] += len(shorter)
        dispatch_stats['noncorrection_shorter_bypass_events'] += sum(other.run.job.action == 'correct' for other in shorter)
        dispatch_stats['current_consecutive_overrides'] = (dispatch_stats['current_consecutive_overrides'] + 1 if shorter else 0)
        dispatch_stats['max_consecutive_overrides'] = max(dispatch_stats['max_consecutive_overrides'], dispatch_stats['current_consecutive_overrides'])

    def fill(t: float):
        # Calls are nonpreemptive: reconsider only when a server slot is free.
        if len(active) >= slots:
            return
        if policy in ("srpt_threshold", "srpt_rescue") and queue:
            def threshold_key(item):
                _, serial, a = item
                waited = a.run.gpu_wait + t - a.enqueued
                # Strictly > T. Among overdue requests use longest cumulative wait.
                # Ties use immutable SRPT estimate, then enqueue serial.
                return ((0, -waited, a.base_priority, serial) if waited - (a.run.last_rescue_wait if policy == "srpt_rescue" else 0.0) > wait_threshold
                        else (1, a.base_priority, 0.0, serial))
            chosen = min(range(len(queue)), key=lambda i: threshold_key(queue[i]))
            _, _, a = queue.pop(chosen)
            heapq.heapify(queue)
            a.run.gpu_wait += t - a.enqueued
            rescued = policy in ("srpt_threshold", "srpt_rescue") and a.run.gpu_wait - (a.run.last_rescue_wait if policy == "srpt_rescue" else 0.0) > wait_threshold
            if policy == "srpt_rescue" and rescued:
                a.run.last_rescue_wait = a.run.gpu_wait
                a.run.rescue_count += 1
            record_dispatch(a, queue, rescued)
            if dispatch_log is not None:
                dispatch_log.append((t, a.run.job.qid, a.call.kind, a.call.idx, a.run.gpu_wait))
            active.append(a)
            return
        if aging_alpha and queue:
            # Re-evaluate immutable base priority at every dispatch. Age is
            # cumulative queue wait, never total time including GPU execution.
            queue[:] = [(a.base_priority - aging_alpha * (a.run.gpu_wait + t - a.enqueued), serial, a)
                        for _, serial, a in queue]
            heapq.heapify(queue)
        while queue and len(active) < slots:
            _, _, a = heapq.heappop(queue)
            a.run.gpu_wait += t - a.enqueued
            record_dispatch(a, queue, False)
            if dispatch_log is not None:
                dispatch_log.append((t, a.run.job.qid, a.call.kind, a.call.idx, a.run.gpu_wait))
            active.append(a)

    while events or active or queue:
        r = rate()
        t_gpu = now + min(a.remaining for a in active) / r if active else math.inf
        t_ev = events[0][0] if events else math.inf
        t_next = min(t_gpu, t_ev)
        for a in active:
            a.remaining -= (t_next - now) * r
        now = t_next
        if t_gpu <= t_ev:
            done = [a for a in active if a.remaining <= 1e-9]
            active = [a for a in active if a.remaining > 1e-9]
            for a in done:
                a.run.pending -= 1
                if a.run.pending == 0:
                    advance(a.run, now)
        else:
            _, _, _, run = heapq.heappop(events)
            advance(run, now)
        fill(now)
    return runs


def poisson_arrivals(n: int, rate: float, rng: np.random.Generator) -> list[float]:
    return np.cumsum(rng.exponential(1.0 / rate, n)).tolist()


def summarize(runs: list[_Run], warmup_frac: float = 0.0, threshold_seconds: float | None = None) -> dict:
    runs = runs[int(len(runs) * warmup_frac) :]
    e2e = np.array([r.finish - r.arrival for r in runs])
    corr = np.array([r.job.action != "correct" for r in runs])
    slow = e2e / np.array([r.job.solo for r in runs])
    span = max(r.finish for r in runs) - min(r.arrival for r in runs)

    def q(x, p):
        return float(np.percentile(x, p)) if len(x) else math.nan

    return {
        "mean": float(e2e.mean()),
        "p50": q(e2e, 50),
        "p95": q(e2e, 95),
        "p99": q(e2e, 99),
        "mean_slowdown": float(slow.mean()),
        "mean_correct": float(e2e[~corr].mean()) if (~corr).any() else math.nan,
        "mean_corrected": float(e2e[corr].mean()) if corr.any() else math.nan,
        "p95_correct": q(e2e[~corr], 95),
        "p95_corrected": q(e2e[corr], 95),
        "mean_gpu_wait": float(np.mean([r.gpu_wait for r in runs])),
        "max_gpu_wait": float(max(r.gpu_wait for r in runs)),
        "threshold_exceed_fraction": (float(np.mean([r.gpu_wait > threshold_seconds for r in runs]))
                                      if threshold_seconds is not None else math.nan),
        "n_requests": len(runs),
        "n_corrected": int(corr.sum()),
        "throughput": len(runs) / span,
    }
