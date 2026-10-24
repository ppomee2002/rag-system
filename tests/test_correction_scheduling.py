import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments" / "correction_scheduling")
)
from scheduling import Call, Job, fit_cost_model, simulate  # noqa: E402


def make_job(qid, grades, gen, action="correct", web=0.0):
    phases = [("delay", 0.01)] + [("gpu", [Call("grade", i, 500, g)]) for i, g in enumerate(grades)]
    if action != "correct":
        phases += [("gpu", [Call("rewrite", 0, 100, 0.2)]), ("delay", web)]
    phases.append(("gpu", [Call("generate", 0, 1000 + 100 * len(grades), gen)]))
    work = sum(c.service for k, cs in phases if k == "gpu" for c in cs)
    solo = sum(v if k == "delay" else sum(c.service for c in v) for k, v in phases)
    return Job(
        qid,
        action,
        np.array([1.0, 0.5, 0.4, 0.1, 0.1, 1.0]),
        phases,
        work,
        solo,
        n_docs=len(grades),
    )


JOBS = [
    make_job("a", [1.0, 1.0], 2.0),
    make_job("b", [1.0, 1.0], 0.5, action="incorrect", web=1.5),
    make_job("c", [0.5], 0.5),
]
MODEL = fit_cost_model(JOBS)


def test_isolated_job_takes_solo_time():
    for policy in ("native", "fifo_call", "srpt_oracle"):
        for job in JOBS:
            (run,) = simulate([job], [0.0], policy, MODEL)
            assert abs(run.finish - job.solo) < 1e-9


def test_gate_is_work_conserving_and_serial():
    # 동시에 도착하면 단일 서버라 마지막 완료 시각 >= 총 GPU 작업량
    runs = simulate(JOBS, [0.0, 0.001, 0.002], "fifo_query", MODEL)
    assert max(r.finish for r in runs) >= sum(j.work for j in JOBS) - 1e-9
    # query FCFS: 먼저 도착한 query 의 호출이 항상 우선
    assert runs[0].finish < runs[2].finish


def test_oracle_srpt_favors_short_job():
    runs = simulate(JOBS, [0.0, 0.0, 0.0], "srpt_oracle", MODEL)
    assert runs[2].finish == min(r.finish for r in runs)


def test_processor_sharing_efficiency_slows_concurrent_calls():
    two = [make_job("x", [1.0], 0.0001), make_job("y", [1.0], 0.0001)]
    full = simulate(two, [0.0, 0.0], "native", MODEL, efficiency=1.0)
    slow = simulate(two, [0.0, 0.0], "native", MODEL, efficiency=0.5)
    assert max(r.finish for r in slow) > max(r.finish for r in full)
