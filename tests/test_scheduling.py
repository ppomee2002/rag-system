"""스케줄러 정책·gate·cost model 기능 검증 (성능 결과 아님)."""

import threading
import time

import numpy as np
import pytest

from rag.graph import CRAGGraph
from rag.llm.engine import LLMResponse
from rag.llm.engine_llm import EngineLLM
from rag.llm.vllm_engine import VLLMEngine
from rag.scheduling import (
    FCFS,
    Aging,
    CostModel,
    FCFSQuery,
    QueryState,
    SchedulingGate,
    SRPTEst,
    WaitThreshold,
    build_policy,
    fit_cost_model,
    job_from_row,
    key_to_engine_priority,
)
from rag.scheduling.policies import CallTicket
from rag.telemetry.calls import CallRecorder, infer_call_kind
from rag.types import Document, QAExample


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def ticket(q, seq, t_enqueue=0.0, base=0.0, kind="grade"):
    t = CallTicket(q, kind, 0, 100, t_enqueue, seq)
    t.base = base
    return t


def model(**kw):
    d = dict(
        grade_coef=np.array([1.0, 0.0]),
        gen_coef=np.array([2.0, 0.0]),
        rewrite_mean=0.5,
        gen_corrected_mean=3.0,
        total_beta=np.zeros(6),
        post_beta=np.array([2.0, 0, 0, 0, 0, 0]),
    )
    d.update(kw)
    return CostModel(**d)


# ---------------------------------------------------------------- policies
def test_fcfs_variants_base():
    q1, q2 = QueryState("a", arrival=5.0), QueryState("b", arrival=1.0)
    assert FCFS().base(ticket(q1, 3)) == 3.0
    assert FCFSQuery().base(ticket(q1, 3)) == 5.0 < 99
    assert FCFSQuery().base(ticket(q2, 9)) == 1.0


def test_srpt_est_matches_original_formula():
    q = QueryState("a", n_docs=5, retrieval_scores=[0.5] * 5, retrieval_chars=[100] * 5)
    p = SRPTEst(model())
    t = CallTicket(q, "grade", 2, 100, 0.0, 0)
    assert p.base(t) == pytest.approx(3 * 1.0 + 2.0)  # 남은 grade 3개 + pred_post
    assert p.base(CallTicket(q, "rewrite", 0, 50, 0.0, 1)) == pytest.approx(3.5)
    assert p.base(CallTicket(q, "generate", 0, 50, 0.0, 2)) == pytest.approx(2.0)
    assert p.base(CallTicket(q, "grade_batch", 0, 500, 0.0, 3)) == pytest.approx(1.0 + 2.0)


def test_aging_reorders_long_waiters():
    old, new = QueryState("old"), QueryState("new")
    p = Aging(FCFS(), alpha=1.0)
    long = ticket(old, 0, t_enqueue=0.0, base=10.0)
    short = ticket(new, 1, t_enqueue=9.0, base=2.0)
    assert p.key(short, 0.0 + 9.0) == (1, 2.0, 0.0, 1)  # 방금 들어옴: 2 - 0
    assert p.key(long, 9.0) == (1, 1.0, 0.0, 0)  # 9초 대기: 10 - 9
    assert p.key(long, 9.0) < p.key(short, 9.0)  # 오래 기다린 요청이 앞섬
    assert p.key(long, 5.0) > p.key(short, 9.0)  # 5초 대기 시점이면 아직 뒤
    old.cum_wait = 5.0  # 이전 호출 대기도 누적
    assert p.key(long, 9.0)[1] == pytest.approx(-4.0)


def test_wait_threshold_tiers():
    a, b = QueryState("a"), QueryState("b")
    p = WaitThreshold(FCFS(), threshold=30.0)
    over = ticket(a, 5, t_enqueue=0.0, base=100.0)
    under = ticket(b, 1, t_enqueue=20.0, base=1.0)
    assert p.key(over, 31.0)[0] == 0 and p.key(under, 31.0)[0] == 1
    assert p.key(over, 31.0) < p.key(under, 31.0)
    assert p.key(over, 29.0) > p.key(under, 29.0)


def test_build_policy():
    assert build_policy({"policy": "fcfs"}).name == "fcfs"
    assert build_policy({"policy": "srpt_est", "aging_alpha": 0.5}, model()).name == (
        "srpt_est+aging"
    )
    assert build_policy({"policy": "srpt_est", "wait_threshold": 60}, model()).name == (
        "srpt_est+wait60"
    )
    with pytest.raises(ValueError):
        build_policy({"policy": "srpt_est"})
    with pytest.raises(ValueError):
        build_policy({"policy": "fcfs", "aging_alpha": 0.5, "wait_threshold": 30})
    with pytest.raises(NotImplementedError):
        build_policy({"policy": "correction_aware"})


def test_engine_priority_mapping_preserves_order():
    keys = [(0, -50.0, 1.0, 3), (1, -2.0, 0.0, 1), (1, 0.5, 0.0, 0), (1, 3.0, 0.0, 2)]
    ints = [key_to_engine_priority(k) for k in keys]
    assert ints == sorted(ints) and all(isinstance(i, int) for i in ints)


# ---------------------------------------------------------------- gate
def wait_until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise TimeoutError
        time.sleep(0.001)


def dispatch_order(gate, clock, specs):
    """첫 호출이 slot 을 잡은 상태에서 specs 를 대기시킨 뒤 dispatch 순서 확인.

    specs: (query, kind, idx, enqueue 시각)
    """
    holder = gate.acquire(QueryState("holder"), "generate", 0, 10)
    order, threads = [], []
    lock = threading.Lock()

    def worker(q, kind, idx):
        t = gate.acquire(q, kind, idx, 100)
        with lock:
            order.append(q.query_id)
        gate.release(t)

    for q, kind, idx, at in specs:
        clock.t = at
        th = threading.Thread(target=worker, args=(q, kind, idx))
        th.start()
        threads.append(th)
        wait_until(lambda n=len(threads): len(gate._waiting) == n)  # noqa: B008
    clock.t = specs[-1][3] + 1.0
    gate.release(holder)
    for th in threads:
        th.join(5)
    return order


def test_gate_fcfs_order():
    clock = Clock()
    gate = SchedulingGate(FCFS(), slots=1, clock=clock)
    qs = [QueryState(n) for n in "abc"]
    assert dispatch_order(gate, clock, [(q, "grade", 0, i) for i, q in enumerate(qs)]) == [
        "a",
        "b",
        "c",
    ]


def test_gate_srpt_prefers_short_remaining():
    clock = Clock()
    gate = SchedulingGate(SRPTEst(model()), slots=1, clock=clock)
    long_q = QueryState("long", n_docs=5)
    short_q = QueryState("short", n_docs=5)
    order = dispatch_order(gate, clock, [(long_q, "grade", 0, 0.0), (short_q, "generate", 0, 1.0)])
    assert order == ["short", "long"]  # generate(2.0) < grade 5개 + post (7.0)


def test_gate_aging_lets_old_request_overtake():
    clock = Clock()
    long_q = QueryState("long", n_docs=5)
    short_q = QueryState("short", n_docs=5)
    specs = [(long_q, "grade", 0, 0.0), (short_q, "generate", 0, 10.0)]
    gate = SchedulingGate(Aging(SRPTEst(model()), alpha=1.0), slots=1, clock=clock)
    # dispatch 시각 11: long = 7 - 11 = -4, short = 2 - 1 = 1 -> long 먼저
    assert dispatch_order(gate, clock, specs) == ["long", "short"]
    assert long_q.cum_wait == pytest.approx(11.0)


def test_gate_k_slots_and_log():
    gate = SchedulingGate(FCFS(), slots=2)
    q = QueryState("q")
    t1 = gate.acquire(q, "grade", 0, 1)
    t2 = gate.acquire(q, "grade", 1, 1)  # slot 2개라 바로 통과
    got = []
    th = threading.Thread(target=lambda: got.append(gate.acquire(q, "grade", 2, 1)))
    th.start()
    wait_until(lambda: len(gate._waiting) == 1)
    assert got == []
    gate.release(t1)
    th.join(5)
    assert len(got) == 1
    gate.release(t2)
    gate.release(got[0])
    assert [e["idx"] for e in gate.log] == [0, 1, 2] and q.n_dispatched == 3


def test_gate_unlimited_slots_never_blocks():
    gate = SchedulingGate(FCFS(), slots=None)
    tickets = [gate.acquire(QueryState(str(i)), "grade", 0, 1) for i in range(20)]
    assert all(t.granted for t in tickets)


# ---------------------------------------------------------------- EngineLLM + scheduler
def test_engine_llm_passes_dispatch_priority_to_vllm(mock_server):
    srv = mock_server("vllm", sched_policy="priority")
    eng = VLLMEngine("mock-model", base_url=srv.url + "/v1", api_key_env=None, send_priority=True)
    gate = SchedulingGate(WaitThreshold(FCFS(), 30.0), slots=1, engine_priority="key")
    rec, q = CallRecorder(), QueryState("q1")
    llm = EngineLLM(eng, recorder=rec, scheduler=gate, query_state=q)
    llm.generate("hello")
    llm.generate("hello again")
    sent = [r["body"]["priority"] for r in srv.chat_requests()]
    calls = rec.to_dicts()
    assert sent == [key_to_engine_priority(tuple(c["dispatch_key"])) for c in calls]
    assert [c["engine_priority"] for c in calls] == sent
    assert calls[0]["priority"] == 0.0 and calls[1]["priority"] == 1.0  # FCFS base = seq


def test_engine_llm_scheduler_requires_query_state():
    with pytest.raises(ValueError):
        EngineLLM(object(), scheduler=SchedulingGate(FCFS()))


class SleepyEngine:
    """in-process 가짜 엔진: 호출 시간을 흉내내고 동시 실행 수를 잰다."""

    name, model = "fake", "fake"

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def complete(self, request):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        t0 = time.perf_counter()
        time.sleep(0.005)
        kind = infer_call_kind(request.prompt)
        text = "0.9" if kind == "grade" else ("q" if kind == "rewrite" else "ans")
        with self.lock:
            self.active -= 1
        return LLMResponse(
            text,
            "fake",
            t_start=t0,
            t_end=time.perf_counter(),
            prompt_tokens=10,
            completion_tokens=2,
        )

    def describe(self):
        return {"engine": "fake"}


class Retriever:
    def retrieve(self, query, top_k):
        return [Document(id=f"d{i}", text=f"doc{i}", score=0.5) for i in range(top_k)]

    search = retrieve


@pytest.mark.parametrize("slots", [1, 3])
def test_concurrent_graph_runs_respect_k_slots(slots):
    eng = SleepyEngine()
    gate = SchedulingGate(FCFS(), slots=slots)
    graph = CRAGGraph(Retriever(), None, Retriever(), top_k=3)
    results, states = [], []

    def one(i):
        q = QueryState(f"q{i}")
        states.append(q)
        llm = EngineLLM(eng, recorder=CallRecorder(), scheduler=gate, query_state=q)
        results.append(graph.run(QAExample(f"q{i}", "Q?", ["ans"]), llm=llm, query_state=q))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(results) == 6 and eng.max_active <= slots
    assert all(s.action == "correct" and s.n_docs == 3 for s in states)
    assert len(gate.log) == 6 * 4  # grade 3 + generate 1


# ---------------------------------------------------------------- cost model
def trace_row(qid, action, grade_services, gen_service, service_key="server_total"):
    calls = [
        {"kind": "grade", "idx": i, "prompt_chars": 500 + 100 * i, "t_start": i, service_key: s}
        for i, s in enumerate(grade_services)
    ]
    t = len(calls)
    if action != "correct":
        calls.append(
            {"kind": "rewrite", "idx": 0, "prompt_chars": 80, "t_start": t, service_key: 0.3}
        )
        t += 1
    calls.append(
        {"kind": "generate", "idx": 0, "prompt_chars": 2000, "t_start": t, service_key: gen_service}
    )
    return {
        "query_id": qid,
        "corrective_action": action,
        "calls": calls,
        "stage_latencies": {"retrieval": 0.05, "web_search": 1.0},
        "extra": {"retrieval_scores": [0.8, 0.6, 0.5, 0.4, 0.3]},
        "retrieval_chars": [500] * 5,
        "s_pop": 50,
        "acc": 1.0,
    }


def test_job_from_row_and_fit():
    rows = [
        trace_row(f"q{i}", "correct" if i % 2 else "incorrect", [0.5 + 0.01 * i] * 5, 2.0 + 0.1 * i)
        for i in range(12)
    ]
    jobs = [job_from_row(r) for r in rows]
    assert jobs[0].phases[0] == ("delay", 0.05)
    assert [k for k, _ in jobs[0].phases].count("delay") == 2  # retrieval + web_search
    m = fit_cost_model(jobs)
    assert m.rewrite_mean == pytest.approx(0.3)
    assert m.meta["n_jobs"] == 12
    m2 = CostModel.from_dict(m.to_dict())
    assert m2.call_est("grade", 700) == pytest.approx(m.call_est("grade", 700))
    wall_rows = [trace_row("w", "correct", [0.5] * 5, 2.0, service_key="wall")]
    assert job_from_row(wall_rows[0], service_key="wall").work == pytest.approx(4.5)
    with pytest.raises(ValueError):
        job_from_row(wall_rows[0])


def test_srpt_nopath_differs_only_on_rewrite():
    from rag.scheduling import SRPTNoPath

    q = QueryState("a", n_docs=5, retrieval_scores=[0.5] * 5, retrieval_chars=[100] * 5)
    full, nopath = SRPTEst(model()), SRPTNoPath(model())
    for kind, idx, chars in [("grade", 0, 100), ("grade", 4, 300), ("generate", 0, 50)]:
        t = CallTicket(q, kind, idx, chars, 0.0, 0)
        assert nopath.base(t) == pytest.approx(full.base(t))
    rw = CallTicket(q, "rewrite", 0, 50, 0.0, 1)
    assert full.base(rw) == pytest.approx(0.5 + 3.0)  # rewrite_mean + gen_corrected_mean
    assert nopath.base(rw) == pytest.approx(2.0)  # pred_post (경로 무관 기대값)
    p = build_policy({"policy": "srpt_nopath"}, model())
    assert p.name == "srpt_nopath" and p.base_is_estimate
    with pytest.raises(ValueError):
        build_policy({"policy": "srpt_nopath"})
