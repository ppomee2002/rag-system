"""Client 측 스케줄링 gate.

`SchedulingGate`: 동시 전송 수 K (slots) 를 넘는 LLM 호출을 대기시키고,
slot 이 비면 정책 key 가 가장 작은 호출을 보낸다. Aging / 대기 임계값처럼 시간이 지나면
순서가 바뀌는 정책은 dispatch 할 때마다 대기 중인 호출의 key 를 다시 계산한다 (O(대기 수)).

- slots=1 : 기존 실험의 client gate (한 번에 하나, 순서 제어 최대, vLLM batching 없음)
- slots=K : K 개까지 vLLM 에 동시에 보냄 → continuous batching 활용. K 가 클수록 vLLM 내부 순서가
            지배적이 되어 client 정책의 영향이 줄어든다 (K 는 실험 변수).
- slots=None : 제한 없음 (gate 는 기록만)

vLLM priority 전달: `engine_priority="key"` 면 dispatch 시점의 key 를 정수로 바꿔 요청에 넣는다.
vLLM 은 대기열에 있는 요청끼리만 priority 로 순서를 정하고, 보낸 뒤에는 priority 를 바꿀 수 없다.
(`--scheduling-policy priority` 서버에서만 의미가 있음. 낮을수록 먼저 — 버전별 확인 필요)

`LLMGate`: 기존 실험 (exp/correction-scheduling `common.py`) 의 정적 우선순위 gate.
호환을 위해 그대로 이식.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import Callable

from rag.scheduling.policies import CallTicket, Policy
from rag.scheduling.state import QueryState

TIER_SCALE = 10**9
PRIMARY_SCALE = 1000  # key primary (초 단위 추정치 등) -> ms 정수


def key_to_engine_priority(key: tuple) -> int:
    tier, primary = key[0], key[1]
    p = max(min(round(primary * PRIMARY_SCALE), TIER_SCALE // 2 - 1), -(TIER_SCALE // 2) + 1)
    return int(tier * TIER_SCALE + p)


class SchedulingGate:
    def __init__(
        self,
        policy: Policy,
        slots: int | None = 1,
        engine_priority: str = "none",  # none | key
        clock: Callable[[], float] = time.perf_counter,
        keep_log: bool = True,
    ):
        if slots is not None and slots < 1:
            raise ValueError("slots >= 1 또는 None")
        if engine_priority not in ("none", "key"):
            raise ValueError("engine_priority: none | key")
        self.policy = policy
        self.slots = slots
        self.engine_priority = engine_priority
        self.clock = clock
        self.keep_log = keep_log
        self._cv = threading.Condition()
        self._waiting: list[CallTicket] = []
        self._busy = 0
        self._seq = itertools.count()
        self.log: list[dict] = []
        self.max_waiting = 0

    def acquire(self, query: QueryState, kind: str, idx: int, prompt_chars: int) -> CallTicket:
        with self._cv:
            t = CallTicket(query, kind, idx, prompt_chars, self.clock(), next(self._seq))
            t.base = self.policy.base(t)
            if self.policy.base_is_estimate:
                query.record_estimate(kind, idx, t.base)
            self._waiting.append(t)
            self.max_waiting = max(self.max_waiting, len(self._waiting))
            self._dispatch()
            while not t.granted:
                self._cv.wait()
            return t

    def release(self, ticket: CallTicket) -> None:
        with self._cv:
            self._busy -= 1
            self._dispatch()

    def _dispatch(self) -> None:
        """lock 을 잡은 상태에서 호출. 빈 slot 만큼 key 가 작은 호출을 보낸다."""
        while self._waiting and (self.slots is None or self._busy < self.slots):
            now = self.clock()
            keys = [self.policy.key(t, now) for t in self._waiting]
            i = min(range(len(keys)), key=keys.__getitem__)
            t = self._waiting.pop(i)
            t.key_at_grant = keys[i]
            t.t_grant = now
            wait = now - t.t_enqueue
            with t.state.lock:
                t.state.cum_wait += wait
                t.state.n_dispatched += 1
            if self.engine_priority == "key":
                t.engine_priority = key_to_engine_priority(keys[i])
            t.granted = True
            self._busy += 1
            if self.keep_log:
                self.log.append(
                    {
                        "t": now,
                        "query_id": t.state.query_id,
                        "request_key": t.state.request_key,
                        "stage": t.state.stage,
                        "kind": t.kind,
                        "idx": t.idx,
                        "seq": t.seq,
                        "wait": wait,
                        "key": list(keys[i]),
                        "n_waiting": len(self._waiting),
                        "busy": self._busy,
                    }
                )
            self._cv.notify_all()

    def describe(self) -> dict:
        return {
            "policy": self.policy.describe(),
            "slots": self.slots,
            "engine_priority": self.engine_priority,
        }


class LLMGate:
    """정적 우선순위 gate (원본 그대로). slots=None 이면 제한 없이 바로 보낸다."""

    def __init__(self, slots: int | None):
        self.slots = slots
        self.busy = 0
        self.heap: list[tuple] = []
        self.seq = itertools.count()
        self.cv = threading.Condition()

    def acquire(self, priority: float) -> None:
        if self.slots is None:
            return
        with self.cv:
            ticket = (priority, next(self.seq))
            heapq.heappush(self.heap, ticket)
            while not (self.busy < self.slots and self.heap[0] == ticket):
                self.cv.wait()
            heapq.heappop(self.heap)
            self.busy += 1
            self.cv.notify_all()

    def release(self) -> None:
        if self.slots is None:
            return
        with self.cv:
            self.busy -= 1
            self.cv.notify_all()
