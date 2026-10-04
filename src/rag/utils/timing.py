import time
from collections import defaultdict
from contextlib import contextmanager


class StageTimer:
    """단계별 latency 기록기.

    같은 stage 가 여러 번 실행되면 (예: rewrite 반복) 시간은 누적되고 횟수도 함께 기록된다.

        timer = StageTimer()
        with timer.stage("retrieval"):
            ...
        timer.totals  # {"retrieval": 0.123}
    """

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - start
            self.counts[name] += 1


@contextmanager
def stopwatch():
    """E2E latency 측정용. `with stopwatch() as sw: ...; sw["elapsed"]`"""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
