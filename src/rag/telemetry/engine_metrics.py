"""실행 중 vLLM `/metrics` 를 주기적으로 읽어 기록한다.

K calibration (vLLM 내부 대기열이 생기는지, preemption 이 일어나는지) 과 priority 검증에 쓴다.
vLLM 0.33.0 지표 이름 (소스 확인): vllm:num_requests_running, vllm:num_requests_waiting,
vllm:kv_cache_usage_perc, vllm:num_preemptions (counter),
vllm:request_queue_time_seconds (histogram).
이름이 바뀐 버전에서는 해당 값이 None 으로 남는다 (오류로 멈추지 않음).
"""

from __future__ import annotations

import threading
import time

GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
}
COUNTERS = {
    "preemptions": "vllm:num_preemptions",
    "queue_time_sum": "vllm:request_queue_time_seconds_sum",
    "queue_time_count": "vllm:request_queue_time_seconds_count",
}


def _sum(metrics: dict, name: str) -> float | None:
    """label 이 다른 series 를 합친다. `name{...}` 와 `name` 만 (다른 접미사는 제외)."""
    vals = [v for k, v in metrics.items() if k == name or k.startswith(name + "{")]
    return sum(vals) if vals else None


def snapshot(metrics: dict) -> dict:
    out = {k: _sum(metrics, n) for k, n in GAUGES.items()}
    for k, n in COUNTERS.items():
        out[k] = _sum(metrics, n)
        if out[k] is None:  # prometheus counter 는 _total 접미사가 붙어 노출됨
            out[k] = _sum(metrics, n + "_total")
    return out


def delta(before: dict | None, after: dict | None) -> dict:
    if not before or not after:
        return {}
    return {
        k: (after[k] - before[k])
        if after.get(k) is not None and before.get(k) is not None
        else None
        for k in COUNTERS
    }


class MetricsSampler:
    """`fetch()` (dict 를 돌려주는 함수, 예: lambda: parse_prometheus(engine.metrics_text())) 를
    interval 초마다 호출해 gauge 값을 모은다."""

    def __init__(
        self, fetch, interval: float = 0.5, clock=time.perf_counter, t0: float | None = None
    ):
        self.fetch = fetch
        self.interval = interval
        self.clock = clock
        self.t0 = clock() if t0 is None else t0
        self.samples: list[dict] = []
        self.errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.first: dict | None = None
        self.last: dict | None = None

    def _take(self) -> None:
        try:
            snap = snapshot(self.fetch())
        except Exception:  # noqa: BLE001 - 측정 보조. 실패해도 실험은 계속
            self.errors += 1
            return
        snap["t"] = self.clock() - self.t0
        self.samples.append(snap)
        self.first = self.first or snap
        self.last = snap

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._take()
            self._stop.wait(self.interval)

    def start(self) -> MetricsSampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._take()
        return self.summary()

    def summary(self) -> dict:
        def stat(key):
            vals = [s[key] for s in self.samples if s.get(key) is not None]
            if not vals:
                return None
            return {"max": max(vals), "mean": sum(vals) / len(vals)}

        return {
            "n_samples": len(self.samples),
            "errors": self.errors,
            "interval_s": self.interval,
            **{k: stat(k) for k in GAUGES},
            "counter_delta": delta(self.first, self.last),
        }
