import time

from rag.utils.timing import StageTimer, stopwatch


def test_stage_timer_accumulates_repeated_stages():
    timer = StageTimer()
    for _ in range(2):
        with timer.stage("rewrite"):
            time.sleep(0.01)
    assert timer.counts["rewrite"] == 2
    assert timer.totals["rewrite"] >= 0.02


def test_stopwatch():
    with stopwatch() as sw:
        time.sleep(0.01)
    assert sw["elapsed"] >= 0.01
