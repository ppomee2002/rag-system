"""c2_gating 구성요소: T5 runner / generator engine (실제 또는 mock), GPU 샘플러."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

from rag.crag_official.t5_service import MockPairRunner, T5PairRunner


# ---------------------------------------------------------------- components
class MockEngine:
    """로컬 smoke 용 generator. prompt 의 함수 (같은 prompt -> 같은 출력). latency 는 sleep."""

    name, model = "mock", "mock-generator"

    def __init__(self, rewrite_s: float = 0.02, generate_s: float = 0.02):
        self.rewrite_s, self.generate_s = rewrite_s, generate_s
        self.lock = threading.Lock()
        self.active = self.max_active = 0

    def complete(self, request):
        from rag.llm.engine import LLMResponse

        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        t0 = time.perf_counter()
        time.sleep(self.rewrite_s if request.kind == "rewrite" else self.generate_s)
        if request.kind == "rewrite":
            q = request.prompt.rsplit("question: ", 1)[-1].split("\n")[0]
            text = f" {q}, keyword\n"
        else:
            doc = request.prompt.split("Documents: ", 1)[-1][:40]
            text = f" ans:{doc}"
        with self.lock:
            self.active -= 1
        return LLMResponse(
            text,
            self.name,
            self.model,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
            finish_reason="stop",
            t_start=t0,
            t_end=time.perf_counter(),
        )

    def describe(self):
        return {"engine": self.name, "note": "mock — 결과를 성능·정확도로 쓰지 않음"}


def mock_score(text: str) -> float:
    """합성 데이터용: 문서에 GOOD -> 1.0 (flag2), MID -> 0.0 (flag1), 그 외 -2.0 (flag0)."""
    p = text.split(" [SEP] ", 1)[-1]
    return 1.0 if "GOOD" in p else (0.0 if "MID" in p else -2.0)


def build_runner(cfg: dict):
    ev = cfg["evaluator"]
    if ev.get("backend") == "mock":
        m = cfg.get("mock") or {}
        return MockPairRunner(
            mock_score, m.get("t5_seconds_per_pair", 0.005), m.get("t5_seconds_per_batch", 0.0)
        )
    from rag.crag_official.evaluator import build_backend

    if ev.get("backend") != "t5":
        raise SystemExit(
            "c2_gating 은 evaluator.backend=t5 (실제 GPU 실행) 또는 mock 만 지원 "
            "(cache 는 T5 시간을 재현하지 않음)"
        )
    return T5PairRunner(build_backend(ev))


def build_engine(cfg: dict):
    if cfg["llm"].get("provider") == "mock":
        m = cfg.get("mock") or {}
        return MockEngine(m.get("rewrite_s", 0.02), m.get("generate_s", 0.02))
    from rag.crag_official.completion_engine import build_generator_engine

    return build_generator_engine(cfg["llm"])


class GpuSampler:
    """nvidia-smi -lms 로 GPU 사용률·메모리 샘플 (없으면 기록하지 않음)."""

    def __init__(self, path: Path, interval_ms: int):
        self.path, self.interval_ms = path, interval_ms
        self.proc = None

    def start(self):
        if self.interval_ms <= 0 or shutil.which("nvidia-smi") is None:
            return self
        self.f = open(self.path, "w", encoding="utf-8")  # noqa: SIM115
        self.proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                f"-lms={self.interval_ms}",
            ],
            stdout=self.f,
            stderr=subprocess.DEVNULL,
        )
        return self

    def stop(self) -> dict | None:
        if self.proc is None:
            return None
        self.proc.terminate()
        self.proc.wait(10)
        self.f.close()
        per: dict[str, list[float]] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 4:
                try:
                    per.setdefault(parts[1], []).append(float(parts[2]))
                except ValueError:
                    pass
        return {
            f"gpu{k}": {
                "n": len(v),
                "util_mean": sum(v) / len(v) if v else None,
                "util_busy_frac": sum(x > 0 for x in v) / len(v) if v else None,
            }
            for k, v in sorted(per.items())
        }
