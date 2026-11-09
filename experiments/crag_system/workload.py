"""요청 도착 패턴 (workload) 생성·저장·검증.

같은 workload 파일을 여러 실험 조건 (grading / scheduling / engine) 에서 재사용해
"동일 질문, 동일 도착 패턴" 비교를 보장한다. 파일에는 요청 목록의 sha256 이 들어 있고,
불러올 때 다시 계산해 바뀌지 않았는지 확인한다.

도착률 기준 (rate_basis) 은 반드시 기록한다:
  rate           : λ 직접 지정 (req/s)
  rho_capacity   : λ = ρ x μ_max (같은 서버 조건에서 측정한 포화 처리량, req/s)
  rho_mean_work  : λ = ρ / E[단독 작업량]
                   (기존 L-CS / L-SO 정의. vLLM batching 에서는 용량 과소평가)
ρ 는 부하율 설정값일 뿐 GPU 사용률이 아니다 (GPU 사용률은 별도 측정).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def resolve_rate(rate=None, rho=None, capacity_rps=None, mean_work=None) -> tuple[float, str]:
    if rate is not None:
        if rho is not None:
            raise ValueError("rate 와 rho 는 함께 지정하지 않는다")
        return float(rate), "rate"
    if rho is None:
        raise ValueError("rate 또는 rho 가 필요")
    if capacity_rps is not None and mean_work is not None:
        raise ValueError("capacity_rps 와 mean_work 중 하나만")
    if capacity_rps is not None:
        return rho * capacity_rps, "rho_capacity"
    if mean_work is not None:
        return rho / mean_work, "rho_mean_work"
    raise ValueError("rho 에는 capacity_rps 또는 mean_work 가 필요")


def requests_sha(requests: list[dict]) -> str:
    blob = json.dumps(requests, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def make_workload(
    query_ids: list[str],
    n_requests: int | None = None,
    seed: int = 0,
    arrival: str = "poisson",  # poisson | sequential (도착 시각 없음, 순서만)
    sample: str = "permute",  # permute: query pool 을 섞어 반복 / replacement: 복원 추출
    rate: float | None = None,
    rho: float | None = None,
    capacity_rps: float | None = None,
    mean_work: float | None = None,
    name: str | None = None,
) -> dict:
    if not query_ids:
        raise ValueError("query_ids 가 비어 있음")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids 에 중복이 있음")
    n = n_requests or len(query_ids)
    rng = np.random.default_rng(seed)
    if sample == "permute":
        order: list[str] = []
        while len(order) < n:
            order += [query_ids[i] for i in rng.permutation(len(query_ids))]
        order = order[:n]
    elif sample == "replacement":
        order = [query_ids[i] for i in rng.integers(0, len(query_ids), n)]
    else:
        raise ValueError("sample: permute | replacement")

    params = {"n_requests": n, "seed": seed, "arrival": arrival, "sample": sample}
    if arrival == "poisson":
        lam, basis = resolve_rate(rate, rho, capacity_rps, mean_work)
        times = np.cumsum(rng.exponential(1.0 / lam, n)).tolist()
        params.update(
            rate=lam, rate_basis=basis, rho=rho, capacity_rps=capacity_rps, mean_work=mean_work
        )
    elif arrival == "sequential":
        times = [None] * n
    else:
        raise ValueError("arrival: poisson | sequential")
    requests = [
        {"i": i, "query_id": q, "arrival": t}
        for i, (q, t) in enumerate(zip(order, times, strict=True))
    ]
    return {
        "version": 1,
        "name": name,
        "created_utc": datetime.now(UTC).isoformat(),
        "params": params,
        "query_pool": list(query_ids),
        "requests": requests,
        "sha256": requests_sha(requests),
    }


def save_workload(wl: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:  # 덮어쓰지 않음
        json.dump(wl, f, ensure_ascii=False, indent=1)
    return path


def load_workload(path: str | Path) -> dict:
    wl = json.loads(Path(path).read_text(encoding="utf-8"))
    if requests_sha(wl["requests"]) != wl["sha256"]:
        raise ValueError(f"workload 내용이 sha256 과 다름 (수정됨?): {path}")
    return wl
