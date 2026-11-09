"""workload (요청 순서 + 도착 시각) 생성. 같은 파일을 여러 조건에서 재사용한다.

    # 동시 부하 (λ = ρ x 측정 용량)
    uv run python experiments/crag_system/make_workload.py --name rho0.8_seed0 \
        --n 500 --rho 0.8 --capacity-rps 0.42 --seed 0
    # 순서만 (sequential 실험에서 query 순서 고정)
    uv run python experiments/crag_system/make_workload.py --name seq_seed0 --arrival sequential

query pool 은 runner 와 같은 규칙 (seed shuffle + limit, 앞 warmup_queries 개 제외) 으로 만든다.
결과: results/crag_system/workloads/<name>.json (이미 있으면 실패)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_examples, load_layered_config, split_warmup  # noqa: E402
from workload import make_workload, save_workload  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/crag_system/config.yaml")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=None, help="요청 수 (기본: pool 크기)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arrival", choices=["poisson", "sequential"], default="poisson")
    ap.add_argument("--sample", choices=["permute", "replacement"], default="permute")
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--capacity-rps", type=float, default=None)
    ap.add_argument("--mean-work", type=float, default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "crag_system" / "workloads"))
    a = ap.parse_args(argv)
    cfg = load_layered_config(a.config, a.overlay, a.set)
    _, pool = split_warmup(cfg, load_examples(cfg))
    wl = make_workload(
        [ex.id for ex in pool],
        n_requests=a.n,
        seed=a.seed,
        arrival=a.arrival,
        sample=a.sample,
        rate=a.rate,
        rho=a.rho,
        capacity_rps=a.capacity_rps,
        mean_work=a.mean_work,
        name=a.name,
    )
    wl["source"] = {
        "dataset": cfg["dataset"],
        "seed": cfg["seed"],
        "warmup_queries": cfg.get("run", {}).get("warmup_queries", 0),
    }
    path = save_workload(wl, Path(a.out_dir) / f"{a.name}.json")
    print(f"saved -> {path} ({len(wl['requests'])} requests, sha256 {wl['sha256'][:12]})")
    return path


if __name__ == "__main__":
    main()
