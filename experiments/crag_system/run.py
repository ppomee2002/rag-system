"""SYS 실험 실행.

    uv run python experiments/crag_system/run.py --config experiments/crag_system/config.yaml \
        [--overlay experiments/crag_system/configs/engine_ollama.yaml] \
        [--set grading.strategy=batch --set scheduler.policy=srpt_est ...] \
        [--workload results/crag_system/workloads/<name>.json] [--exp-id SMOKE-x] [--limit 20]

설정 우선순위: configs/base.yaml < --config < --overlay (순서대로) < --set
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import Runner, load_layered_config  # noqa: E402
from workload import load_workload  # noqa: E402


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/crag_system/config.yaml")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--workload", default=None)
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=None, help="dataset.limit (smoke run)")
    ap.add_argument(
        "--paper-approved",
        action="store_true",
        help="SYS 본실험 (run.purpose=paper) 실행 승인. 별도 승인 후에만 사용",
    )
    a = ap.parse_args(argv)
    load_dotenv()
    cfg = load_layered_config(a.config, a.overlay, a.set)
    if a.exp_id:
        cfg["exp_id"] = a.exp_id
    if a.limit is not None:
        cfg["dataset"]["limit"] = a.limit
        cfg.setdefault("run", {})["limit_override"] = a.limit
    workload = load_workload(a.workload) if a.workload else None
    run_dir = Runner(cfg, workload=workload, run_id=a.run_id, paper_approved=a.paper_approved).run()
    print(f"saved -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
