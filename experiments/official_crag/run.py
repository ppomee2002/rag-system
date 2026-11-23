"""MODE B 실행 (Controlled Server Comparison).

    uv run python experiments/official_crag/run.py --config experiments/official_crag/config.yaml \
        --overlay experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml \
        --set grader.type=official_t5 --set evidence_policy=official --exp-id SMOKE-C1-off

설정 우선순위: configs/base.yaml < --config < --overlay (순서대로) < --set
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mode_b import ModeBRunner  # noqa: E402

from rag.config import load_layered_config  # noqa: E402


def main(argv=None) -> Path:
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/official_crag/config.yaml")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=None, help="inputs.limit (smoke)")
    ap.add_argument("--workload", default=None, help="live 모드 workload json (crag_system 형식)")
    ap.add_argument("--paper-approved", action="store_true")
    a = ap.parse_args(argv)
    load_dotenv()
    cfg = load_layered_config(a.config, a.overlay, a.set)
    if a.exp_id:
        cfg["exp_id"] = a.exp_id
    if a.limit is not None:
        cfg.setdefault("inputs", {})["limit"] = a.limit
    workload = json.loads(Path(a.workload).read_text(encoding="utf-8")) if a.workload else None
    run_dir = ModeBRunner(
        cfg, run_id=a.run_id, paper_approved=a.paper_approved, workload=workload
    ).run()
    print(f"saved -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
