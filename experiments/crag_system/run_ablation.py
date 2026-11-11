"""진단용 Ablation 실행 (ablation.py). 인자는 run.py 와 같다.

    uv run python experiments/crag_system/run_ablation.py --config <Full run>/config.yaml \
        --overlay experiments/crag_system/configs/diag/replay_full_retry.yaml \
        --overlay experiments/crag_system/configs/diag/abl_incorrect_keep_faiss.yaml \
        --set run.purpose=pre --exp-id PRE-ABL-A-INC-KEEP

ablation 키가 없으면 실행하지 않는다 (Ablation 없는 실행은 run.py).
주의: ablation overlay 를 run.py 로 실행하면 ablation 키가 무시되어 일반 Full 이 된다.
결과의 pipeline 이름 (`+abl_...`) 과 extra.ablation 으로 구분한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import AblationCRAGGraph, parse_ablation  # noqa: E402
from runner import Runner, load_layered_config  # noqa: E402
from workload import load_workload  # noqa: E402


class AblationRunner(Runner):
    """Runner 를 그대로 쓰고 그래프만 AblationCRAGGraph 로 바꾼다."""

    def __init__(self, cfg: dict, *args, **kwargs):
        ablation = parse_ablation(cfg)  # 결과 폴더를 만들기 전에 검증
        super().__init__(cfg, *args, **kwargs)
        self.graph = AblationCRAGGraph.from_graph(self.graph, ablation)


def main(argv=None) -> Path:
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="기준 Full run 의 config.yaml 권장")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--workload", default=None)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    load_dotenv()
    cfg = load_layered_config(a.config, a.overlay, a.set)
    cfg["exp_id"] = a.exp_id
    if (cfg.get("run") or {}).get("purpose") == "paper":
        raise SystemExit("진단 ablation 은 run.purpose=paper 로 실행하지 않음")
    workload = load_workload(a.workload) if a.workload else None
    run_dir = AblationRunner(cfg, workload=workload, run_id=a.run_id).run()
    print(f"saved -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
