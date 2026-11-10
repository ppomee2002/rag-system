"""sequential run 의 trace 로 SRPT-est cost model 을 학습해 json 으로 저장.

    uv run python experiments/crag_system/fit_cost_model.py results/crag_system/SYS-2a/<run_id> \
        --service-key wall --out results/crag_system/cost_models/<name>.json

service_key: vLLM trace 는 wall (부하 없는 sequential run 의 호출 시간), Ollama 는 server_total.
학습한 질문 목록 (train_query_ids) 이 모델에 저장되고, runner 가 평가 workload 와 겹치면 거부한다.
  - 독립 학습: trace 를 dataset.split.part=train 으로 만들고 평가는 part=eval
  - OOF      : --oof 5 (평가 질문이 trace 에 있어도 자기 fold 를 뺀 모델로 예측)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import load_cost_model  # noqa: E402


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--service-key", default="wall")
    ap.add_argument(
        "--oof",
        type=int,
        default=None,
        metavar="K",
        help="K-fold OOF 모델 (평가 질문이 trace 에 포함될 때)",
    )
    ap.add_argument("--oof-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model = load_cost_model(
        {"trace": a.trace, "service_key": a.service_key, "oof_folds": a.oof, "oof_seed": a.oof_seed}
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2)
    print(
        f"saved -> {out} (n_jobs={model.meta['n_jobs']}, "
        f"train_queries={len(model.train_query_ids or [])}, "
        f"type={model.meta.get('type', 'single')})"
    )
    return out


if __name__ == "__main__":
    main()
