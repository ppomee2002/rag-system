"""sequential run 의 trace 로 SRPT-est cost model 을 학습해 json 으로 저장.

    uv run python experiments/crag_system/fit_cost_model.py results/crag_system/SYS-2a/<run_id> \
        --service-key wall --out results/crag_system/cost_models/<name>.json

service_key: vLLM trace 는 wall (부하 없는 sequential run 의 호출 시간), Ollama 는 server_total.
스케줄링 평가에 쓰는 workload 의 query 로 학습하면 예측이 낙관적이 되므로, 가능하면 다른 query 의
trace 로 학습하거나 OOF 로 비교한다 (rag.scheduling.cost_model.attach_oof_predictions).
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model = load_cost_model({"trace": a.trace, "service_key": a.service_key})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2)
    print(f"saved -> {out} (n_jobs={model.meta['n_jobs']})")
    return out


if __name__ == "__main__":
    main()
