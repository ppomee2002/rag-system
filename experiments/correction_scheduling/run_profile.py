"""실험 1 (Stage-Level Profiling): 부하 없이 query 를 하나씩 실행하며 stage / LLM 호출 trace 수집.

    uv run python experiments/correction_scheduling/run_profile.py [--limit 8] [--resume <run_dir>]

이 trace 가 실험 2 (workload 분석) 의 데이터이자 실험 3·4 simulation 의 서비스 시간 입력이 된다.
출력: results/correction_scheduling/<timestamp>/profile.jsonl, config.yaml, errors.jsonl
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crag_latency"))
from common import Components, QueryState, fifo_priority, load_examples  # noqa: E402
from run import apply_overrides  # noqa: E402

from rag.config import load_config  # noqa: E402
from rag.eval.metrics import score_all  # noqa: E402


def to_row(res, state: QueryState, ex, t_start: float) -> dict:
    row = asdict(res)
    row["contexts"] = [d["id"] for d in row["contexts"]]
    row["gold_answers"] = ex.answers
    row["s_pop"] = ex.metadata.get("s_pop")
    row["retrieval_chars"] = state.retrieval_chars
    row["calls"] = sorted(state.calls, key=lambda c: c["t_start"])
    row["t_start"] = t_start
    row.update(score_all(res.answer, ex.answers))
    return row


def read_done(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # 강제 종료로 잘린 마지막 줄
                pass
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/correction_scheduling/config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg["dataset"]["limit"] = args.limit
    apply_overrides(cfg, args.set)
    pcfg = cfg["profile"]

    if args.resume:
        out_dir = args.resume
    else:
        out_dir = (
            Path(cfg["output_dir"])
            / "correction_scheduling"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
    done = {(r["query_id"], r["pipeline"]) for r in read_done(out_dir / "profile.jsonl")}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    print(f"out_dir={out_dir} done={len(done)}")

    examples = load_examples(cfg)
    comp = Components(cfg, gate_slots=None)
    t0 = time.perf_counter()

    warmup = pcfg["warmup_queries"]
    for ex in examples[:warmup]:
        for name in pcfg["pipelines"] + pcfg["subset_pipelines"]:
            comp.pipeline(name, QueryState(ex.id), fifo_priority, t0).run(ex)

    # 같은 query 를 pipeline 끼리 연달아 실행하면 동일 prompt 가 Ollama prompt(KV) cache 에 걸려
    # 두 번째 실행이 수 배 빨라진다 -> query 단위 interleave 대신 pipeline 별 pass 로 나눠 실행
    measured = examples[warmup:]
    passes = [(name, measured) for name in pcfg["pipelines"]]
    passes += [(name, measured[: pcfg["subset_n"]]) for name in pcfg["subset_pipelines"]]
    n_err = 0
    with (
        open(out_dir / "profile.jsonl", "a", encoding="utf-8") as f,
        open(out_dir / "errors.jsonl", "a", encoding="utf-8") as ferr,
    ):
        for name, exs in passes:
            for ex in tqdm(exs, desc=name):
                if (ex.id, name) in done:
                    continue
                state = QueryState(ex.id)
                t_start = time.perf_counter() - t0
                try:
                    res = comp.pipeline(name, state, fifo_priority, t0).run(ex)
                except Exception as e:  # 기록 후 계속. --resume 으로 재시도
                    n_err += 1
                    ferr.write(
                        json.dumps(
                            {"query_id": ex.id, "pipeline": name, "error": repr(e)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    ferr.flush()
                    continue
                f.write(json.dumps(to_row(res, state, ex, t_start), ensure_ascii=False) + "\n")
                f.flush()
    if n_err:
        print(f"{n_err} errors -> {out_dir / 'errors.jsonl'} (--resume 으로 재시도)")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
