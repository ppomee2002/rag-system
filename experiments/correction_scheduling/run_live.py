"""simulation 검증: 실제 Ollama 서버에 open-loop 동시 부하를 걸고 정책별 latency 를 측정.

    uv run python experiments/correction_scheduling/run_live.py <run_dir>

실험 1 query 중 live.n_queries 개를 골라 (cost model 은 나머지로 fit) 같은 Poisson 도착 trace 를
정책마다 재생하고, 같은 입력으로 돌린 simulation 예측과 비교한다.
출력: <run_dir>/live_<timestamp>/live.jsonl, comparison.csv, config.yaml
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Components, LLMGate, QueryState, call_kind, load_examples  # noqa: E402
from run_profile import read_done, to_row  # noqa: E402
from scheduling import (  # noqa: E402
    features,
    fit_cost_model,
    job_from_row,
    poisson_arrivals,
    simulate,
    summarize,
)

from rag.config import load_config  # noqa: E402
from rag.llm.ollama_llm import OllamaLLM  # noqa: E402


def make_priority_fn(policy: str, model):
    def fn(state: QueryState, kind: str, prompt: str, idx: int) -> float:
        if policy in ("native", "fifo_call"):
            return time.perf_counter()
        if policy == "fifo_query":
            return state.arrival
        feats = features(state.retrieval_scores, state.retrieval_chars, state.s_pop)
        if policy == "sjf_pred":
            return model.pred_total(feats)
        if policy == "srpt_est":
            n_docs = len(state.retrieval_scores)
            return model.est_remaining(call_kind(prompt), idx, len(prompt), n_docs, feats)
        raise ValueError(f"live 에서 지원하지 않는 정책: {policy}")

    return fn


def flush_prompt_cache(llm_cfg: dict, slots: int = 4) -> None:
    """이전 정책 실행의 prompt 가 Ollama slot cache 에 남지 않도록 무관한 prompt 로 덮어쓴다."""
    llm = OllamaLLM(
        llm_cfg["model"], max_tokens=4, num_ctx=llm_cfg["num_ctx"], host=llm_cfg["host"]
    )
    with ThreadPoolExecutor(slots) as pool:
        list(
            pool.map(
                lambda i: llm.generate(f"Count to three. ({i} {time.time()})"), range(slots * 2)
            )
        )


def run_policy(comp, policy, model, examples, arrivals, s_pops, llm_cfg):
    comp.gate = LLMGate(None if policy == "native" else 1)
    prio = make_priority_fn(policy, model)
    rows = []
    lock = threading.Lock()
    t0 = time.perf_counter()

    def worker(ex, arrival):
        time.sleep(max(0.0, t0 + arrival - time.perf_counter()))
        state = QueryState(ex.id, arrival=arrival, s_pop=s_pops[ex.id])
        t_start = time.perf_counter() - t0
        try:
            res = comp.pipeline("crag", state, prio, t0).run(ex)
            row = to_row(res, state, ex, t_start)
        except Exception as e:  # 기록하고 계속
            row = {"query_id": ex.id, "error": repr(e)}
        row.update(policy=policy, arrival=arrival, finish=time.perf_counter() - t0)
        row["latency"] = row["finish"] - arrival
        with lock:
            rows.append(row)

    threads = [
        threading.Thread(target=worker, args=(ex, a))
        for ex, a in zip(examples, arrivals, strict=True)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", default="experiments/correction_scheduling/config.yaml")
    parser.add_argument("--policies", nargs="+", default=None, help="live.policies override")
    args = parser.parse_args()
    cfg = load_config(args.config)
    lcfg = cfg["live"]

    prof = [r for r in read_done(args.run_dir / "profile.jsonl") if r["pipeline"] == "crag"]
    jobs = {r["query_id"]: job_from_row(r) for r in prof}
    mean_work = float(np.mean([j.work for j in jobs.values()]))
    rng = np.random.default_rng(lcfg["seed"])
    live_ids = [prof[i]["query_id"] for i in rng.choice(len(prof), lcfg["n_queries"], False)]
    model = fit_cost_model([j for q, j in jobs.items() if q not in set(live_ids)])
    arrivals = poisson_arrivals(len(live_ids), lcfg["load"] / mean_work, rng)

    by_id = {ex.id: ex for ex in load_examples(cfg)}
    examples = [by_id[q] for q in live_ids]
    s_pops = {q: by_id[q].metadata.get("s_pop") for q in live_ids}

    out = args.run_dir / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {
                **cfg,
                "live_run": {
                    "mean_work": mean_work,
                    "rate": lcfg["load"] / mean_work,
                    "query_ids": live_ids,
                    "arrivals": arrivals,
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    comp = Components(cfg, gate_slots=None)
    summary = []
    with open(out / "live.jsonl", "w", encoding="utf-8") as f:
        for policy in args.policies or lcfg["policies"]:
            flush_prompt_cache(cfg["llm"])
            print(f"[{datetime.now():%H:%M:%S}] live {policy} ...", flush=True)
            rows = run_policy(comp, policy, model, examples, arrivals, s_pops, cfg["llm"])
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            ok = [r for r in rows if "error" not in r]
            lat = np.array([r["latency"] for r in ok])
            sim_jobs = [jobs[q] for q in live_ids]
            for eff in sorted({1.0, cfg["sim"].get("native_efficiency", 1.0)}):
                if eff != 1.0 and policy != "native":
                    continue
                sim = summarize(
                    simulate(
                        sim_jobs,
                        arrivals,
                        policy,
                        model,
                        cfg["sim"]["native_slots"],
                        efficiency=eff,
                    )
                )
                summary.append(
                    {
                        "policy": policy,
                        "efficiency": eff,
                        "n_ok": len(ok),
                        "live_mean": lat.mean(),
                        "live_p50": np.percentile(lat, 50),
                        "live_p95": np.percentile(lat, 95),
                        "sim_mean": sim["mean"],
                        "sim_p50": sim["p50"],
                        "sim_p95": sim["p95"],
                    }
                )
            print(pd.DataFrame(summary).round(2).to_string(), flush=True)
    pd.DataFrame(summary).to_csv(out / "comparison.csv", index=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
