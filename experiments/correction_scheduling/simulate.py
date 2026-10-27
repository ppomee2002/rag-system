"""실험 3 (Concurrent Query) / 4 (Scheduling Baseline): 실험 1 trace 기반 discrete-event simulation.

    uv run python experiments/correction_scheduling/simulate.py <run_dir>

open-loop Poisson 도착 (rho = 도착률 x 평균 GPU 작업량), trace 에서 query 를 복원추출해 재생한다.
출력: <run_dir>/sim/sim_results.csv, cost_model.json, figures/*.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scheduling import (  # noqa: E402
    attach_oof_predictions,
    fit_cost_model,
    job_from_row,
    poisson_arrivals,
    simulate,
    summarize,
)

from rag.config import load_config  # noqa: E402


def load_jobs(run_dir: Path, parallel_grading: bool = False):
    rows = [json.loads(line) for line in (run_dir / "profile.jsonl").open(encoding="utf-8")]
    jobs = [job_from_row(r, parallel_grading) for r in rows if r["pipeline"] == "crag"]
    attach_oof_predictions(jobs)
    return jobs


POLICY_LABEL = {
    "native": "native (Ollama parallel slots)",
    "fifo_call": "gate FIFO (call)",
    "fifo_query": "gate FCFS (query)",
    "sjf_pred": "gate SJF (predicted)",
    "srpt_est": "gate SRPT-est (correction-aware)",
    "srpt_oracle": "gate SRPT-oracle",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", default="experiments/correction_scheduling/config.yaml")
    args = parser.parse_args()
    # trace 는 run_dir 에서, simulation 설정은 현재 실험 config 에서 읽는다 (sim 만 반복 가능)
    scfg = load_config(args.config)["sim"]
    out = args.run_dir / "sim"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    variants = {"seq": load_jobs(args.run_dir), "par": load_jobs(args.run_dir, True)}
    model = fit_cost_model(variants["seq"])
    mean_work = float(np.mean([j.work for j in variants["seq"]]))
    (out / "cost_model.json").write_text(
        json.dumps({**model.to_dict(), "mean_work": mean_work}, indent=2), encoding="utf-8"
    )
    eff = scfg.get("native_efficiency", 1.0)

    rows = []
    for grading, jobs in variants.items():
        for load in scfg["loads"]:
            for seed in scfg["seeds"]:
                rng = np.random.default_rng(seed)
                sample = [jobs[i] for i in rng.integers(0, len(jobs), scfg["n_queries"])]
                arrivals = poisson_arrivals(len(sample), load / mean_work, rng)
                for policy in scfg["policies"]:
                    if grading == "par" and policy not in ("native", "fifo_call"):
                        continue  # 병렬 grading 은 gate 가 없을 때만 의미가 있음
                    effs = [eff, 1.0] if policy == "native" and eff != 1.0 else [eff]
                    for e in effs:
                        runs = simulate(
                            sample, arrivals, policy, model, scfg["native_slots"], efficiency=e
                        )
                        rows.append(
                            {
                                "grading": grading,
                                "load": load,
                                "seed": seed,
                                "policy": policy,
                                "efficiency": e if policy == "native" else 1.0,
                                **summarize(runs, scfg["warmup_frac"]),
                            }
                        )
        print(f"{grading} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "sim_results.csv", index=False)

    main_df = df[(df.efficiency == eff) | (df.policy != "native")]
    agg = main_df.groupby(["grading", "policy", "load"]).mean(numeric_only=True).reset_index()
    for metric in ("mean", "p95", "p95_corrected"):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for policy in scfg["policies"]:
            d = agg[(agg.grading == "seq") & (agg.policy == policy)]
            ax.plot(d.load, d[metric], marker="o", label=POLICY_LABEL[policy])
        ax.set_xlabel("load ρ")
        ax.set_ylabel(f"E2E {metric} (s)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"policy_{metric}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for grading in ("seq", "par"):
        for policy in ("native", "fifo_call"):
            d = agg[(agg.grading == grading) & (agg.policy == policy)]
            ax.plot(d.load, d["mean"], marker="o", label=f"{grading} grading / {policy}")
    ax.set_xlabel("load ρ")
    ax.set_ylabel("E2E mean (s)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "grading_parallel_vs_load.png", dpi=150)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
