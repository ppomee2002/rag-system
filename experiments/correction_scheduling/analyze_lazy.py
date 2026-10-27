"""run_lazy.py 결과 분석: 품질 (paired bootstrap), GPU 작업량, 같은 도착률에서의 부하 지연.

    uv run python experiments/correction_scheduling/analyze_lazy.py <run_dir>

generation 시간은 측정값 대신 cost model 추정치 (실험 1 의 cache 없는 generate 호출로 fit,
prompt 길이 기반) 를 쓴다. Ollama prompt cache 가 여러 prompt 를 기억해서, 같은 문서로 시작하는
prompt 는 실제보다 빠르게 측정되기 때문.
출력: <run_dir>/lazy/report_lazy.md, lazy_sim.csv
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crag_latency"))
from analyze import paired_bootstrap, to_markdown  # noqa: E402  (crag_latency/analyze.py)
from scheduling import (  # noqa: E402
    Call,
    attach_oof_predictions,
    fit_cost_model,
    job_from_row,
    poisson_arrivals,
    simulate,
    summarize,
)

from rag.config import load_config  # noqa: E402

VARIANTS = ("full", "lazy_hit", "lazy_rest")


def lazy_job(job, lz: dict, gen_service: float):
    """full C-RAG job 에서 hit 이후 grade 를 빼고 generation 서비스 시간을 바꾼다."""
    j = copy.deepcopy(job)
    if lz["hit"] is None:
        return j
    phases, n_grade = [], 0
    for kind, val in j.phases:
        if kind == "gpu" and val[0].kind == "grade":
            n_grade += 1
            if n_grade > lz["n_grade"]:
                continue
        if kind == "gpu" and val[0].kind == "generate":
            val = [Call("generate", 0, lz["gen_chars"], gen_service)]
        phases.append((kind, val))
    j.phases = phases
    j.n_docs = lz["n_grade"]
    j.work = sum(c.service for k, cs in phases if k == "gpu" for c in cs)
    j.solo = sum(v if k == "delay" else sum(c.service for c in v) for k, v in phases)
    return j


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", default="experiments/correction_scheduling/config.yaml")
    parser.add_argument("--loads", nargs="+", type=float, default=[0.5, 0.8, 0.9])
    args = parser.parse_args()
    cfg = load_config(args.config)
    scfg = cfg["sim"]
    out = args.run_dir / "lazy"
    lz = pd.DataFrame([json.loads(x) for x in (out / "lazy.jsonl").open(encoding="utf-8")])
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    rows = [json.loads(x) for x in (args.run_dir / "profile.jsonl").open(encoding="utf-8")]
    jobs = {j.qid: j for j in (job_from_row(r) for r in rows if r["pipeline"] == "crag")}
    model = fit_cost_model(list(jobs.values()))
    a, b = model.gen_coef
    lz["gen_est"] = np.where(lz.reused & (lz.variant != "full"), lz.gen_work, a + b * lz.gen_chars)
    lz.loc[lz.variant == "full", "gen_est"] = lz.loc[lz.variant == "full", "gen_work"]
    # correction 된 query (hit 없음) 는 rewrite 포함 post 작업량 그대로
    lz["post_est"] = np.where(lz.hit.isna(), lz.post_work, lz.gen_est)
    lz["gpu_work_est"] = lz.grade_work + lz.post_est
    lz["pipeline"], lz["repeat"] = lz.variant, 0

    # ---- 품질
    q = (
        lz.groupby("variant")[
            ["acc", "em", "f1", "n_grade", "grade_work", "post_est", "gpu_work_est"]
        ]
        .mean()
        .loc[list(VARIANTS)]
    )
    q["n_ctx"] = lz.groupby("variant").contexts.apply(lambda s: s.map(len).mean())
    q["regenerated"] = lz.groupby("variant").reused.apply(lambda s: int((~s).sum()))
    cmp = [
        to_markdown(paired_bootstrap(lz, "full", v, ["acc", "f1", "gpu_work_est"]), 3)
        for v in VARIANTS[1:]
    ]
    changed = lz[(lz.variant != "full") & ~lz.reused]
    full_acc = lz[lz.variant == "full"].set_index("query_id").acc
    flips = []
    for v in VARIANTS[1:]:
        c = changed[changed.variant == v].set_index("query_id")
        base = full_acc.loc[c.index]
        flips.append(
            {
                "variant": v,
                "n_changed_ctx": len(c),
                "full_acc": base.mean(),
                "variant_acc": c.acc.mean(),
                "1->0": int(((base == 1) & (c.acc == 0)).sum()),
                "0->1": int(((base == 0) & (c.acc == 1)).sum()),
            }
        )
    flips = pd.DataFrame(flips).set_index("variant")

    # ---- 같은 도착률에서의 부하 지연 (도착률은 full 기준으로 고정)
    variant_jobs = {"full": list(jobs.values())}
    for v in VARIANTS[1:]:
        d = lz[lz.variant == v].set_index("query_id")
        variant_jobs[v] = [
            lazy_job(
                jobs[qid],
                d.loc[qid].to_dict()
                | {"hit": (None if pd.isna(d.loc[qid, "hit"]) else int(d.loc[qid, "hit"]))},
                float(d.loc[qid, "post_est"]),
            )
            for qid in jobs
        ]
    for js in variant_jobs.values():
        attach_oof_predictions(js)
    mean_work_full = float(np.mean([j.work for j in variant_jobs["full"]]))
    sim_rows = []
    for load in args.loads:
        for seed in scfg["seeds"]:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(jobs), scfg["n_queries"])
            arrivals = poisson_arrivals(len(idx), load / mean_work_full, rng)
            for v, js in variant_jobs.items():
                vm = fit_cost_model(js)
                for policy in ("native", "fifo_query", "srpt_est"):
                    runs = simulate(
                        [js[i] for i in idx],
                        arrivals,
                        policy,
                        vm,
                        scfg["native_slots"],
                        efficiency=1.0,
                    )
                    sim_rows.append(
                        {
                            "variant": v,
                            "policy": policy,
                            "load_full": load,
                            "seed": seed,
                            **summarize(runs, scfg["warmup_frac"]),
                        }
                    )
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(out / "lazy_sim.csv", index=False)
    simt = sim.groupby(["load_full", "policy", "variant"])[["mean", "p95", "mean_corrected"]].mean()
    eff_load = {
        v: float(np.mean([j.work for j in js])) / mean_work_full for v, js in variant_jobs.items()
    }

    report = f"""# Lazy grading 품질 / 지연

retrieval 순서로 grade 하다 score >= {summ["upper"]} 가 나오면 나머지 grade 를 생략한다.
PopQA long-tail {summ["n_queries"]} query, llama3.1:8b. grade score 는 실험 1 값을 재사용하고,
context 가 바뀐 query 만 generation 을 새로 했다.

- `lazy_hit`: 처음 기준을 넘은 문서 1개만 context 로 사용
- `lazy_rest`: 그 문서 + 아직 grade 하지 않은 뒷 문서

답 재사용 검증: 실험 1 full context 를 그대로 다시 generate 한
{summ["determinism_n"]}개 중 {summ["determinism_same"]}개가 동일.

## 품질 / 작업량 (query 평균, GPU 시간은 초)

{to_markdown(q, 3)}

generation 시간은 측정값이 아니라 prompt 길이 기반 cost model 추정치다.
Ollama prompt cache 때문에 측정값은 실제보다 작게 나온다.

## full 대비 paired bootstrap 95% CI

{chr(10).join(cmp)}

## context 가 바뀐 query 만

{to_markdown(flips, 3)}

## 같은 도착률에서의 부하 지연 (simulation, 도착률은 full 기준 rho 로 고정)

variant 별 실효 부하 (full 대비 GPU 작업량 비율):
{", ".join(f"{k} {v:.2f}" for k, v in eff_load.items())}

{to_markdown(simt, 1)}
"""
    (out / "report_lazy.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
