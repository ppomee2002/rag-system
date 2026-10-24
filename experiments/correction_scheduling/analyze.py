"""실험 1~4 결과를 모아 report.md 작성. 실험 2 (Workload Analysis) 는 여기서 계산한다.

    uv run python experiments/correction_scheduling/analyze.py <run_dir>

입력: profile.jsonl (실험 1), sim/ (simulate.py), live_*/ (run_live.py, 있으면)
출력: <run_dir>/report.md, figures/*.png
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
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crag_latency"))
from analyze import paired_bootstrap, to_markdown  # noqa: E402  (crag_latency/analyze.py)
from scheduling import (  # noqa: E402
    FEATURE_NAMES,
    attach_oof_predictions,
    features,
    job_from_row,
    post_work,
)

from rag.pipelines.crag import STAGES  # noqa: E402

HAIKU_RUN = Path("results/crag_latency/20261006_113027")  # 같은 query 셋, claude-haiku-4-5 grader


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """label=1 을 score 가 클수록 잘 고르는 정도 (Mann-Whitney, 동점은 평균 rank)."""
    r = pd.Series(score).rank().to_numpy()
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_rows(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "profile.jsonl").open(encoding="utf-8")]


def stage_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for p, g in df.groupby("pipeline"):
        row = {
            "pipeline": p,
            "n": len(g),
            "e2e_mean": g.e2e.mean(),
            "e2e_p50": g.e2e.median(),
            "e2e_p95": g.e2e.quantile(0.95),
        }
        for s in STAGES:
            row[s] = g[s].mean()
        row["acc"] = g.acc.mean()
        out.append(row)
    return pd.DataFrame(out).set_index("pipeline")


def call_table(rows: list[dict]) -> pd.DataFrame:
    calls = pd.DataFrame([{**c, "pipeline": r["pipeline"]} for r in rows for c in r["calls"]])
    calls["overhead"] = calls.wall - calls.server_total
    g = calls.groupby(["pipeline", "kind"])
    n_q = pd.Series({p: sum(r["pipeline"] == p for r in rows) for p in calls.pipeline.unique()})
    t = pd.DataFrame(
        {
            "calls_per_query": g.size()
            / n_q.reindex(g.size().index.get_level_values(0)).to_numpy(),
            "prompt_tokens": g.prompt_tokens.mean(),
            "out_tokens": g.out_tokens.mean(),
            "server_s": g.server_total.mean(),
            "prompt_eval_s": g.server_prompt_eval.mean(),
            "decode_s": g.server_eval.mean(),
            "overhead_s": g.overhead.mean(),
        }
    )
    return t


def lazy_grading(rows: list[dict], upper: float) -> dict:
    """retrieval 순서로 grade 하다 upper 이상이 나오면 멈출 때 줄어드는 grade 호출 수."""
    saved, total, n_corr = 0, 0, 0
    for r in rows:
        s = r["grader_scores"]
        total += len(s)
        hit = next((i for i, v in enumerate(s) if v >= upper), None)
        if hit is not None:
            saved += len(s) - hit - 1
            n_corr += 1
    return {
        "grade_calls": total,
        "saved": saved,
        "saved_frac": saved / total if total else 0.0,
        "queries_with_hit": n_corr,
    }


def plot_stage_share(st: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    left = np.zeros(len(st))
    for s in STAGES:
        ax.barh(st.index, st[s], left=left, label=s)
        left += st[s].to_numpy()
    ax.set_xlabel("mean latency (s)")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_work_hist(jobs, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, max(j.work for j in jobs) * 1.02, 40)
    for name, sel in (
        ("correct", lambda j: j.action == "correct"),
        ("corrected (ambiguous/incorrect)", lambda j: j.action != "correct"),
    ):
        ax.hist([j.work for j in jobs if sel(j)], bins=bins, alpha=0.6, label=name)
    ax.set_xlabel("GPU work per query (s)")
    ax.set_ylabel("queries")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    (run_dir / "figures").mkdir(exist_ok=True)
    rows = load_rows(run_dir)
    crag_rows = [r for r in rows if r["pipeline"] == "crag"]

    df = pd.DataFrame(
        [
            {
                "query_id": r["query_id"],
                "pipeline": r["pipeline"],
                "repeat": 0,
                "e2e": r["e2e_latency"],
                "acc": r["acc"],
                "action": r["corrective_action"],
                **{s: r["stage_latencies"].get(s, 0.0) for s in STAGES},
            }
            for r in rows
        ]
    )
    st = stage_table(df)
    plot_stage_share(st, run_dir / "figures" / "stage_latency.png")
    ct = call_table(rows)

    subset_ids = set(df[df.pipeline == "crag_parallel"].query_id)
    sub = df[df.query_id.isin(subset_ids)]
    cmp_txt = []
    for a, b, cols in (
        ("crag", "crag_parallel", ["e2e", "grading"]),
        ("vanilla", "crag", ["e2e", "acc"]),
    ):
        if {a, b} <= set(sub.pipeline):
            cmp_txt.append(to_markdown(paired_bootstrap(sub, a, b, cols), 3))
    cmp_txt = "\n\n".join(cmp_txt) or "(subset pipeline 미실행)"

    # ---- 실험 2: workload
    jobs = [job_from_row(r) for r in crag_rows]
    attach_oof_predictions(jobs)
    work = np.array([j.work for j in jobs])
    corr = np.array([j.action != "correct" for j in jobs])
    grade_work = np.array(
        [
            sum(c.service for k, cs in j.phases if k == "gpu" for c in cs if c.kind == "grade")
            for j in jobs
        ]
    )
    post = np.array([post_work(j) for j in jobs])
    by_action = pd.DataFrame(
        [
            {
                "action": a,
                "n": int(sum(j.action == a for j in jobs)),
                "share": float(np.mean([j.action == a for j in jobs])),
                "e2e_solo": float(np.mean([j.solo for j in jobs if j.action == a])),
                "gpu_work": float(np.mean([j.work for j in jobs if j.action == a])),
                "grading_work": float(np.mean(grade_work[[j.action == a for j in jobs]])),
                "post_grading_work": float(np.mean(post[[j.action == a for j in jobs]])),
                "acc": float(np.mean([j.acc for j in jobs if j.action == a])),
            }
            for a in ("correct", "ambiguous", "incorrect")
            if any(j.action == a for j in jobs)
        ]
    ).set_index("action")
    plot_work_hist(jobs, run_dir / "figures" / "work_hist.png")

    F = np.array([j.feats for j in jobs])
    feat_rows = []
    for i, name in enumerate(FEATURE_NAMES[1:], start=1):
        feat_rows.append(
            {
                "feature": name,
                "AUC(corrected)": auc(F[:, i], corr),
                "spearman(work)": pd.Series(F[:, i]).corr(pd.Series(work), method="spearman"),
            }
        )
    pred = np.array([j.pred_total for j in jobs])
    feat_rows.append(
        {
            "feature": "OOF 회귀 예측 (pred_total)",
            "AUC(corrected)": auc(pred, corr),
            "spearman(work)": pd.Series(pred).corr(pd.Series(work), method="spearman"),
        }
    )
    feat_tbl = pd.DataFrame(feat_rows).set_index("feature")
    r2 = 1 - np.sum((work - pred) ** 2) / np.sum((work - work.mean()) ** 2)

    haiku_txt = "(Haiku run 없음)"
    hp = HAIKU_RUN / "results.jsonl"
    if hp.exists():
        h = [json.loads(line) for line in hp.open(encoding="utf-8")]
        h = {r["query_id"]: r for r in h if r["pipeline"] == "crag"}
        both = [(j, h[j.qid]) for j in jobs if j.qid in h]
        hc = np.array([hr["corrective_action"] != "correct" for _, hr in both])
        hf = np.array([features(hr["extra"]["retrieval_scores"], [0], 0)[1] for _, hr in both])
        lc = np.array([j.action != "correct" for j, _ in both])
        agree = pd.crosstab(
            pd.Series([j.action for j, _ in both], name="llama"),
            pd.Series([hr["corrective_action"] for _, hr in both], name="haiku"),
        )
        haiku_txt = (
            f"같은 {len(both)} query 에서 Haiku grader correction 비율 {hc.mean():.1%} "
            f"(llama {lc.mean():.1%}). top1 retrieval score 의 AUC: Haiku correction "
            f"{auc(-hf, hc):.3f} / llama {auc(-hf, lc):.3f} (score 가 낮을수록 correction)."
            f"\n\nllama vs Haiku action 교차표:\n\n{to_markdown(agree, 0)}"
        )

    lazy = lazy_grading(crag_rows, cfg["crag"]["upper_threshold"])

    # ---- 실험 3·4: simulation
    sim_txt, live_txt = "(simulate.py 미실행)", "(run_live.py 미실행)"
    sp = run_dir / "sim" / "sim_results.csv"
    if sp.exists():
        sim = pd.read_csv(sp)
        cm = json.loads((run_dir / "sim" / "cost_model.json").read_text(encoding="utf-8"))
        eff = sim[sim.policy == "native"].efficiency.min()
        conc = (
            sim[sim.policy.isin(["native", "fifo_call"])]
            .assign(efficiency=lambda d: d.efficiency.map("{:.2f}".format))
            .groupby(["grading", "policy", "efficiency", "load"])[["mean", "p95", "throughput"]]
            .mean()
            .unstack("load")
        )
        conc_mean = conc["mean"]
        conc_p95 = conc["p95"]
        pol = (
            sim[(sim.grading == "seq") & ((sim.policy != "native") | (sim.efficiency == eff))]
            .groupby(["load", "policy"])
            .mean(numeric_only=True)
        )
        pol_cols = [
            "mean",
            "p50",
            "p95",
            "p99",
            "mean_correct",
            "mean_corrected",
            "p95_corrected",
            "mean_slowdown",
        ]
        base = pol.xs("fifo_query", level="policy")["mean"]
        pol["vs_fifo_query"] = pol["mean"] / base.reindex(pol.index.get_level_values(0)).values
        sel = pol.loc[[ld for ld in (0.5, 0.8, 0.9) if ld in pol.index.get_level_values(0)]]
        sim_txt = (
            f"평균 GPU 작업량 {cm['mean_work']:.2f}s / query → 처리 한계 "
            f"{1 / cm['mean_work'] * 60:.2f} query/min. seed {sim.seed.nunique()}개 평균.\n\n"
            f"### 3-1. 부하별 평균 E2E (s) — native(Ollama 병렬 slot) vs gate(1개씩)\n\n"
            f"{to_markdown(conc_mean, 1)}\n\n### 3-2. 부하별 p95 E2E (s)\n\n"
            f"{to_markdown(conc_p95, 1)}\n\n"
            f"![](sim/figures/grading_parallel_vs_load.png)\n\n"
            f"## 4. Scheduling Baseline (simulation, 순차 grading)\n\n"
            f"{to_markdown(sel[pol_cols + ['vs_fifo_query']], 1)}\n\n"
            f"![](sim/figures/policy_mean.png) ![](sim/figures/policy_p95_corrected.png)\n\n"
            f"cost model: {json.dumps(cm, ensure_ascii=False)}"
        )
    lives = sorted(run_dir.glob("live_*/comparison.csv"))
    if lives:
        lv = pd.read_csv(lives[-1])
        lv["err_mean_%"] = (lv.sim_mean / lv.live_mean - 1) * 100
        lv["err_p95_%"] = (lv.sim_p95 / lv.live_p95 - 1) * 100
        lcfg = yaml.safe_load((lives[-1].parent / "config.yaml").read_text(encoding="utf-8"))
        live_txt = (
            f"`{lives[-1].parent.name}`: query {lcfg['live']['n_queries']}개, "
            f"rho={lcfg['live']['load']}, 같은 Poisson 도착 trace 를 정책마다 재생.\n\n"
            f"{to_markdown(lv.set_index('policy'), 2)}"
        )

    n = len(crag_rows)
    report = f"""# Correction-Aware Scheduling 실험 결과

run: `{run_dir.name}` · LLM `{cfg["llm"]["model"]}` (Ollama, {cfg["server"]["backend"]},
NUM_PARALLEL={cfg["server"]["OLLAMA_NUM_PARALLEL"]}) · PopQA long-tail {n} query (warmup 제외)
· web search 는 mock (실측 Tavily latency 재생 + 로컬 FAISS 문서)

## 1. Stage-Level Profiling (부하 없음, query 1개씩)

vanilla / crag_parallel 은 앞 {len(subset_ids)} query 에서만 실행 (pipeline 별 pass 로 분리:
같은 prompt 가 Ollama prompt cache 에 걸리는 것 방지).

{to_markdown(st, 2)}

![](figures/stage_latency.png)

### LLM 호출 단위 (query 당 호출 수, 평균 토큰, 서버 처리 시간)

{to_markdown(ct, 3)}

### 같은 query 끼리 비교 (paired bootstrap 95% CI)

{cmp_txt}

## 2. Workload Analysis

{to_markdown(by_action, 2)}

![](figures/work_hist.png)

- GPU 작업량 분포: 평균 {work.mean():.2f}s, p50 {np.median(work):.2f}s, p95
  {np.percentile(work, 95):.2f}s, CV {work.std() / work.mean():.2f}
- 전체 GPU 작업 중 grading 비중 {grade_work.sum() / work.sum():.1%}, correction 된 query
  (rewrite+generation) 비중 {post[corr].sum() / work.sum():.1%}

### correction / 작업량 예측 가능성 (retrieval 직후 feature)

{to_markdown(feat_tbl, 3)}

OOF 회귀 R² = {r2:.3f}. {haiku_txt}

### lazy grading 잠재 효과

retrieval 순서로 grade 하다가 score >= {cfg["crag"]["upper_threshold"]} 가 나오면 멈추면
grade 호출 {lazy["saved"]}/{lazy["grade_calls"]} ({lazy["saved_frac"]:.1%}) 감소
(해당 query {lazy["queries_with_hit"]}개). 단 context 가 달라지므로 품질은 별도 측정 필요.

## 3. Concurrent Query (simulation)

{sim_txt}

## 5. Simulation 검증 (live 동시 부하)

{live_txt}
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"saved -> {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
