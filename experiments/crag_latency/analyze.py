"""run.py 결과(results.jsonl)를 집계해 표 / 그래프 / report.md 를 만든다.

    uv run python experiments/crag_latency/analyze.py results/crag_latency/<run_id>

출력 (run_dir 아래):
  report.md           결과 정리 (표 + 그래프 링크)
  figures/*.png       E2E 비교 / component 구성 / 난이도별 latency
  results_flat.csv    query 단위 flat 테이블 (난이도 라벨 포함)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rag.pipelines.crag import STAGES

QUALITY = ["acc", "em", "f1"]
T_COLS = [f"t_{s}" for s in STAGES]
ACTIONS = ["correct", "ambiguous", "incorrect"]

# 난이도 기준: (컬럼, 설명). 라벨은 C-RAG 행에서 계산해 query_id 로 vanilla 행에도 붙인다.
DIFFICULTY = {
    "diff_action": "corrective action (correct=easy, ambiguous/incorrect=hard)",
    "diff_grader": "평균 grader score 3분위 (high=easy)",
    "diff_retrieval": "최초 top-1 retrieval score 3분위 (high=easy)",
}
LEVELS = ["easy", "medium", "hard"]
PANEL_TITLES = {
    "diff_action": "by corrective action",
    "diff_grader": "by mean grader score",
    "diff_retrieval": "by top-1 retrieval score",
}

# dataviz 기본 팔레트 (categorical slot 1-5, light)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(run_dir: Path) -> pd.DataFrame:
    df = pd.read_json(run_dir / "results.jsonl", lines=True)
    stages = pd.json_normalize(df["stage_latencies"]).add_prefix("t_")
    df = pd.concat([df.drop(columns="stage_latencies"), stages], axis=1)
    for c in T_COLS:
        df[c] = df[c].fillna(0.0) if c in df else 0.0
    retrieval = df["extra"].map(lambda e: (e or {}).get("retrieval_scores") or [None])
    df["retrieval_top1"] = retrieval.map(lambda s: s[0])
    df["max_grader_score"] = df["grader_scores"].map(lambda s: max(s) if s else None)
    df["mean_grader_score"] = df["grader_scores"].map(lambda s: sum(s) / len(s) if s else None)
    df["is_corrective"] = df["n_corrective_steps"] > 0
    return df


def _tertile(s: pd.Series) -> pd.Series:
    """값이 높을수록 easy. 동점이 많아 3구간이 안 나오면 가능한 만큼만 나눈다."""
    ranks = s.rank(method="first", pct=True)
    return pd.cut(ranks, [0, 1 / 3, 2 / 3, 1], labels=["hard", "medium", "easy"]).astype(str)


def add_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    crag = df[df.pipeline == "crag"].drop_duplicates("query_id").set_index("query_id")
    labels = {
        "diff_action": crag["corrective_action"].map(
            {"correct": "easy", "ambiguous": "hard", "incorrect": "hard"}
        ),
        "diff_grader": _tertile(crag["mean_grader_score"]),
        "diff_retrieval": _tertile(crag["retrieval_top1"]),
    }
    for col, label in labels.items():
        df[col] = df["query_id"].map(label)
    return df


# ---------------------------------------------------------------- tables


def pipeline_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("pipeline")
    out = pd.DataFrame(
        {
            "n": g.size(),
            "e2e_mean_s": g["e2e_latency"].mean(),
            "e2e_p50_s": g["e2e_latency"].median(),
            "e2e_p90_s": g["e2e_latency"].quantile(0.9),
            **{q: g[q].mean() for q in QUALITY},
        }
    )
    for p in ("crag", "crag_parallel"):
        if {"vanilla", p} <= set(out.index):
            out.loc[f"{p} / vanilla"] = out.loc[p] / out.loc["vanilla"]
            out.loc[f"{p} / vanilla", "n"] = None
    return out


def _ci_row(name: str, point: float, boots: np.ndarray) -> dict:
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"metric": name, "estimate": point, "ci95_low": lo, "ci95_high": hi}


def paired_bootstrap(
    df: pd.DataFrame, a: str, b: str, cols: list[str], n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """같은 query 끼리 짝지어 resample. latency 는 b/a 배율과 차이, 품질은 b-a 차이의 95% CI."""
    p = (
        df[df.pipeline.isin([a, b])]
        .pivot_table(index=["query_id", "repeat"], columns="pipeline", values=cols)
        .dropna()
    )
    idx = np.random.default_rng(seed).integers(0, len(p), (n_boot, len(p)))
    rows = []
    for col in cols:
        x, y = p[(col, a)].to_numpy(), p[(col, b)].to_numpy()
        diff = (y.mean() - x.mean(), y[idx].mean(1) - x[idx].mean(1))
        if col in QUALITY:
            rows.append(_ci_row(f"{col}: {b} - {a}", *diff))
        else:
            ratio = (y.mean() / x.mean(), y[idx].mean(1) / x[idx].mean(1))
            rows.append(_ci_row(f"{col}: {b} / {a}", *ratio))
            rows.append(_ci_row(f"{col}: {b} - {a} (s)", *diff))
    return pd.DataFrame(rows).set_index("metric")


def corrective_effect(crag: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """corrective 발생 query 와 미발생 query 의 평균 차이 (비짝지음 bootstrap)."""
    rng = np.random.default_rng(seed)
    yes, no = crag[crag.is_corrective], crag[~crag.is_corrective]
    rows = []
    for col in ["e2e_latency", "acc"]:
        y, n = yes[col].to_numpy(), no[col].to_numpy()
        boots = y[rng.integers(0, len(y), (n_boot, len(y)))].mean(1) - n[
            rng.integers(0, len(n), (n_boot, len(n)))
        ].mean(1)
        rows.append(_ci_row(f"{col}: corrective - non-corrective", y.mean() - n.mean(), boots))
    return pd.DataFrame(rows).set_index("metric")


def parallel_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """순차 grading (crag) vs 병렬 grading (crag_parallel)."""
    sub = df[df.pipeline.isin(["vanilla", "crag", "crag_parallel"])]
    g = sub.groupby("pipeline")
    means = g[["e2e_latency", *T_COLS, *QUALITY]].mean()
    means.insert(0, "n", g.size())
    ci = paired_bootstrap(df, "crag", "crag_parallel", ["e2e_latency", "t_grading", *QUALITY])
    key = ["query_id", "repeat"]
    m = df[df.pipeline == "crag"].merge(df[df.pipeline == "crag_parallel"], on=key)
    agree = pd.DataFrame(
        {
            "same_grader_scores": [(m.grader_scores_x == m.grader_scores_y).mean()],
            "same_corrective_action": [(m.corrective_action_x == m.corrective_action_y).mean()],
            "same_answer": [(m.answer_x == m.answer_y).mean()],
        },
        index=pd.Index([len(m)], name="n_pairs"),
    )
    return means, ci, agree


def component_share(crag: pd.DataFrame) -> pd.DataFrame:
    mean = crag[T_COLS].mean()
    out = pd.DataFrame({"mean_s": mean, "share_of_e2e": mean / crag["e2e_latency"].mean()})
    out.index = pd.Index(list(STAGES), name="stage")
    out.loc["(untracked)"] = [
        crag["e2e_latency"].mean() - mean.sum(),
        1 - mean.sum() / crag["e2e_latency"].mean(),
    ]
    return out


def by_group(crag: pd.DataFrame, col: str) -> pd.DataFrame:
    g = crag.groupby(col)
    out = g[["e2e_latency", *T_COLS, "n_corrective_steps", *QUALITY]].mean()
    out.insert(0, "n", g.size())
    out.insert(1, "share", g.size() / len(crag))
    return out


def difficulty_x_pipeline(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby([col, "pipeline"])
    out = g[["e2e_latency", *QUALITY]].mean()
    out.insert(0, "n", g.size())
    return out.reindex([lv for lv in LEVELS if lv in df[col].unique()], level=0)


def action_mix(crag: pd.DataFrame, col: str) -> pd.DataFrame:
    """난이도 구간별 corrective action 비율 = query 특성에 따른 처리 경로 변화."""
    mix = pd.crosstab(crag[col], crag["corrective_action"], normalize="index")
    return mix.reindex(columns=[a for a in ACTIONS if a in mix.columns]).reindex(
        [lv for lv in LEVELS if lv in mix.index]
    )


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def fmt(v, col):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if col == "n":
            return str(int(v))
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += [
        "| " + " | ".join(fmt(v, c) for v, c in zip(row, cols, strict=True)) + " |"
        for row in df.itertuples(False)
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- figures


def _style(ax, title: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=28)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def plot_components(df: pd.DataFrame, path: Path) -> None:
    """vanilla / C-RAG 전체 / action 별 평균 component latency 누적 막대."""
    import matplotlib.pyplot as plt

    crag, vanilla = df[df.pipeline == "crag"], df[df.pipeline == "vanilla"]
    rows = {"Vanilla": vanilla, "C-RAG (all)": crag}
    rows["C-RAG parallel grading"] = df[df.pipeline == "crag_parallel"]
    rows |= {f"C-RAG: {a}": crag[crag.corrective_action == a] for a in ACTIONS}
    rows = {k: v for k, v in rows.items() if len(v)}
    labels = [f"{k}  (n={len(v)})" for k, v in rows.items()][::-1]
    means = [v[T_COLS].mean() for v in rows.values()][::-1]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.6), facecolor=SURFACE)
    left = [0.0] * len(rows)
    for i, (stage, color) in enumerate(zip(STAGES, SERIES, strict=True)):
        vals = [m[T_COLS[i]] for m in means]
        ax.barh(
            labels,
            vals,
            left=left,
            color=color,
            height=0.55,
            label=stage,
            edgecolor=SURFACE,
            linewidth=2,
        )
        left = [a + b for a, b in zip(left, vals, strict=True)]
    for y, total in enumerate(left):
        ax.text(total, y, f"  {total:.2f}s", va="center", color=INK, fontsize=9)
    _style(ax, "Mean latency by component", "seconds")
    ax.set_xlim(0, max(left) * 1.15)
    ax.legend(
        ncol=5,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        handlelength=1,
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_difficulty(df: pd.DataFrame, path: Path) -> None:
    """난이도 기준별 small multiples: 구간 x pipeline 평균 E2E."""
    import matplotlib.pyplot as plt

    pipes = [p for p in ("vanilla", "crag") if p in df.pipeline.unique()]
    fig, axes = plt.subplots(1, len(DIFFICULTY), figsize=(11, 3.4), facecolor=SURFACE, sharex=True)
    xmax = 0.0
    for ax, col in zip(axes, DIFFICULTY, strict=True):
        levels = [lv for lv in LEVELS if lv in df[col].unique()][::-1]
        h = 0.8 / len(pipes)
        for j, (p, color) in enumerate(zip(pipes, SERIES, strict=False)):
            vals = [df[(df[col] == lv) & (df.pipeline == p)]["e2e_latency"].mean() for lv in levels]
            ys = [i - (j - (len(pipes) - 1) / 2) * h for i in range(len(levels))]
            ax.barh(ys, vals, height=h, color=color, label=p, edgecolor=SURFACE, linewidth=2)
            xmax = max(xmax, *vals)
        ax.set_yticks(range(len(levels)), levels)
        _style(ax, PANEL_TITLES[col], "mean E2E latency (s)")
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        handlelength=1,
        borderaxespad=0,
    )
    axes[0].set_xlim(0, xmax * 1.1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report


def build_report(df: pd.DataFrame, run_dir: Path) -> str:
    crag = df[df.pipeline == "crag"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_components(df, fig_dir / "components.png")
    plot_difficulty(df, fig_dir / "difficulty.png")

    s = [f"# C-RAG latency 결과 ({run_dir.name})", ""]
    s += [
        "## 1. Vanilla RAG vs C-RAG: E2E latency / 답변 품질",
        "",
        to_markdown(pipeline_summary(df)),
        "",
    ]
    s += [
        "## 2. C-RAG component 별 평균 latency 와 E2E 대비 비율",
        "",
        to_markdown(component_share(crag)),
        "",
        "![components](figures/components.png)",
        "",
    ]
    s += [
        "## 3. Corrective action 발생 여부별",
        "",
        "### corrective 단계 수행 (rewrite + web search) 여부",
        "",
        to_markdown(by_group(crag, "is_corrective")),
        "",
        "### corrective action 별",
        "",
        to_markdown(by_group(crag, "corrective_action")),
        "",
    ]
    s += ["## 4. Query 난이도별 비교", "", "![difficulty](figures/difficulty.png)", ""]
    for col, desc in DIFFICULTY.items():
        s += [
            f"### 기준: {desc}",
            "",
            "E2E / 품질 (pipeline 별)",
            "",
            to_markdown(difficulty_x_pipeline(df, col)),
            "",
            "C-RAG component latency / corrective 횟수",
            "",
            to_markdown(
                by_group(crag, col).reindex([lv for lv in LEVELS if lv in crag[col].unique()])
            ),
            "",
        ]
        if col != "diff_action":
            s += ["처리 경로 (corrective action 비율)", "", to_markdown(action_mix(crag, col)), ""]

    s += [
        "## 5. 95% 신뢰구간 (bootstrap 2,000회)",
        "",
        "### Vanilla vs C-RAG (같은 query 끼리 짝지은 paired bootstrap)",
        "",
        to_markdown(paired_bootstrap(df, "vanilla", "crag", ["e2e_latency", *QUALITY])),
        "",
        "### Corrective action 발생 효과 (C-RAG)",
        "",
        to_markdown(corrective_effect(crag)),
        "",
    ]
    if "crag_parallel" in set(df.pipeline):
        means, ci, agree = parallel_comparison(df)
        s += [
            "## 6. Grader 순차 vs 병렬 실행",
            "",
            "문서별 grader 호출을 동시에 실행 (crag_parallel). 같은 run 에서 같은 query 로 비교.",
            "",
            to_markdown(means),
            "",
            "### 순차 대비 병렬 (paired bootstrap 95% CI)",
            "",
            to_markdown(ci),
            "",
            "### 결과 일치율 (같은 query)",
            "",
            to_markdown(agree),
            "",
        ]
    return "\n".join(s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    df = add_difficulty(load(args.run_dir))
    report = build_report(df, args.run_dir)
    (args.run_dir / "report.md").write_text(report, encoding="utf-8")
    df.drop(columns=["contexts", "extra"], errors="ignore").to_csv(
        args.run_dir / "results_flat.csv", index=False
    )
    print(report)
    print(f"\nsaved -> {args.run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
