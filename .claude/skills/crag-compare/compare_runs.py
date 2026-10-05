"""C-RAG latency 실험의 두 run 폴더를 비교한다 (crag-compare 스킬용).

    uv run python .claude/skills/crag-compare/compare_runs.py <run_A> <run_B>

A = 기준, B = 비교 대상. 결과는 stdout 과 <run_B>/compare_vs_<A>.md 에 저장.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from rag.pipelines.crag import STAGES

QUALITY = ["acc", "em", "f1"]
T_COLS = [f"t_{s}" for s in STAGES]
ACTIONS = ["correct", "ambiguous", "incorrect"]
MIN_COMMON = 30


def load(run_dir: Path) -> pd.DataFrame:
    df = pd.read_json(run_dir / "results.jsonl", lines=True)
    stages = pd.json_normalize(df["stage_latencies"]).add_prefix("t_")
    df = pd.concat([df.drop(columns="stage_latencies"), stages], axis=1)
    for c in T_COLS:
        df[c] = df[c].fillna(0.0) if c in df else 0.0
    df["query_id"] = df["query_id"].astype(str)
    # repeats > 1 이면 query 단위 평균으로 접는다
    num = ["e2e_latency", *T_COLS, *[q for q in QUALITY if q in df]]
    agg = df.groupby(["query_id", "pipeline"])[num].mean()
    agg["corrective_action"] = df.groupby(["query_id", "pipeline"])["corrective_action"].first()
    return agg.reset_index()


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= flatten(v, f"{key}.")
        else:
            out[key] = v
    return out


def config_diff(a: Path, b: Path) -> pd.DataFrame:
    def read(p: Path) -> dict:
        f = p / "config.yaml"
        return flatten(yaml.safe_load(f.read_text(encoding="utf-8"))) if f.exists() else {}

    ca, cb = read(a), read(b)
    keys = sorted(k for k in ca.keys() | cb.keys() if ca.get(k) != cb.get(k))
    return pd.DataFrame(
        {"A": [ca.get(k) for k in keys], "B": [cb.get(k) for k in keys]},
        index=pd.Index(keys, name="key"),
    )


def pipeline_summary(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    def summ(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("pipeline")
        out = pd.DataFrame(
            {
                "n": g.size(),
                "e2e_mean": g["e2e_latency"].mean(),
                "e2e_p50": g["e2e_latency"].median(),
                "e2e_p90": g["e2e_latency"].quantile(0.9),
            }
        )
        for q in QUALITY:
            if q in df:
                out[q] = g[q].mean()
        return out

    sa, sb = summ(a), summ(b)
    rows = []
    for p in sorted(set(sa.index) | set(sb.index)):
        for label, s in (("A", sa), ("B", sb)):
            if p in s.index:
                rows.append({"pipeline": p, "run": label, **s.loc[p].to_dict()})
        if p in sa.index and p in sb.index:
            rows.append({"pipeline": p, "run": "Δ(B−A)", **(sb.loc[p] - sa.loc[p]).to_dict()})
    return pd.DataFrame(rows).set_index(["pipeline", "run"])


def stage_diff(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    ma = a[a.pipeline == "crag"][["e2e_latency", *T_COLS]].mean()
    mb = b[b.pipeline == "crag"][["e2e_latency", *T_COLS]].mean()
    out = pd.DataFrame({"A_s": ma, "B_s": mb, "Δ_s": mb - ma, "Δ%": (mb - ma) / ma * 100})
    out.index = pd.Index(["e2e", *STAGES], name="stage")
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = a.merge(b, on=["query_id", "pipeline"], suffixes=("_a", "_b"))
    m["d"] = m["e2e_latency_b"] - m["e2e_latency_a"]
    g = m.groupby("pipeline")
    out = pd.DataFrame(
        {
            "n_common": g.size(),
            "Δe2e_mean": g["d"].mean(),
            "Δe2e_median": g["d"].median(),
            "Δe2e_median_%": g.apply(
                lambda x: x["d"].median() / x["e2e_latency_a"].median() * 100,
                include_groups=False,
            ),
            "B_faster_ratio": g["d"].apply(lambda d: (d < 0).mean()),
        }
    )
    for q in QUALITY:
        if f"{q}_a" in m and f"{q}_b" in m:
            out[f"Δ{q}"] = g[f"{q}_b"].mean() - g[f"{q}_a"].mean()
    return out, m


def action_dist(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for label, df in (("A", a), ("B", b)):
        c = df[df.pipeline == "crag"]
        if len(c) == 0:
            continue
        share = c["corrective_action"].value_counts(normalize=True)
        e2e = c.groupby("corrective_action")["e2e_latency"].mean()
        for act in ACTIONS:
            rows.setdefault(act, {})[f"{label}_share"] = share.get(act, 0.0)
            rows[act][f"{label}_e2e"] = e2e.get(act)
    return pd.DataFrame(rows).T.rename_axis("action")


def action_transition(m: pd.DataFrame) -> pd.DataFrame:
    c = m[m.pipeline == "crag"]
    if len(c) == 0:
        return pd.DataFrame()
    t = pd.crosstab(
        c["corrective_action_a"].rename("A \\ B"), c["corrective_action_b"], dropna=False
    )
    return t.reindex(
        index=[x for x in ACTIONS if x in t.index], columns=[x for x in ACTIONS if x in t.columns]
    )


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "(없음)"
    df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def fmt(v, col):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if col.startswith("n") and isinstance(v, (int, float)):
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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path, help="기준 run")
    parser.add_argument("run_b", type=Path, help="비교 대상 run")
    args = parser.parse_args()

    a, b = load(args.run_a), load(args.run_b)
    pt, merged = paired(a, b)

    s = [f"# Run 비교: A={args.run_a.name} → B={args.run_b.name}", ""]
    n_common = int(pt["n_common"].min()) if len(pt) else 0
    if n_common < MIN_COMMON:
        s += [
            f"> ⚠ 공통 query 가 적음 (pipeline 당 최소 {n_common}개). paired 결과 위주로 해석.",
            "",
        ]
    s += ["## 1. config 차이", "", to_markdown(config_diff(args.run_a, args.run_b)), ""]
    s += ["## 2. pipeline 요약 (각 run 전체)", "", to_markdown(pipeline_summary(a, b)), ""]
    s += ["## 3. C-RAG stage 별 평균 latency", "", to_markdown(stage_diff(a, b)), ""]
    s += ["## 4. Paired 비교 (공통 query, B−A)", "", to_markdown(pt), ""]
    s += ["## 5. C-RAG corrective action 분포", "", to_markdown(action_dist(a, b)), ""]
    s += ["## 6. Action 전이 (공통 query, 행=A / 열=B)", "", to_markdown(action_transition(merged))]
    report = "\n".join(s) + "\n"

    out = args.run_b / f"compare_vs_{args.run_a.name}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
