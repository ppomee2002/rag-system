"""run.py 결과(results.jsonl)를 집계.

    uv run python experiments/crag_latency/analyze.py results/crag_latency/<run_id>
"""

import argparse
from pathlib import Path

import pandas as pd

from rag.pipelines.crag import STAGES

QUALITY = ["em", "f1", "acc"]


def load(run_dir: Path) -> pd.DataFrame:
    df = pd.read_json(run_dir / "results.jsonl", lines=True)
    stages = pd.json_normalize(df["stage_latencies"]).add_prefix("t_")
    return pd.concat([df.drop(columns="stage_latencies"), stages], axis=1)


def add_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """C-RAG 의 corrective action 으로 난이도 라벨링 후 vanilla 행에도 query_id 기준으로 붙인다.

    다른 기준(최초 retrieval score, max grader score 등)을 쓰려면 여기만 바꾸면 된다.
    """
    crag = df[df.pipeline == "crag"].drop_duplicates("query_id")
    label = crag.set_index("query_id")["corrective_action"].map(
        {"correct": "easy", "ambiguous": "hard", "incorrect": "hard"}
    )
    df["difficulty"] = df["query_id"].map(label)
    df["max_grader_score"] = df["grader_scores"].map(lambda s: max(s) if s else None)
    return df


def report(df: pd.DataFrame) -> None:
    t_cols = [f"t_{s}" for s in STAGES if f"t_{s}" in df]
    crag = df[df.pipeline == "crag"]

    print("\n## 1. Pipeline 별 E2E latency / 답변 품질")
    print(df.groupby("pipeline")[["e2e_latency", *QUALITY]].agg(["mean", "median"]).round(4))
    print(df.groupby("pipeline")["e2e_latency"].quantile([0.5, 0.9, 0.99]).unstack().round(4))

    print("\n## 2. C-RAG component 별 평균 latency 및 비율")
    mean = crag[t_cols].mean()
    print(pd.DataFrame({"mean_s": mean, "share": mean / crag["e2e_latency"].mean()}).round(4))

    print("\n## 3. Corrective action 별")
    print(
        crag.groupby("corrective_action")[["e2e_latency", *t_cols, "n_corrective_steps", *QUALITY]]
        .mean()
        .assign(n=crag.groupby("corrective_action").size())
        .round(4)
        .T
    )

    print("\n## 4. 난이도 x pipeline")
    print(
        df.groupby(["difficulty", "pipeline"])[["e2e_latency", *QUALITY]]
        .mean()
        .round(4)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    df = add_difficulty(load(args.run_dir))
    report(df)
    df.to_csv(args.run_dir / "results_flat.csv", index=False)


if __name__ == "__main__":
    main()
