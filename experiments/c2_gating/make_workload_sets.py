"""Correct 비율별 질문 집합 (Workload Sensitivity Analysis, 새 데이터셋 일반화가 아님).

MODE A score run 의 실제 판정 분포 (공식 T5, 1,399문항) 에서 Low / Mid / High Correct 비율 집합을
각각 n (기본 200) 문항으로 뽑는다. retrieval·모델·prompt·공식 evidence 는 그대로이고 바뀌는 것은
질문 집합 (= Correct 비율) 뿐이다.

비율 결정 (--ratios 를 주지 않으면):
  mid  = 전체 Correct 비율을 0.05 단위로 반올림 (Go/No-go 집합으로도 사용)
  low/high = mid -/+ d, d 는 (0.30, 0.25, 0.20, 0.15, 0.10) 중 세 집합을
             **겹치지 않게** 뽑을 수 있는
             가장 큰 값 (0.05 <= low, high <= 0.95). 없으면 겹침을 허용하고 manifest 에 기록.
  Correct 가 아닌 문항은 전체의 Ambiguous : Incorrect 비율을 유지한다
  (Correct 비율 외 action 구성 고정).
질문 집합이 다르면 문서 길이 등도 달라질 수 있으므로 집합별 입력 통계를 함께 기록한다.

    uv run python experiments/c2_gating/make_workload_sets.py \
        --scores results/official_crag/mode_a/score/20261126_072612 --name popqa_ratio_200
-> results/c2_gating/workload_sets/<name>/{low,mid,high}.txt, manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from c2_data import read_jsonl  # noqa: E402

from rag.telemetry.run_meta import create_run_dir, write_new  # noqa: E402

DELTAS = (0.30, 0.25, 0.20, 0.15, 0.10)


def composition(n: int, ratio: float, a_share: float) -> dict[str, int]:
    c = round(n * ratio)
    a = round((n - c) * a_share)
    return {"correct": c, "ambiguous": a, "incorrect": n - c - a}


def feasible(comps: list[dict], avail: dict[str, int], disjoint: bool) -> bool:
    if disjoint:
        return all(sum(x[k] for x in comps) <= avail[k] for k in avail)
    return all(x[k] <= avail[k] for x in comps for k in avail)


def choose(avail: dict[str, int], n: int, ratios: list[float] | None):
    total = sum(avail.values())
    a_share = avail["ambiguous"] / max(1, avail["ambiguous"] + avail["incorrect"])
    if ratios:
        comps = [composition(n, r, a_share) for r in ratios]
        dj = feasible(comps, avail, True)
        if not dj and not feasible(comps, avail, False):
            raise SystemExit(f"비율 {ratios} 로 {n}문항을 만들 수 없음: {avail}")
        return ratios, comps, dj, a_share
    mid = round(avail["correct"] / total / 0.05) * 0.05
    for disjoint in (True, False):
        for d in DELTAS:
            rs = [round(mid - d, 2), round(mid, 2), round(mid + d, 2)]
            if rs[0] < 0.05 or rs[2] > 0.95:
                continue
            comps = [composition(n, r, a_share) for r in rs]
            if feasible(comps, avail, disjoint):
                return rs, comps, disjoint, a_share
    raise SystemExit(f"Low/Mid/High 집합을 만들 수 없음: {avail}")


def build(scores_rows: list[dict], n: int, ratios, seed: int):
    by: dict[str, list[int]] = {}
    for r in scores_rows:
        by.setdefault(r["action"], []).append(int(r["index"]))
    avail = {k: len(by.get(k, [])) for k in ("correct", "ambiguous", "incorrect")}
    rs, comps, disjoint, a_share = choose(avail, n, ratios)
    rng = random.Random(seed)
    pools = {k: rng.sample(sorted(v), len(v)) for k, v in by.items()}
    sets = {}
    used: dict[str, int] = Counter()
    for name, comp in zip(("low", "mid", "high"), comps, strict=True):
        idx = []
        for k, m in comp.items():
            pool = pools[k]
            if disjoint:
                take = pool[used[k] : used[k] + m]
                used[k] += m
            else:
                take = rng.sample(pool, m)
            idx += take
        sets[name] = sorted(idx)
    overlap = {
        f"{a}&{b}": len(set(sets[a]) & set(sets[b]))
        for a, b in (("low", "mid"), ("low", "high"), ("mid", "high"))
    }
    return sets, {
        "available": avail,
        "overall_correct_ratio": avail["correct"] / sum(avail.values()),
        "ratios": dict(zip(("low", "mid", "high"), rs, strict=True)),
        "compositions": dict(zip(("low", "mid", "high"), comps, strict=True)),
        "ambiguous_share_of_non_correct": a_share,
        "disjoint": disjoint,
        "overlap": overlap,
        "seed": seed,
    }


def set_stats(idx: list[int], test_lines: list[str] | None) -> dict | None:
    if test_lines is None:
        return None
    chars, empty = [], 0
    for i in idx:
        for line in test_lines[i * 10 : i * 10 + 10]:
            p = line.split(" [SEP] ", 1)[1] if " [SEP] " in line else ""
            chars.append(len(p))
            empty += not p.strip()
    return {"mean_passage_chars": sum(chars) / len(chars), "empty_passages": empty}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ratios", default=None, help="예: 0.2,0.5,0.8 (주지 않으면 분포로 결정)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--repo", default=str(Path(__file__).resolve().parents[2] / "third_party" / "CRAG")
    )
    a = ap.parse_args(argv)
    rows = read_jsonl(Path(a.scores) / "scores.jsonl")
    ratios = [float(x) for x in a.ratios.split(",")] if a.ratios else None
    sets, info = build(rows, a.n, ratios, a.seed)
    test = Path(a.repo) / "data" / "popqa" / "test_popqa.txt"
    lines = test.read_text(encoding="utf-8").split("\n") if test.exists() else None
    d = create_run_dir("c2_gating/workload_sets", "results", a.name)
    for k, idx in sets.items():
        write_new(d / f"{k}.txt", "\n".join(map(str, idx)) + "\n")
    act = {int(r["index"]): r["action"] for r in rows}
    manifest = {
        **info,
        "scores_run": a.scores,
        "n_per_set": a.n,
        "analysis_type": "workload sensitivity (Correct 비율), 새 데이터셋 일반화 아님",
        "sets": {
            k: {
                "file": str(d / f"{k}.txt"),
                "action_counts": dict(Counter(act[i] for i in v)),
                "input_stats": set_stats(v, lines),
            }
            for k, v in sets.items()
        },
        "go_nogo_set": "mid",
    }
    write_new(d / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in manifest.items() if k != "sets"}, indent=2))
    for k, v in manifest["sets"].items():
        print(k, v["action_counts"], v["input_stats"])
    print(f"saved -> {d}")


if __name__ == "__main__":
    main()
