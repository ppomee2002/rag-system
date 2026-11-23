"""C1 비교 분석 (읽기 전용).

MODE B run 여러 개 -> 조건별 요약, 질문 단위 paired 비교, grader 판정 일치.

    uv run python experiments/official_crag/analyze.py <run_dir> <run_dir> ... \
        --baseline <run_dir> --name c1_matrix

결과 (이미 있으면 실패): results/official_crag/analysis/<name>/
  conditions.csv     run 별 정확도 (acc / official_match), 판정 분포, evidence, 단계 시간, 토큰
  paired.csv         baseline 대비 질문 단위 (둘 다 맞음 / baseline 만 / 비교 run 만 / 둘 다 틀림),
                     McNemar exact p (양측)
  agreement.csv      run 쌍별 판정 (correct/ambiguous/incorrect) 일치율과 교차표
  per_question.csv   질문 x run 정답 여부·판정·evidence 수
  report.md
비교 전 같은 입력 (frozen sha256) 과 같은 질문 집합인지 확인하고, 다르면 실패한다
(--allow-mismatch).
"""

from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np

from rag.retrieval.web_cache import read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def load_run(d: Path) -> dict:
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    rows = {r["question_id"]: r for r in read_jsonl(d / "queries.jsonl") if not r.get("warmup")}
    pid = next((r.get("policy_id") for r in rows.values() if r.get("policy_id")), None)
    return {
        "dir": str(d),
        "label": f"{pid or meta.get('pipeline', {}).get('policy_id')}|{d.name}",
        "meta": meta,
        "rows": rows,
        "ok": {q: r for q, r in rows.items() if not r.get("error")},
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def mcnemar_p(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def condition_row(run: dict) -> dict:
    ok = list(run["ok"].values())
    st = lambda s: mean([r["stage_timings"].get(s) for r in ok])  # noqa: E731
    return {
        "label": run["label"],
        "grader": ok[0].get("grader_type") if ok else None,
        "policy": ok[0].get("evidence_policy") if ok else None,
        "budget": (ok[0].get("budget") or {}).get("name") if ok else None,
        "n": len(run["rows"]),
        "n_errors": len(run["rows"]) - len(ok),
        "acc": mean([r["metrics"]["acc"] for r in ok]),
        "official_match": mean([r["metrics"]["official_match"] for r in ok]),
        "n_correct_path": sum(r["correction_path"] == "correct" for r in ok),
        "n_ambiguous_path": sum(r["correction_path"] == "ambiguous" for r in ok),
        "n_incorrect_path": sum(r["correction_path"] == "incorrect" for r in ok),
        "correction_invalid": sum(
            bool(r["correction"].get("performed") and not r["correction"].get("valid")) for r in ok
        ),
        "mean_docs_graded": mean([len(r["grading_events"]) for r in ok]),
        "mean_evidence_units": mean([r["evidence"]["n_units"] for r in ok]),
        "mean_evidence_chars": mean([r["evidence"]["n_chars"] for r in ok]),
        "mean_removed_retrieved": mean([len(r["evidence"]["removed_retrieved_ids"]) for r in ok]),
        "grading_s": st("grading"),
        "refinement_s": st("refinement"),
        "rewrite_s": st("rewrite"),
        "web_search_s": st("web_search"),
        "generation_s": st("generation"),
        "e2e_s": mean([r["e2e_latency"] for r in ok]),
        "evaluator_compute_s": mean([r.get("evaluator_compute_seconds") for r in ok]),
        "gen_in_tokens": mean([r.get("generation_input_tokens") for r in ok]),
        "gen_out_tokens": mean([r.get("generation_output_tokens") for r in ok]),
        "inputs_sha256": (run["meta"].get("inputs") or {}).get("sha256"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with open(path, "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline", default=None, help="paired 비교 기준 (기본: 첫 run)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--allow-mismatch", action="store_true")
    a = ap.parse_args(argv)
    runs = [load_run(Path(d)) for d in a.runs]
    base = load_run(Path(a.baseline)) if a.baseline else runs[0]
    if a.baseline and all(r["dir"] != base["dir"] for r in runs):
        runs.insert(0, base)

    shas = {r["meta"].get("inputs", {}).get("sha256") for r in runs}
    qsets = {frozenset(r["rows"]) for r in runs}
    problems = []
    if len(shas) > 1:
        problems.append(f"고정 입력 sha256 이 다름: {shas}")
    if len(qsets) > 1:
        problems.append("질문 집합이 다름")
    if problems and not a.allow_mismatch:
        raise SystemExit("; ".join(problems))

    out = ROOT / "results" / "official_crag" / "analysis" / a.name
    out.mkdir(parents=True, exist_ok=False)
    cond = [condition_row(r) for r in runs]
    write_csv(out / "conditions.csv", cond)

    paired = []
    for r in runs:
        if r is base:
            continue
        common = sorted(set(base["ok"]) & set(r["ok"]))
        both = only_b = only_r = neither = 0
        for q in common:
            x, y = base["ok"][q]["is_correct"], r["ok"][q]["is_correct"]
            both += x and y
            only_b += x and not y
            only_r += y and not x
            neither += not x and not y
        paired.append(
            {
                "baseline": base["label"],
                "compare": r["label"],
                "n_common": len(common),
                "both_correct": both,
                "baseline_only": only_b,
                "compare_only": only_r,
                "both_wrong": neither,
                "delta_acc": (only_r - only_b) / len(common) if common else None,
                "mcnemar_p": mcnemar_p(only_b, only_r),
            }
        )
    write_csv(out / "paired.csv", paired)

    agree = []
    acts = ("correct", "ambiguous", "incorrect")
    graded = [r for r in runs if any(x.get("grader_type") for x in r["ok"].values())]
    for i, r1 in enumerate(graded):
        for r2 in graded[i + 1 :]:
            common = sorted(set(r1["ok"]) & set(r2["ok"]))
            tab = {f"{x}->{y}": 0 for x in acts for y in acts}
            for q in common:
                tab[f"{r1['ok'][q]['correction_path']}->{r2['ok'][q]['correction_path']}"] += 1
            same = sum(tab[f"{x}->{x}"] for x in acts)
            agree.append(
                {
                    "run_a": r1["label"],
                    "run_b": r2["label"],
                    "n": len(common),
                    "agreement": same / len(common) if common else None,
                    **tab,
                }
            )
    write_csv(out / "agreement.csv", agree)

    per_q = []
    for q in sorted(base["rows"]):
        row = {"question_id": q}
        for r in runs:
            x = r["rows"].get(q) or {}
            tag = r["label"]
            row[f"{tag}:correct"] = x.get("is_correct")
            row[f"{tag}:path"] = x.get("correction_path")
            row[f"{tag}:n_units"] = (x.get("evidence") or {}).get("n_units")
        per_q.append(row)
    write_csv(out / "per_question.csv", per_q)

    lines = [f"# C1 비교: {a.name}", "", f"baseline: `{base['label']}`", ""]
    if problems:
        lines += ["**경고 (--allow-mismatch)**: " + "; ".join(problems), ""]
    lines += [
        "| 조건 | n | err | acc | official_match | C/A/I | evidence units | e2e s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cond:
        f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(
            f"| {c['label']} | {c['n']} | {c['n_errors']} | {f(c['acc'])} | {f(c['official_match'])} | "  # noqa: E501
            f"{c['n_correct_path']}/{c['n_ambiguous_path']}/{c['n_incorrect_path']} | "
            f"{f(c['mean_evidence_units'])} | {f(c['e2e_s'])} |"
        )
    lines += ["", "paired / agreement: paired.csv, agreement.csv", ""]
    lines.append(
        "주의: MODE B 는 공식 논문 수치의 재현이 아님. 230문항은 공통 문항 집합 (일반화 결과 아님)."
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
