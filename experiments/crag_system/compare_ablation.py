"""C0 (Full 재실행) 대비 Ablation A / B 비교 (읽기 전용).

    uv run python experiments/crag_system/compare_ablation.py --reference c0 \
        --run c0=results/crag_system/pre/PRE-ABL-C0-FULL/<ts> \
        --run abl_a=results/crag_system/pre/PRE-ABL-A-INC-KEEP/<ts> \
        --run abl_b=results/crag_system/pre/PRE-ABL-B-COR-KEEP/<ts> \
        --run full_retry=results/crag_system/smoke/SMOKE-A1-FULL-RETRY/20261110_111726 \
        --run vanilla=results/crag_system/smoke/SMOKE-A1-VANILLA/20261110_103018

비교 항목 (모든 run 에 정상 결과가 있는 공통 질문 기준, 같은 query_id 끼리 짝지음)
  Accuracy, F1, Mean / p95 E2E, query 당 LLM 호출 수, Generation 입력 토큰 (generate 호출
  prompt_tokens), corrective action 분포
  reference 대비 paired 차이: accuracy (bootstrap 95% CI, McNemar exact), E2E 평균·p95 (CI),
    generation 입력 토큰·LLM 호출 수 평균 차이 (CI)
  action 별 accuracy: (1) reference 의 action 으로 묶은 paired 비교, (2) 각 run 자체 action 기준
  action 이 reference 와 달라진 질문 목록 (+ 그 질문을 뺀 paired accuracy 차이)
  Web replay: run_end.json 의 hit / miss / empty / rewrite 불일치, 오류 행 (replay miss 포함),
    공통 질문에서 빠진 질문과 이유. 조용히 제외하지 않고 report 맨 앞에 보고한다.

결과: results/crag_system/analysis/ablation/<name>/<timestamp>/
  report.md, summary.csv, paired.csv, by_action_paired.csv, by_action_own.csv,
  action_changes.csv, replay.csv, excluded.csv, per_query.csv, compare_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_regression import ROOT, md_cell, resolve  # noqa: E402
from analyze_runs import (  # noqa: E402
    boot_idx,
    ci,
    doc_scores,
    latency_of,
    load_run,
    mcnemar_exact,
    md_table,
    paired_tables,
    pctl,
    read_jsonl,
    write_csv,
)

ACTIONS = ("correct", "ambiguous", "incorrect")


# ---------------------------------------------------------------- per query values
def gen_prompt_tokens(calls: list[dict]):
    gs = [c for c in calls if c.get("kind") == "generate"]
    return gs[-1].get("prompt_tokens") if gs else None


def n_llm_calls(row: dict, calls: list[dict]) -> int:
    n = (row.get("calls_summary") or {}).get("n_calls")
    return n if n is not None else len(calls)


def has_actions(run: dict) -> bool:
    return any(r.get("corrective_action") for r in run["rows"].values())


def boot_mean_ci(d: np.ndarray, b: int, seed: int) -> tuple[float, float] | tuple[None, None]:
    if len(d) == 0:
        return None, None
    return ci(d[boot_idx(len(d), b, seed)].mean(axis=1))


# ---------------------------------------------------------------- replay / exclusion
def replay_report(runs: dict, raw: dict, common: set[str]) -> tuple[list[dict], list[dict]]:
    rep, excl = [], []
    for name, run in runs.items():
        end_p = Path(run["dir"]) / "run_end.json"
        ws = (
            json.loads(end_p.read_text(encoding="utf-8")).get("web_search")
            if end_p.exists()
            else None
        ) or {}
        rows = [r for r in raw[name] if not r.get("warmup")]
        errors = [r for r in rows if r.get("error")]
        miss_err = [r for r in errors if "web cache miss" in r["error"]]
        not_hit = [
            r
            for r in rows
            if not r.get("error") and (r.get("web_replay") or {}).get("hit") is False
        ]
        wcfg = (run["cfg"].get("crag") or {}).get("web_search") or {}
        provider = "-" if run["cfg"].get("pipeline") == "vanilla" else wcfg.get("provider")
        rep.append(
            {
                "run": name,
                "provider": provider,
                "replay_on_miss": wcfg.get("replay_on_miss") if provider == "replay" else None,
                "hits": ws.get("hits"),
                "misses": ws.get("misses"),
                "empty_returned": ws.get("empty_returned"),
                "rewrite_mismatch": ws.get("rewrite_mismatch"),
                "n_error_rows": len(errors),
                "n_error_replay_miss": len(miss_err),
                "n_rows_replay_not_hit": len(not_hit),
                "miss_queries": json.dumps(ws.get("miss_queries") or [], ensure_ascii=False),
            }
        )
        for r in errors:
            excl.append(
                {
                    "run": name,
                    "query_id": r["query_id"],
                    "reason": "replay miss (오류 행)" if r in miss_err else "오류 행",
                    "detail": r["error"][:300],
                }
            )
        for r in not_hit:  # replay_on_miss=empty: 결과는 있으나 web 문서 0개
            excl.append(
                {
                    "run": name,
                    "query_id": r["query_id"],
                    "reason": "replay miss (빈 결과로 진행, 비교에는 포함)",
                    "detail": json.dumps(r.get("web_replay"), ensure_ascii=False)[:300],
                }
            )
        ok = set(run["rows"])
        for q in sorted(ok - common):
            excl.append(
                {
                    "run": name,
                    "query_id": q,
                    "reason": "다른 run 에 정상 결과가 없어 공통 질문에서 제외",
                    "detail": "",
                }
            )
    return rep, excl


# ---------------------------------------------------------------- tables
def summary_rows(runs: dict, qids: list[str]) -> list[dict]:
    out = []
    for name, run in runs.items():
        rs = [run["rows"][q] for q in qids]
        cs = [run["calls_by_query"].get(q, []) for q in qids]
        lat = [latency_of(r) for r in rs]
        gtok = [gen_prompt_tokens(c) for c in cs]
        gtok_k = [t for t in gtok if t is not None]
        kinds = Counter(c["kind"] for cl in cs for c in cl)
        acts = Counter(r.get("corrective_action") for r in rs)
        out.append(
            {
                "run": name,
                "pipeline": rs[0].get("pipeline") if rs else None,
                "n": len(rs),
                "accuracy": float(np.mean([r["acc"] for r in rs])),
                "f1": float(np.mean([r["f1"] for r in rs])),
                "e2e_mean_s": float(np.mean(lat)),
                "e2e_p95_s": pctl(lat, 95),
                "llm_calls_mean": float(
                    np.mean([n_llm_calls(r, c) for r, c in zip(rs, cs, strict=True)])
                ),
                **{f"calls_{k}_mean": kinds[k] / len(rs) for k in sorted(kinds)},
                "gen_input_tokens_mean": float(np.mean(gtok_k)) if gtok_k else None,
                "gen_input_tokens_p50": pctl(gtok_k, 50) if gtok_k else None,
                "gen_input_tokens_p95": pctl(gtok_k, 95) if gtok_k else None,
                "gen_input_tokens_known": len(gtok_k),
                **{f"n_{a}": acts.get(a, 0) for a in ACTIONS},
            }
        )  # fmt: skip
    return out


def paired_rows(runs, qids, ref, b, seed) -> list[dict]:
    pairs = [(ref, k) for k in runs if k != ref]
    acc, lat = paired_tables(runs, qids, pairs, b, seed)
    out = []
    for (_, k), a_row, l_row in zip(pairs, acc, lat, strict=True):
        r, o = runs[ref], runs[k]
        gt = [
            (
                gen_prompt_tokens(r["calls_by_query"].get(q, [])),
                gen_prompt_tokens(o["calls_by_query"].get(q, [])),
            )
            for q in qids
        ]
        gd = np.array([y - x for x, y in gt if x is not None and y is not None], dtype=float)
        cd = np.array(
            [
                n_llm_calls(o["rows"][q], o["calls_by_query"].get(q, []))
                - n_llm_calls(r["rows"][q], r["calls_by_query"].get(q, []))
                for q in qids
            ],
            dtype=float,
        )
        glo, ghi = boot_mean_ci(gd, b, seed)
        clo, chi = boot_mean_ci(cd, b, seed)
        out.append(
            {
                "pair": f"{k} - {ref}",
                "n": len(qids),
                "acc_ref": a_row["acc_a"],
                "acc_run": a_row["acc_b"],
                "acc_diff": a_row["diff"],
                "acc_ci95_low": a_row["ci95_low"],
                "acc_ci95_high": a_row["ci95_high"],
                "ref_right_run_wrong": a_row["a_right_b_wrong"],
                "ref_wrong_run_right": a_row["a_wrong_b_right"],
                "mcnemar_exact_p": a_row["mcnemar_exact_p"],
                "e2e_mean_diff_s": l_row["mean_diff"],
                "e2e_mean_ci95_low": l_row["mean_ci95_low"],
                "e2e_mean_ci95_high": l_row["mean_ci95_high"],
                "e2e_p95_diff_s": l_row["p95_diff"],
                "e2e_p95_ci95_low": l_row["p95_ci95_low"],
                "e2e_p95_ci95_high": l_row["p95_ci95_high"],
                "gen_tokens_mean_diff": float(gd.mean()) if len(gd) else None,
                "gen_tokens_ci95_low": glo,
                "gen_tokens_ci95_high": ghi,
                "gen_tokens_n_pairs": len(gd),
                "llm_calls_mean_diff": float(cd.mean()),
                "llm_calls_ci95_low": clo,
                "llm_calls_ci95_high": chi,
            }
        )  # fmt: skip
    return out


def paired_acc(xa, xb, b, seed) -> dict:
    xa, xb = np.asarray(xa, dtype=float), np.asarray(xb, dtype=float)
    d = xb - xa
    lo, hi = boot_mean_ci(d, b, seed)
    n01 = int(((xa == 1) & (xb == 0)).sum())
    n10 = int(((xa == 0) & (xb == 1)).sum())
    return {
        "n": len(d),
        "acc_ref": float(xa.mean()) if len(d) else None,
        "acc_run": float(xb.mean()) if len(d) else None,
        "acc_diff": float(d.mean()) if len(d) else None,
        "acc_ci95_low": lo,
        "acc_ci95_high": hi,
        "ref_right_run_wrong": n01,
        "ref_wrong_run_right": n10,
        "mcnemar_exact_p": mcnemar_exact(n01, n10),
    }


def by_action_paired(runs, qids, ref, b, seed) -> list[dict]:
    """reference 의 action 으로 질문을 묶고 같은 질문끼리 비교."""
    out = []
    R = runs[ref]["rows"]
    for k, run in runs.items():
        if k == ref:
            continue
        for act in (*ACTIONS, "all"):
            qs = [q for q in qids if act == "all" or R[q].get("corrective_action") == act]
            row = {"run": k, "ref_action": act}
            row.update(
                paired_acc([R[q]["acc"] for q in qs], [run["rows"][q]["acc"] for q in qs], b, seed)
            )
            out.append(row)
    return out


def by_action_own(runs, qids) -> list[dict]:
    out = []
    for k, run in runs.items():
        if not has_actions(run):
            continue
        for act in ACTIONS:
            rs = [run["rows"][q] for q in qids if run["rows"][q].get("corrective_action") == act]
            out.append(
                {
                    "run": k,
                    "own_action": act,
                    "n": len(rs),
                    "accuracy": float(np.mean([r["acc"] for r in rs])) if rs else None,
                    "f1": float(np.mean([r["f1"] for r in rs])) if rs else None,
                }
            )
    return out


def action_changes(ref_rows: dict, rows: dict, qids: list[str]) -> list[dict]:
    out = []
    for q in qids:
        a, b = ref_rows[q], rows[q]
        if a.get("corrective_action") != b.get("corrective_action"):
            sa, sb = doc_scores(a), doc_scores(b)
            out.append(
                {
                    "query_id": q,
                    "ref_action": a.get("corrective_action"),
                    "run_action": b.get("corrective_action"),
                    "grade_scores_equal": sa == sb,
                    "ref_scores": json.dumps(sa),
                    "run_scores": json.dumps(sb),
                    "ref_acc": int(bool(a.get("acc"))),
                    "run_acc": int(bool(b.get("acc"))),
                }
            )
    return out


def score_diff_count(ref_rows, rows, qids) -> int:
    return sum(doc_scores(ref_rows[q]) != doc_scores(rows[q]) for q in qids)


# ---------------------------------------------------------------- report
def t(rows, cols=None):
    if not rows:
        return "(없음)\n"
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    return md_table([{k: md_cell(r.get(k)) for k in cols} for r in rows], cols)


def build_report(m, tabs) -> str:
    ref = m["reference"]
    L = [
        f"# Ablation 비교 (기준: {ref})",
        "",
        *[
            f"- {k}: `{v['dir']}` (pipeline={v['pipeline']}, ablation={v['ablation']})"
            for k, v in m["runs"].items()
        ],  # fmt: skip
        f"- 공통 질문 {m['n_common']}개. 같은 query_id 끼리 짝지음. bootstrap B={m['bootstrap']}, "
        f"seed={m['seed']} (query 복원 추출, 기술 통계)",
        "",
        "## 0. Web replay 와 제외된 질문 (먼저 확인)",
        "",
        t(
            m["replay"],
            [
                "run",
                "provider",
                "replay_on_miss",
                "hits",
                "misses",
                "empty_returned",
                "rewrite_mismatch",
                "n_error_rows",
                "n_error_replay_miss",
                "n_rows_replay_not_hit",
            ],
        ),  # fmt: skip
    ]
    if m["n_replay_problems"]:
        L += [
            f"**주의: replay miss / 오류가 {m['n_replay_problems']}건 있다.** 아래 목록의 질문은 "
            "공통 질문에서 빠졌거나 (오류 행) web 문서 없이 진행되었다 (empty). 기준 run 과 검색 "
            "조건이 다르므로 해석 시 따로 본다.",
            "",
        ]
    else:
        L += ["- replay miss, 오류 행 없음", ""]
    L += [
        t(tabs["excluded"]),
        "## 1. 전체 지표",
        "",
        t(
            tabs["summary"],
            [
                "run",
                "n",
                "accuracy",
                "f1",
                "e2e_mean_s",
                "e2e_p95_s",
                "llm_calls_mean",
                "gen_input_tokens_mean",
                "gen_input_tokens_p95",
                "n_correct",
                "n_ambiguous",
                "n_incorrect",
            ],
        ),  # fmt: skip
        "## 2. 기준 대비 paired 차이 (같은 질문, 95% CI)",
        "",
        t(
            tabs["paired"],
            [
                "pair",
                "n",
                "acc_diff",
                "acc_ci95_low",
                "acc_ci95_high",
                "ref_right_run_wrong",
                "ref_wrong_run_right",
                "mcnemar_exact_p",
            ],
        ),
        t(
            tabs["paired"],
            [
                "pair",
                "e2e_mean_diff_s",
                "e2e_mean_ci95_low",
                "e2e_mean_ci95_high",
                "e2e_p95_diff_s",
                "e2e_p95_ci95_low",
                "e2e_p95_ci95_high",
            ],
        ),
        t(
            tabs["paired"],
            [
                "pair",
                "gen_tokens_mean_diff",
                "gen_tokens_ci95_low",
                "gen_tokens_ci95_high",
                "llm_calls_mean_diff",
                "llm_calls_ci95_low",
                "llm_calls_ci95_high",
            ],
        ),  # fmt: skip
        "E2E 는 같은 서버·같은 replay 조건인 run 끼리만 비교한다 (기존 FULL-RETRY 는 실제 Tavily, "
        "다른 시점 실행).",
        "",
        f"## 3. Action 별 Accuracy (기준 {ref} 의 action 으로 묶은 paired 비교)",
        "",
        t(
            tabs["by_action_paired"],
            [
                "run",
                "ref_action",
                "n",
                "acc_ref",
                "acc_run",
                "acc_diff",
                "acc_ci95_low",
                "acc_ci95_high",
                "ref_right_run_wrong",
                "ref_wrong_run_right",
            ],
        ),  # fmt: skip
        "각 run 자체 action 기준 (집단 구성이 run 마다 다를 수 있음):",
        "",
        t(tabs["by_action_own"]),
        f"## 4. 기준 {ref} 와 action 이 달라진 질문",
        "",
        t(m["action_change_summary"]),
        t(tabs["action_changes"]),
        "action 이 달라진 질문은 grading 이 재현되지 않은 것이다 (Ablation 은 판정 이후만 바꾼다). "
        "위 표의 `acc_diff_action_same` 은 그 질문을 뺀 paired accuracy 차이다.",
        "",
    ]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--reference", default="c0")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="abl")
    ap.add_argument(
        "--out-root", default=str(ROOT / "results" / "crag_system" / "analysis" / "ablation")
    )
    a = ap.parse_args(argv)
    specs = dict(s.partition("=")[::2] for s in a.run)
    if a.reference not in specs:
        ap.error(f"--reference {a.reference} 가 --run 에 없음")
    runs = {k: load_run(k, resolve(v)) for k, v in specs.items()}
    raw = {k: read_jsonl(Path(r["dir"]) / "queries.jsonl") for k, r in runs.items()}
    ref = a.reference
    common = set.intersection(*(set(r["rows"]) for r in runs.values()))
    qids = [q for q in runs[ref]["rows"] if q in common]
    if not qids:
        raise SystemExit("공통 질문이 없음")
    replay, excluded = replay_report(runs, raw, common)

    changes, change_summary = [], []
    for k, run in runs.items():
        if k == ref or not has_actions(run) or not has_actions(runs[ref]):
            continue
        ch = action_changes(runs[ref]["rows"], run["rows"], qids)
        changes += [{"run": k, **c} for c in ch]
        same = [q for q in qids if q not in {c["query_id"] for c in ch}]
        pa = paired_acc(
            [runs[ref]["rows"][q]["acc"] for q in same], [run["rows"][q]["acc"] for q in same],
            a.bootstrap, a.seed,
        )  # fmt: skip
        change_summary.append(
            {
                "run": k,
                "n_action_changed": len(ch),
                "n_grade_scores_differ": score_diff_count(runs[ref]["rows"], run["rows"], qids),
                "n_action_same": len(same),
                "acc_diff_action_same": pa["acc_diff"],
                "ci95_low": pa["acc_ci95_low"],
                "ci95_high": pa["acc_ci95_high"],
            }
        )
    tabs = {
        "summary": summary_rows(runs, qids),
        "paired": paired_rows(runs, qids, ref, a.bootstrap, a.seed),
        "by_action_paired": by_action_paired(runs, qids, ref, a.bootstrap, a.seed)
        if has_actions(runs[ref])
        else [],
        "by_action_own": by_action_own(runs, qids),
        "action_changes": changes,
        "replay": replay,
        "excluded": excluded,
    }
    per_query = []
    for q in qids:
        row = {"query_id": q}
        for k, run in runs.items():
            r = run["rows"][q]
            row.update(
                {
                    f"{k}_acc": int(bool(r["acc"])),
                    f"{k}_action": r.get("corrective_action"),
                    f"{k}_latency": latency_of(r),
                    f"{k}_gen_tokens": gen_prompt_tokens(run["calls_by_query"].get(q, [])),
                }
            )
        per_query.append(row)
    tabs["per_query"] = per_query
    meta = {
        "created": datetime.now().isoformat(),
        "reference": ref,
        "runs": {
            k: {
                "dir": str(r["dir"]),
                "exp_id": r["cfg"].get("exp_id"),
                "pipeline": next(iter(r["rows"].values()), {}).get("pipeline"),
                "ablation": r["cfg"].get("ablation"),
                "n_ok": len(r["rows"]),
                "errors": r["errors"],
            }
            for k, r in runs.items()
        },
        "n_common": len(qids),
        "bootstrap": a.bootstrap,
        "seed": a.seed,
        "replay": replay,
        # miss 하나가 run_end 의 misses 와 오류 행에 함께 잡히므로 중복 없이 센다
        "n_replay_problems": sum(
            max(x["misses"] or 0, x["n_error_replay_miss"] + x["n_rows_replay_not_hit"])
            + x["n_error_rows"]
            - x["n_error_replay_miss"]
            for x in replay
        ),
        "action_change_summary": change_summary,
    }
    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    for r in runs.values():
        if out.resolve().is_relative_to(Path(r["dir"]).resolve()):
            raise ValueError(f"출력 폴더가 run 폴더 안에 있음: {out}")
    out.mkdir(parents=True, exist_ok=False)
    for k, rows in tabs.items():
        write_csv(out / f"{k}.csv", rows)
    with open(out / "compare_meta.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    with open(out / "report.md", "x", encoding="utf-8") as f:
        f.write(build_report(meta, tabs))
    for s in tabs["summary"]:
        print(f"{s['run']:12s} acc={s['accuracy']:.4f} f1={s['f1']:.4f} e2e={s['e2e_mean_s']:.2f}s")
    if meta["n_replay_problems"]:
        print(f"주의: replay miss / 오류 {meta['n_replay_problems']}건 (report.md 0절)")
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
