"""SV live run 정책 쌍의 paired bootstrap 95% CI (읽기 전용).

기존 보고서 (analyze_sched_validation.py) 는 모든 정책을 fcfs 기준으로만 비교한다.
"A 보다 B 가 낫다" 는 A 와 B 를 직접 짝지어 비교해야 하므로 정책 쌍마다 따로 계산한다.

    uv run python experiments/crag_system/paired_bootstrap_sv.py \
        --live-root results/crag_system/pre/PRE-SV1-LIVE --k 4 --rhos 0.8 0.95 --seeds 0 1 \
        --pairs fcfs_query:srpt_est srpt_nopath:srpt_est --name SV1-bootstrap

  --pairs BASE:TARGET   차이 = TARGET - BASE (양수 = TARGET 이 느림). 기본: 위 두 쌍

짝짓기: 같은 workload 의 같은 요청 (request_index, query_id). 두 run 모두 정상 완료된 요청만.
통계량: 평균 / p50 / p95 / p99 latency 의 차이 (TARGET 통계량 - BASE 통계량) 와 평균 비율.
재표본:
  iid    요청을 복원 추출 (기존 analyze_sched_validation 과 같은 방식)
  block  도착 순서로 정렬한 요청을 길이 L 의 연속 block 으로 복원 추출 (circular moving block).
         같은 시기에 도착한 요청은 같은 대기열을 겪어 서로 상관되므로
         iid 는 CI 를 좁게 잡을 수 있다.
         두 방식이 다르면 block 쪽을 보수적인 값으로 본다.
묶음:
  조건별   rho x seed 하나
  rho 별   같은 rho 의 seed 들을 층으로 두고 층마다 따로 재표본한 뒤 합쳐 통계량 계산 (stratified)
  전체     모든 조건 (stratified)
  seed 가 2개뿐이라 workload (seed) 간 변동은 CI 에 들어가지 않는다. 조건별 부호 일치를 함께 본다.
부분집합: all / corrected_both / not_corrected_both (두 run 에서 경로가 같은 요청)
          + path_mismatch 수

해석 주의: 조건당 run 이 1개라 run 간 변동 (vLLM 비결정성) 이 CI 에 반영되지 않는다.
CI 가 0 을 포함하지 않아도 "이 run 쌍에서의 요청 표본 변동을 넘는 차이" 까지만 말할 수 있다.

결과: results/crag_system/analysis/<name>/<timestamp>/ bootstrap.csv, report.md, bootstrap.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_sv_policy import POLICIES, corrected, discover, load_run, md_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = ("fcfs_query:srpt_est", "srpt_nopath:srpt_est")
STATS = ("mean", "p50", "p95", "p99")


def stat(x: np.ndarray, name: str, axis=-1):
    if name == "mean":
        return x.mean(axis=axis)
    return np.percentile(x, float(name[1:]), axis=axis)


def pair_arrays(base: dict, target: dict) -> dict:
    """같은 요청끼리 (도착 순서) -> {subset: (base latency, target latency)}, path_mismatch."""
    keys = sorted(set(base["rows"]) & set(target["rows"]), key=lambda k: base["rows"][k]["arrival"])
    a = [base["rows"][k] for k in keys]
    b = [target["rows"][k] for k in keys]
    ca = np.array([corrected(r) for r in a], dtype=bool)
    cb = np.array([corrected(r) for r in b], dtype=bool)
    la = np.array([r["latency"] for r in a])
    lb = np.array([r["latency"] for r in b])
    subsets = {
        "all": np.ones(len(keys), dtype=bool),
        "corrected_both": ca & cb,
        "not_corrected_both": ~ca & ~cb,
    }
    return {
        "subsets": {s: (la[m], lb[m]) for s, m in subsets.items()},
        "n_only_base": len(set(base["rows"]) - set(target["rows"])),
        "n_only_target": len(set(target["rows"]) - set(base["rows"])),
        "path_mismatch": int((ca != cb).sum()),
    }


def resample_idx(n: int, n_boot: int, rng, method: str, block: int) -> np.ndarray:
    if method == "iid" or n <= block:
        return rng.integers(0, n, (n_boot, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, (n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n  # circular
    return idx.reshape(n_boot, -1)[:, :n]


def boot(strata: list[tuple[np.ndarray, np.ndarray]], n_boot: int, seed: int, method: str,
         block: int) -> dict:  # fmt: skip
    """층별로 따로 재표본한 뒤 합쳐서 통계량 차이. strata 하나면 조건별 분석."""
    rng = np.random.default_rng(seed)
    xa = np.concatenate([a for a, _ in strata])
    xb = np.concatenate([b for _, b in strata])
    ra, rb = [], []
    for a, b in strata:
        idx = resample_idx(len(a), n_boot, rng, method, block)
        ra.append(a[idx])
        rb.append(b[idx])
    ba, bb = np.concatenate(ra, axis=1), np.concatenate(rb, axis=1)
    out = {"n_pairs": len(xa), "base_mean": float(xa.mean()), "target_mean": float(xb.mean())}
    for s in STATS:
        d = stat(bb, s) - stat(ba, s)
        out[f"{s}_diff"] = float(stat(xb, s) - stat(xa, s))
        out[f"{s}_lo"] = float(np.percentile(d, 2.5))
        out[f"{s}_hi"] = float(np.percentile(d, 97.5))
    rel = bb.mean(axis=1) / ba.mean(axis=1) - 1
    out["mean_rel"] = float(xb.mean() / xa.mean() - 1)
    out["mean_rel_lo"] = float(np.percentile(rel, 2.5))
    out["mean_rel_hi"] = float(np.percentile(rel, 97.5))
    out["share_target_faster"] = float((xb < xa).mean())
    return out


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "target 느림"
    if hi < 0:
        return "target 빠름"
    return "구분 불가"


def run_all(conds: dict, pairs: list[str], a) -> tuple[list[dict], list[str]]:
    rows, warns = [], []
    loaded = {c: {p: load_run(d) for p, d in runs.items()} for c, runs in conds.items()}
    for pair in pairs:
        base, target = pair.split(":", 1)
        per = {}
        for c, runs in loaded.items():
            if base not in runs or target not in runs:
                warns.append(f"{c}: {pair} 생략 (run 없음)")
                continue
            pa = pair_arrays(runs[base], runs[target])
            if pa["n_only_base"] or pa["n_only_target"]:
                warns.append(
                    f"{c} {pair}: 한쪽에만 있는 요청 base {pa['n_only_base']} / "
                    f"target {pa['n_only_target']} (오류 요청, 제외)"
                )
            per[c] = pa
        groups = {c: [c] for c in per}
        for rho in a.rhos:
            cs = [c for c in per if c.startswith(f"rho{rho}_")]
            if len(cs) > 1:
                groups[f"rho{rho}_pooled"] = cs
        if len(per) > 1:
            groups["all_pooled"] = list(per)
        for g, cs in groups.items():
            for sub in ("all", "corrected_both", "not_corrected_both"):
                strata = [per[c]["subsets"][sub] for c in cs]
                if sum(len(x) for x, _ in strata) < 2 or any(len(x) < 2 for x, _ in strata):
                    continue
                for method in a.methods:
                    r = boot(strata, a.n_boot, a.seed, method, a.block)
                    rows.append(
                        {
                            "pair": pair, "base": base, "target": target, "group": g,
                            "n_conditions": len(cs), "subset": sub, "method": method,
                            "path_mismatch": sum(per[c]["path_mismatch"] for c in cs),
                            **r,
                            "mean_verdict": verdict(r["mean_lo"], r["mean_hi"]),
                            "p95_verdict": verdict(r["p95_lo"], r["p95_hi"]),
                        }
                    )  # fmt: skip
    return rows, warns


def sign_table(rows: list[dict]) -> list[dict]:
    """조건별 (pooled 제외) 점추정 부호 일치."""
    out = []
    for pair in dict.fromkeys(r["pair"] for r in rows):
        rs = [
            r for r in rows
            if r["pair"] == pair and r["subset"] == "all" and r["method"] == rows[0]["method"]
            and r["n_conditions"] == 1
        ]  # fmt: skip
        if not rs:
            continue
        out.append(
            {
                "pair": pair, "n_conditions": len(rs),
                "mean_target_faster": sum(r["mean_diff"] < 0 for r in rs),
                "p95_target_faster": sum(r["p95_diff"] < 0 for r in rs),
                "mean_ci_excludes_0": sum(r["mean_verdict"] != "구분 불가" for r in rs),
                "p95_ci_excludes_0": sum(r["p95_verdict"] != "구분 불가" for r in rs),
            }
        )  # fmt: skip
    return out


def report_md(rows: list[dict], warns: list[str], missing: list[str], a) -> str:
    L = [
        f"# 정책 쌍 paired bootstrap ({a.name})",
        "",
        f"- 차이 = target - base (초). 양수면 target 이 느림. 재표본 {a.n_boot}회, seed {a.seed}, "
        f"block 길이 {a.block} (도착 순서)",
        "- 조건당 run 1개: run 간 변동은 CI 에 포함되지 않음."
        " seed 2개: workload 간 변동도 포함되지 않음",
        "",
        "## 조건별 부호 일치 (subset=all, 점추정)",
        "",
    ]
    st = sign_table(rows)
    L += md_table(st, list(st[0]) if st else ["pair"])
    cols = ["pair", "group", "subset", "method", "n_pairs", "path_mismatch", "base_mean",
            "target_mean", "mean_diff", "mean_lo", "mean_hi", "mean_rel", "mean_verdict",
            "p95_diff", "p95_lo", "p95_hi", "p95_verdict", "p50_diff", "p99_diff",
            "share_target_faster"]  # fmt: skip
    for sub in ("all", "corrected_both", "not_corrected_both"):
        rs = [r for r in rows if r["subset"] == sub]
        if rs:
            L += ["", f"## subset = {sub}", ""] + md_table(rs, cols)
    L += ["", "## 확인", ""]
    L += [f"- 없는 run: {m}" for m in missing]
    L += [f"- {w}" for w in warns] or ["- 특이사항 없음"]
    L += [""]
    return "\n".join(L)


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-root", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rhos", nargs="+", default=["0.8", "0.95"])
    ap.add_argument("--seeds", nargs="+", default=["0", "1"])
    ap.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS), metavar="BASE:TARGET")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--block", type=int, default=20, help="block bootstrap 길이 (요청 수)")
    ap.add_argument("--methods", nargs="+", default=["iid", "block"], choices=["iid", "block"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="sv_bootstrap")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    for p in a.pairs:
        if ":" not in p or any(x not in POLICIES for x in p.split(":", 1)):
            raise SystemExit(f"--pairs 는 BASE:TARGET ({POLICIES}): {p}")
    pols = sorted({x for p in a.pairs for x in p.split(":", 1)})
    conds, missing = discover(a.live_root, a.rhos, a.seeds, a.k, pols)
    if not conds:
        raise SystemExit(f"run 이 없음: {a.live_root}")
    rows, warns = run_all(conds, a.pairs, a)
    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "bootstrap.csv", "x", newline="", encoding="utf-8-sig") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "bootstrap.json").write_text(
        json.dumps({"created": datetime.now().isoformat(), "args": vars(a), "conditions": conds,
                    "missing": missing, "warnings": warns, "rows": rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )  # fmt: skip
    md = report_md(rows, warns, missing, a)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not a.quiet:
        print(md)
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
