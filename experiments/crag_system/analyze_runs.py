"""여러 run (Vanilla / Full / Lazy-hit / Batch 등) 의 기존 결과 파일만 읽어 비교 분석한다.

추가 LLM 추론이나 Tavily 호출은 하지 않는다.
읽는 파일: queries.jsonl, calls.jsonl, summary.json, config.yaml, (있으면) web_cache.jsonl,
정답 문서 확인용 chunk 파일 (config 의 retriever.docs_path).

    uv run python experiments/crag_system/analyze_runs.py --preset a-smoke
    # 또는 직접 지정
    uv run python experiments/crag_system/analyze_runs.py --base results/crag_system/smoke \
        --run vanilla=SMOKE-A1-VANILLA/20261110_103018 --run full=SMOKE-A1-FULL-RETRY/... \
        --baseline vanilla --reference full

결과: results/crag_system/analysis/<name>/<timestamp>/ (기존 결과는 덮어쓰지 않음)
  runs.json, report.md
  accuracy.csv / paired_accuracy.csv       정답률, 짝지은 차이와 paired bootstrap 95% CI, McNemar
  latency.csv / paired_latency.csv         평균·p50·p95,
                                           평균 차이와 p95 차이의 paired bootstrap 95% CI
  stages.csv / calls_by_kind.csv           단계별 지연, 호출 종류별 호출 수·지연·TTFT·토큰
  transitions.csv                          기준 run 대비 정답 변화 (corrective action 별)
  regression_cases.csv / regression.json   기준 (Vanilla) 정답 -> 비교 (Full) 오답 사례의
                                           로그 기반 분류
  batch_diagnosis.json                     Batch grading 호출 수 감소 vs 지연 증가 원인 지표
  per_query.csv                            query 별 run 결과

통계는 같은 query_id 를 짝지어 계산한다 (오류·warmup 행 제외, 모든 run 에 있는 query 만).
bootstrap 은 query 를 복원 추출하며 기술 통계이다 (smoke 결과는 논문 결과로 쓰지 않는다).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag.eval.metrics import normalize_answer  # noqa: E402

PRESETS = {
    "a-smoke": {
        "base": "results/crag_system/smoke",
        "runs": {
            "vanilla": "SMOKE-A1-VANILLA/20261110_103018",
            "full": "SMOKE-A1-FULL-RETRY/20261110_111726",
            "lazy_hit": "SMOKE-A2-LAZY/20261110_122447",
            "batch": "SMOKE-A3-BATCH/20261110_124257",
        },
        "baseline": "vanilla",
        "reference": "full",
        "excluded": {
            "SMOKE-A1-CHECK": "사전 테스트",
            "SMOKE-A1-FULL": "Tavily API 오류로 실패한 실행",
        },
    }
}
GRADE_KINDS = ("grade", "grade_batch")
KIND_STAGE = {
    "grade": "grading",
    "grade_batch": "grading",
    "rewrite": "correction",
    "generate": "generation",
}
STAGES = ("retrieval", "grading", "rewrite", "web_search", "generation")
NUM_ONLY = re.compile(r"^\s*\d*\.?\d+\s*\.?\s*$")


# ---------------------------------------------------------------- loading
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_run(name: str, run_dir: Path) -> dict:
    if not (run_dir / "queries.jsonl").exists():
        raise FileNotFoundError(f"{name}: {run_dir}/queries.jsonl 없음")
    rows = read_jsonl(run_dir / "queries.jsonl")
    calls = [c for c in read_jsonl(run_dir / "calls.jsonl") if not c.get("warmup")]
    cfg = (
        yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        if (run_dir / "config.yaml").exists()
        else {}
    )
    summary = (
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if (run_dir / "summary.json").exists()
        else {}
    )
    measured = [r for r in rows if not r.get("warmup")]
    ok = {r["query_id"]: r for r in measured if not r.get("error")}
    by_req = defaultdict(list)
    for c in calls:
        by_req[(c["query_id"], c["request_index"])].append(c)
    return {
        "name": name,
        "dir": run_dir,
        "cfg": cfg or {},
        "summary": summary,
        "rows": ok,
        "n_measured": len(measured),
        "errors": [r["query_id"] for r in measured if r.get("error")],
        "calls": calls,
        "calls_by_query": {q: by_req[(q, r["request_index"])] for q, r in ok.items()},
    }


def run_info(run: dict) -> dict:
    cfg = run["cfg"]
    llm = cfg.get("llm") or {}
    ws = (cfg.get("crag") or {}).get("web_search") or {}
    return {
        "dir": str(run["dir"]),
        "exp_id": cfg.get("exp_id"),
        "purpose": (cfg.get("run") or {}).get("purpose"),
        "pipeline": cfg.get("pipeline"),
        "grading": (cfg.get("grading") or {}).get("strategy"),
        "engine_profile": cfg.get("engine_profile"),
        "model": llm.get("model"),
        "web_search": {k: ws.get(k) for k in ("provider", "record", "latency_mode")},
        "dataset_limit": (cfg.get("dataset") or {}).get("limit"),
        "n_measured": run["n_measured"],
        "n_ok": len(run["rows"]),
        "errors": run["errors"],
        "summary_json": {k: run["summary"].get(k) for k in ("n_ok", "n_errors", "accuracy")},
    }


# ---------------------------------------------------------------- stats
def pctl(x, q):
    return float(np.percentile(np.asarray(x, dtype=float), q)) if len(x) else math.nan


def boot_idx(n: int, b: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def mcnemar_exact(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def latency_of(r: dict) -> float:
    return r.get("latency", r.get("e2e_latency"))


def paired_tables(runs: dict, qids: list[str], pairs: list[tuple[str, str]], b: int, seed: int):
    idx = boot_idx(len(qids), b, seed)
    acc_rows, lat_rows = [], []
    for a, c in pairs:
        ra, rc = runs[a]["rows"], runs[c]["rows"]
        xa = np.array([ra[q]["acc"] for q in qids], dtype=float)
        xc = np.array([rc[q]["acc"] for q in qids], dtype=float)
        la = np.array([latency_of(ra[q]) for q in qids], dtype=float)
        lc = np.array([latency_of(rc[q]) for q in qids], dtype=float)
        d = xc - xa
        dist = d[idx].mean(axis=1)
        lo, hi = ci(dist)
        b01 = int(((xa == 1) & (xc == 0)).sum())
        b10 = int(((xa == 0) & (xc == 1)).sum())
        acc_rows.append(
            {
                "pair": f"{c} - {a}",
                "a": a,
                "b": c,
                "n": len(qids),
                "acc_a": xa.mean(),
                "acc_b": xc.mean(),
                "diff": d.mean(),
                "ci95_low": lo,
                "ci95_high": hi,
                "a_right_b_wrong": b01,
                "a_wrong_b_right": b10,
                "mcnemar_exact_p": mcnemar_exact(b01, b10),
            }
        )
        md = (lc - la)[idx].mean(axis=1)
        p95d = np.percentile(lc[idx], 95, axis=1) - np.percentile(la[idx], 95, axis=1)
        lat_rows.append(
            {
                "pair": f"{c} - {a}",
                "a": a,
                "b": c,
                "n": len(qids),
                "mean_a": la.mean(),
                "mean_b": lc.mean(),
                "mean_diff": (lc - la).mean(),
                "mean_ci95_low": ci(md)[0],
                "mean_ci95_high": ci(md)[1],
                "p95_a": pctl(la, 95),
                "p95_b": pctl(lc, 95),
                "p95_diff": pctl(lc, 95) - pctl(la, 95),
                "p95_ci95_low": ci(p95d)[0],
                "p95_ci95_high": ci(p95d)[1],
            }
        )
    return acc_rows, lat_rows


# ---------------------------------------------------------------- per-run tables
def accuracy_rows(runs: dict, qids: list[str]) -> list[dict]:
    out = []
    for name, run in runs.items():
        rs = [run["rows"][q] for q in qids]
        acts = Counter(r.get("corrective_action") for r in rs)
        out.append(
            {
                "run": name,
                "n": len(rs),
                "accuracy": np.mean([r["acc"] for r in rs]),
                "f1": np.mean([r["f1"] for r in rs]),
                "em": np.mean([r["em"] for r in rs]),
                "correct": acts.get("correct", 0),
                "ambiguous": acts.get("ambiguous", 0),
                "incorrect": acts.get("incorrect", 0),
                "n_errors_in_run": len(run["errors"]),
            }
        )
    return out


def latency_rows(runs: dict, qids: list[str]) -> list[dict]:
    out = []
    for name, run in runs.items():
        lat = [latency_of(run["rows"][q]) for q in qids]
        out.append(
            {
                "run": name,
                "n": len(lat),
                "mean": np.mean(lat),
                "p50": pctl(lat, 50),
                "p95": pctl(lat, 95),
                "p99": pctl(lat, 99),
                "max": max(lat),
            }
        )
    return out


def stage_rows(runs: dict, qids: list[str]) -> list[dict]:
    out = []
    for name, run in runs.items():
        rs = [run["rows"][q] for q in qids]
        row = {"run": name, "n": len(rs)}
        for s in STAGES:
            vals = [(r.get("stage_latencies") or {}).get(s, 0.0) or 0.0 for r in rs]
            row[f"{s}_mean_s"] = float(np.mean(vals))
            row[f"{s}_share"] = float(np.sum(vals) / max(np.sum([latency_of(r) for r in rs]), 1e-9))
        row["correction_rate"] = float(
            np.mean([r.get("corrective_action") not in (None, "correct") for r in rs])
        )
        out.append(row)
    return out


def calls_rows(runs: dict, qids: list[str]) -> list[dict]:
    out = []
    for name, run in runs.items():
        n = len(qids)
        calls = [c for q in qids for c in run["calls_by_query"].get(q, [])]
        for kind in sorted({c["kind"] for c in calls}):
            cs = [c for c in calls if c["kind"] == kind]
            walls = [c["wall"] for c in cs]
            ttft = [c["ttft"] for c in cs if c.get("ttft") is not None]
            decode = [c["wall"] - c["ttft"] for c in cs if c.get("ttft") is not None]
            fin = Counter(c.get("finish_reason") for c in cs)
            out.append(
                {
                    "run": name,
                    "kind": kind,
                    "stage": KIND_STAGE.get(kind, "other"),
                    "n_calls": len(cs),
                    "calls_per_query": len(cs) / n,
                    "mean_wall_s": np.mean(walls),
                    "p95_wall_s": pctl(walls, 95),
                    "mean_ttft_s": np.mean(ttft) if ttft else None,
                    "mean_after_first_token_s": np.mean(decode) if decode else None,
                    "mean_prompt_tokens": np.mean([c.get("prompt_tokens") or 0 for c in cs]),
                    "mean_out_tokens": np.mean([c.get("out_tokens") or 0 for c in cs]),
                    "prompt_tokens_per_query": sum(c.get("prompt_tokens") or 0 for c in cs) / n,
                    "out_tokens_per_query": sum(c.get("out_tokens") or 0 for c in cs) / n,
                    "finish_length": fin.get("length", 0),
                    "finish_stop": fin.get("stop", 0),
                    "n_errors": sum(c.get("error") is not None for c in cs),
                }
            )
    return out


def transitions(base: dict, comp: dict, qids: list[str]) -> list[dict]:
    t = Counter()
    for q in qids:
        a, c = base["rows"][q], comp["rows"][q]
        t[(c.get("corrective_action"), int(a["acc"]), int(c["acc"]))] += 1
    return [
        {"comp_action": k[0], f"{base['name']}_acc": k[1], f"{comp['name']}_acc": k[2], "n": n}
        for k, n in sorted(t.items(), key=lambda x: (str(x[0][0]), x[0][1:]))
    ]


# ---------------------------------------------------------------- texts for gold check
def load_texts(runs: dict, docs_path: Path | None) -> dict[str, str]:
    texts: dict[str, str] = {}
    if docs_path and docs_path.exists():
        for line in docs_path.open(encoding="utf-8"):
            r = json.loads(line)
            texts[str(r["id"])] = r["text"]
    for run in runs.values():
        for w in read_jsonl(Path(run["dir"]) / "web_cache.jsonl"):
            for d in w.get("docs", []):
                texts.setdefault(d["id"], d["text"])
    return texts


def gold_in(ids, gold, texts) -> bool | None:
    if not ids:
        return False
    found = [texts.get(i) for i in ids]
    if all(t is None for t in found):
        return None  # 문서 원문을 찾지 못함 (판정 불가)
    blob = " ".join(normalize_answer(t or "") for t in found)
    return any(normalize_answer(g) and normalize_answer(g) in blob for g in gold)


def doc_scores(row: dict) -> list:
    g = (row.get("extra") or {}).get("grading") or {}
    return g.get("scores_full") or row.get("grader_scores") or []


def regression_analysis(base: dict, comp: dict, qids: list[str], texts: dict) -> dict:
    cases, cls = [], Counter()
    bands, score_hist = Counter(), Counter()
    for q in qids:
        c = comp["rows"][q]
        ids = c.get("retrieval_doc_ids") or []
        scores = doc_scores(c)
        for i, s in zip(ids, scores, strict=False):
            if s is None:
                continue
            score_hist[round(float(s), 2)] += 1
            g = gold_in([i], c.get("gold_answers") or [], texts)
            if g:
                bands[">=0.7" if s >= 0.7 else "0.3-0.7" if s >= 0.3 else "<0.3"] += 1
    for q in qids:
        a, c = base["rows"][q], comp["rows"][q]
        if not (a["acc"] and not c["acc"]):
            continue
        gold = c.get("gold_answers") or []
        act = c.get("corrective_action")
        g_base = gold_in(a.get("contexts") or [], gold, texts)
        g_comp = gold_in(c.get("contexts") or [], gold, texts)
        extra = c.get("extra") or {}
        web_ids = extra.get("web_doc_ids")
        if g_base is None or g_comp is None:
            label = "판정 불가 (문서 원문 없음)"
        elif g_comp:
            label = "근거 있음, 생성 오답"
        elif not g_base:
            label = "기준 run 근거에도 정답 문자열 없음 (모델 지식 등)"
        elif act == "correct":
            label = "Correct: 0.7 필터로 정답 문서 제거"
        elif act == "incorrect":
            label = "Incorrect: FAISS 문서 폐기 + web 근거에 정답 없음"
        else:
            label = "Ambiguous: 정답 문서 제거/미포함"
        cls[label] += 1
        cases.append(
            {
                "query_id": q,
                "action": act,
                "class": label,
                "gold": " | ".join(gold[:3]),
                f"{base['name']}_answer": (a.get("answer") or "")[:120],
                f"{comp['name']}_answer": (c.get("answer") or "")[:120],
                "grader_scores": json.dumps(scores),
                "n_contexts": len(c.get("contexts") or []),
                "gold_in_base_ctx": g_base,
                "gold_in_comp_ctx": g_comp,
                "rewritten_query": (extra.get("rewritten_query") or "")[:120],
                "n_web_docs": None if web_ids is None else len(web_ids),
            }
        )
    outs = [
        c.get("output_text")
        for q in qids
        for c in comp["calls_by_query"].get(q, [])
        if c["kind"] == "grade" and c.get("output_text") is not None
    ]
    fmt = None
    if outs:
        fmt = {
            "n": len(outs),
            "number_only_rate": float(np.mean([bool(NUM_ONLY.match(o)) for o in outs])),
            "no_digit_rate": float(np.mean([re.search(r"\d", o) is None for o in outs])),
            "non_numeric_examples": [o[:100] for o in outs if not NUM_ONLY.match(o)][:8],
        }
    web_empty = sum(
        1
        for q in qids
        if comp["rows"][q].get("corrective_action") in ("incorrect", "ambiguous")
        and ((comp["rows"][q].get("extra") or {}).get("web_doc_ids") == [])
    )
    return {
        "base": base["name"],
        "comp": comp["name"],
        "n_regressions": len(cases),
        "n_improvements": sum(
            1 for q in qids if not base["rows"][q]["acc"] and comp["rows"][q]["acc"]
        ),
        "classes": dict(cls),
        "gold_doc_score_bands": dict(bands),
        "score_value_histogram": {str(k): v for k, v in sorted(score_hist.items())},
        "grade_output_format": fmt
        or "output_text 미기록 (run.record_texts=false): 파싱 형식 판정 불가",
        "corrections_with_empty_web": web_empty,
        "gold_check_available": bool(texts),
        "cases": cases,
    }


# ---------------------------------------------------------------- batch diagnosis
def batch_diagnosis(runs: dict, qids: list[str], full: str, batch: str) -> dict:
    def grading(run):
        per_q = []
        for q in qids:
            r = run["rows"][q]
            cs = [c for c in run["calls_by_query"].get(q, []) if c["kind"] in GRADE_KINDS]
            info = ((r.get("extra") or {}).get("grading") or {}).get("info") or {}
            per_q.append(
                {
                    "stage_s": (r.get("stage_latencies") or {}).get("grading", 0.0),
                    "n_calls": len(cs),
                    "n_batch_calls": sum(c["kind"] == "grade_batch" for c in cs),
                    "n_fallback": len(info.get("fallback_indices") or []),
                    "statuses": info.get("batch_status") or [],
                    "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in cs),
                    "out_tokens": sum(c.get("out_tokens") or 0 for c in cs),
                    "ttft_sum": sum(c.get("ttft") or 0 for c in cs),
                    "decode_sum": sum(
                        (c["wall"] - c["ttft"]) for c in cs if c.get("ttft") is not None
                    ),
                    "wall_sum": sum(c["wall"] for c in cs),
                    "finish_length": sum(c.get("finish_reason") == "length" for c in cs),
                }
            )
        return per_q

    f, b = grading(runs[full]), grading(runs[batch])

    def m(rows, k):
        return float(np.mean([r[k] for r in rows])) if rows else None

    keys = (
        "stage_s",
        "n_calls",
        "n_batch_calls",
        "n_fallback",
        "prompt_tokens",
        "out_tokens",
        "ttft_sum",
        "decode_sum",
        "wall_sum",
        "finish_length",
    )
    table = {k: {full: m(f, k), batch: m(b, k)} for k in keys}
    statuses = Counter(s for r in b for s in r["statuses"])
    bc = [
        c
        for q in qids
        for c in runs[batch]["calls_by_query"].get(q, [])
        if c["kind"] == "grade_batch"
    ]
    fc = [c for q in qids for c in runs[full]["calls_by_query"].get(q, []) if c["kind"] == "grade"]

    def ratio(k):
        a, c = table[k][full], table[k][batch]
        return None if not a else c / a

    findings = []
    if (
        table["stage_s"][batch]
        and table["stage_s"][full]
        and table["stage_s"][batch] > table["stage_s"][full]
    ):
        findings.append(
            f"grading 단계 평균 {table['stage_s'][full]:.2f}s -> "
            f"{table['stage_s'][batch]:.2f}s (x{ratio('stage_s'):.2f})"
        )
    if ratio("out_tokens") and ratio("out_tokens") > 1.2:
        findings.append(
            f"query 당 grading 출력 토큰 x{ratio('out_tokens'):.2f}: "
            "출력 생성 (decode) 은 토큰마다 순차라 출력이 길면 지연이 커짐"
        )
    if ratio("decode_sum") and ratio("decode_sum") > 1.2:
        findings.append(f"첫 토큰 이후 시간 (decode) 합 x{ratio('decode_sum'):.2f}")
    if table["finish_length"][batch]:
        findings.append(
            f"batch 호출이 max_tokens 에 도달 (finish=length) 평균 "
            f"{table['finish_length'][batch]:.2f}회/query: JSON 뒤에 설명을 계속 생성"
        )
    if table["n_fallback"][batch]:
        findings.append(
            f"파싱 실패로 개별 재평가 평균 {table['n_fallback'][batch]:.2f}회/query "
            f"(status: {dict(statuses)})"
        )
    if ratio("ttft_sum") and ratio("ttft_sum") > 1.0:
        findings.append(
            f"첫 토큰까지 시간 (prefill) 합 x{ratio('ttft_sum'):.2f}: 긴 prompt 하나의 "
            "prefill 이 짧은 prompt 여러 개보다 느릴 수 있음"
        )
    if ratio("prompt_tokens") is not None:
        findings.append(
            f"query 당 grading 입력 토큰 x{ratio('prompt_tokens'):.2f} "
            "(지시문·질문 반복 제거분만 감소)"
        )
    return {
        "full": full,
        "batch": batch,
        "per_query_mean": table,
        "ratios_batch_over_full": {k: ratio(k) for k in keys},
        "batch_status_counts": dict(statuses),
        "batch_call": {
            "n": len(bc),
            "mean_wall_s": float(np.mean([c["wall"] for c in bc])) if bc else None,
            "mean_out_tokens": float(np.mean([c.get("out_tokens") or 0 for c in bc]))
            if bc
            else None,
            "output_examples": [c.get("output_text")[:200] for c in bc if c.get("output_text")][:5],
        },
        "full_grade_call": {
            "n": len(fc),
            "mean_wall_s": float(np.mean([c["wall"] for c in fc])) if fc else None,
            "mean_out_tokens": float(np.mean([c.get("out_tokens") or 0 for c in fc]))
            if fc
            else None,
        },
        "findings": findings,
        "note": "지표 기반 기술. 원인 확정은 출력 원문 (record_texts) 과 엔진 지표로 교차 확인",
    }


# ---------------------------------------------------------------- output
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "(없음)\n"
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))

    def fmt(v):
        if isinstance(v, float | np.floating):
            return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
        return "" if v is None else str(v)

    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n" for r in rows)


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--base", default=None, help="run 폴더 기준 경로")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=REL_DIR")
    ap.add_argument("--baseline", default=None, help="정답 변화 기준 run (예: vanilla)")
    ap.add_argument("--reference", default=None, help="grading 비교 기준 run (예: full)")
    ap.add_argument("--batch-run", default="batch")
    ap.add_argument("--docs-path", default=None, help="chunk jsonl (기본: run config)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
    a = ap.parse_args(argv)

    preset = PRESETS.get(a.preset, {}) if a.preset else {}
    base = Path(a.base or preset.get("base", "."))
    if not base.is_absolute():
        base = ROOT / base
    specs = dict(preset.get("runs", {}))
    for s in a.run:
        k, _, v = s.partition("=")
        specs[k] = v
    if len(specs) < 2:
        ap.error("run 이 2개 이상 필요 (--preset 또는 --run NAME=DIR)")
    runs = {k: load_run(k, base / v) for k, v in specs.items()}
    baseline = a.baseline or preset.get("baseline") or next(iter(runs))
    reference = a.reference or preset.get("reference") or list(runs)[1]

    qids = sorted(set.intersection(*(set(r["rows"]) for r in runs.values())))
    order = [q for q in runs[baseline]["rows"] if q in set(qids)]
    missing = {k: sorted(set(r["rows"]) - set(qids)) for k, r in runs.items()}

    pairs = [(baseline, k) for k in runs if k != baseline]
    pairs += [(reference, k) for k in runs if k not in (baseline, reference)]
    others = [k for k in runs if k not in (baseline, reference)]
    pairs += [(x, y) for i, x in enumerate(others) for y in others[i + 1 :]]
    acc_pairs, lat_pairs = paired_tables(runs, order, pairs, a.bootstrap, a.seed)

    docs_path = Path(a.docs_path) if a.docs_path else None
    if docs_path is None:
        dp = (runs[reference]["cfg"].get("retriever") or {}).get("docs_path")
        docs_path = (ROOT / dp) if dp and not Path(dp).is_absolute() else (Path(dp) if dp else None)
    texts = load_texts(runs, docs_path)
    regression = regression_analysis(runs[baseline], runs[reference], order, texts)
    batch = (
        batch_diagnosis(runs, order, reference, a.batch_run)
        if a.batch_run in runs and reference in runs
        else None
    )

    name = a.name or (a.preset or "custom")
    out = Path(a.out_root) / name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    tables = {
        "accuracy": accuracy_rows(runs, order),
        "paired_accuracy": acc_pairs,
        "latency": latency_rows(runs, order),
        "paired_latency": lat_pairs,
        "stages": stage_rows(runs, order),
        "calls_by_kind": calls_rows(runs, order),
        "transitions": transitions(runs[baseline], runs[reference], order),
        "regression_cases": regression["cases"],
    }
    for k, rows in tables.items():
        write_csv(out / f"{k}.csv", rows)
    per_query = []
    for q in order:
        row = {"query_id": q}
        for k, r in runs.items():
            x = r["rows"][q]
            row.update(
                {
                    f"{k}_acc": x["acc"],
                    f"{k}_latency": latency_of(x),
                    f"{k}_action": x.get("corrective_action"),
                    f"{k}_n_calls": (x.get("calls_summary") or {}).get("n_calls"),
                }
            )
        per_query.append(row)
    write_csv(out / "per_query.csv", per_query)
    meta = {
        "created": datetime.now().isoformat(),
        "purpose_note": "smoke 결과 분석. 논문 결과 아님",
        "base": str(base),
        "runs": {k: run_info(r) for k, r in runs.items()},
        "excluded": preset.get("excluded", {}),
        "baseline": baseline,
        "reference": reference,
        "n_common_queries": len(order),
        "missing_from_common": missing,
        "bootstrap": {"B": a.bootstrap, "seed": a.seed, "method": "paired, query 복원 추출"},
        "docs_path": str(docs_path) if docs_path else None,
        "gold_check_available": bool(texts),
    }
    (out / "runs.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out / "regression.json").write_text(
        json.dumps(
            {k: v for k, v in regression.items() if k != "cases"}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    if batch:
        (out / "batch_diagnosis.json").write_text(
            json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    md = [
        f"# RAG run 비교 분석 ({name})",
        "",
        f"- 공통 query {len(order)}개 (오류·warmup 제외)",
        f"- 기준 {baseline}, grading 기준 {reference}",
        f"- paired bootstrap B={a.bootstrap}, seed={a.seed}. smoke 결과이며 논문 결과로 쓰지 않음",
        f"- 제외 run: {preset.get('excluded', {})}",
        "",
        "## 정확도",
        md_table(tables["accuracy"]),
        "## 정확도 차이 (paired bootstrap 95% CI)",
        md_table(acc_pairs),
        "## 지연시간",
        md_table(tables["latency"]),
        "## 지연 차이 (평균·p95, paired bootstrap 95% CI)",
        md_table(lat_pairs),
        "## 단계별 지연",
        md_table(tables["stages"]),
        "## 호출 종류별 호출 수·지연·토큰",
        md_table(tables["calls_by_kind"]),
        f"## 정답 변화 ({baseline} -> {reference}, corrective action 별)",
        md_table(tables["transitions"]),
        f"## {baseline} 정답 -> {reference} 오답 원인 (로그 기반)",
        f"- 회귀 {regression['n_regressions']}건, 개선 {regression['n_improvements']}건",
        f"- 분류: {regression['classes']}",
        f"- 정답 포함 문서의 grader 점수 구간: {regression['gold_doc_score_bands']}",
        f"- correction 인데 web 결과 0개: {regression['corrections_with_empty_web']}",
        f"- grade 출력 형식: {regression['grade_output_format']}",
        "",
    ]
    if batch:
        md += [
            "## Batch grading: 호출 수 감소 vs 지연 증가",
            "",
            md_table([{"metric": k, **v} for k, v in batch["per_query_mean"].items()]),
            "",
            *[f"- {x}" for x in batch["findings"]],
            "",
        ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:12]))
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
