"""CRAG 로그 진단 (읽기 전용): grading 점수·출력 길이, rewrite 이상 징후, web 문서.

기존 run 폴더의 파일만 읽는다. LLM / Tavily 호출, run 폴더 쓰기는 하지 않는다.
결과는 새 폴더 results/crag_system/analysis/log_diag/<name>/<timestamp>/ 에만 쓴다.

    uv run python experiments/crag_system/diag_crag_logs.py --preset a1-smoke
    # 또는
    uv run python experiments/crag_system/diag_crag_logs.py \
        --vanilla results/crag_system/smoke/SMOKE-A1-VANILLA/20261110_103018 \
        --full results/crag_system/smoke/SMOKE-A1-FULL-RETRY/20261110_111726

분석 항목 (모두 query 집단별: 회귀 = Vanilla 정답 & Full 오답 / Full 정답 / 둘 다 오답)
  1. grading 점수 분포: 0.0, (0, lower), [lower, upper), >= upper
  2. grading 호출 출력 토큰 수: 평균·p50·p95·최대, 긴 출력 사례
  3. corrective action 별 grading 점수 특성
  4. query rewrite 이상 징후: 설명 문구, 따옴표, 여러 줄, 지나치게 긴 검색어
  5. web 문서 길이, 빈 본문, 정답 문자열 포함 여부

근거 종류 (모든 표에 basis 열)
  [로그]  로그에 기록된 값을 센 것: 점수 (scores_full), action, out_tokens (엔진 usage),
          finish_reason, rewrite 원문과 길이, web 문서 수·글자 수.
  [로그*] 로그 값이지만 grade 호출과 문서의 대응을 '순차 평가에서 호출 순서 = 문서 순서' 로 가정.
  [추정]  규칙으로 만든 참고 지표: 정답 문자열 포함 (정규화 + 단어 경계), rewrite 이상 징후
          정규식, '긴 출력' 기준 (--long-tokens). 확정 판정이 아니다.
  점수 0.0 은 모델이 0 을 출력한 것과 숫자가 없어 parse_score 가 0 으로 바꾼 것을 구분할 수 없다.
  grade 출력 원문 (run.record_texts=true) 이 있을 때만 '숫자 없음' 비율을 [로그] 로 센다.

출력
  report.md
  score_distribution.csv     집단 x 점수 구간 (문서 단위)
  score_values.csv           집단 x 점수 값 빈도 (상위)
  score_by_action.csv        집단 x action: 점수 평균·최대·구간 비율
  grade_tokens.csv           집단별 grading 출력 토큰 통계
  grade_tokens_by_band.csv   점수 구간별 grading 출력 토큰 (호출-문서 대응 가능할 때)
  long_grade_calls.csv       출력 토큰이 긴 grading 호출 (상위 --top-long 개)
  rewrite_cases.csv          correction query 별 rewrite 원문·길이·이상 징후
  rewrite_summary.csv        집단 x 이상 징후 비율
  web_docs.csv               web 문서 단위: 글자 수, 빈 본문, 정답 문자열
  web_summary.csv            집단 x action 별 web 문서 통계
  diag_meta.json             입력, 데이터 가용성, 기준값
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_regression import (  # noqa: E402
    PRESETS,
    ROOT,
    full_view,
    gold_norms,
    has_gold,
    load_web,
    md_cell,
    resolve,
    run_web_sources,
)
from analyze_runs import load_run, md_table, write_csv  # noqa: E402

from rag.eval.metrics import normalize_answer  # noqa: E402

GROUPS = ("regression", "full_correct", "both_wrong")
GROUP_KO = {
    "regression": "회귀 (V 정답 -> F 오답)",
    "full_correct": "Full 정답",
    "both_wrong": "둘 다 오답",
    "all": "전체",
}
ACTIONS = ("correct", "ambiguous", "incorrect")
GRADE_KINDS = ("grade", "grade_batch")

PREFACE = re.compile(
    r"^\s*(?:here(?:'s| is| are)\b|sure\b|certainly\b|okay\b|ok\b|of course\b|"
    r"i (?:would|will|'d|can)\b|let me\b|"
    r"(?:the )?(?:rewritten|revised|concise|web|search)(?: web)?(?: search)? (?:query|question)\b|"
    r"query\s*:)",
    re.I,
)
EXPLAIN = re.compile(
    r"\b(?:this (?:query|search)|will (?:help|return|find)|in order to|to find (?:the|out)|"
    r"note\s*:|explanation|because|i (?:removed|added|used|changed))\b",
    re.I,
)
# 단어 안의 apostrophe (O'Brien, Earth's) 는 제외
QUOTES = re.compile(r"[\"“”`«»]|(?<![A-Za-z])['‘’]|['‘’](?![A-Za-z])")
MARKDOWN = re.compile(r"\*\*|^\s*(?:[-*•]|\d+[.)])\s|^\s*#", re.M)
REWRITE_FLAGS = {
    "empty": "rewrite 가 빈 문자열",
    "preface": "앞머리 안내 문구 (Here is / Sure / Search query: 등)",
    "explanation": "설명 문구 (this query will / because / note: 등)",
    "quotes": "따옴표·backtick 포함 (단어 안 apostrophe 제외)",
    "multiline": "여러 줄",
    "markdown": "markdown (굵게, 목록 기호, 번호)",
    "long_chars": "글자 수 > --rewrite-max-chars",
    "long_words": "단어 수 > --rewrite-max-words",
    "same_as_question": "정규화하면 원 질문과 같음 (rewrite 안 됨)",
}


# ---------------------------------------------------------------- helpers
def group_of(v: dict, f: dict) -> str:
    if f.get("acc"):
        return "full_correct"
    return "regression" if v.get("acc") else "both_wrong"


def band(s, upper: float, lower: float) -> str:
    if s is None:
        return "not_graded"
    if s == 0.0:
        return "0.0"
    if s < lower:
        return f"(0,{lower})"
    return f"[{lower},{upper})" if s < upper else f">={upper}"


def band_order(upper: float, lower: float) -> list[str]:
    return ["0.0", f"(0,{lower})", f"[{lower},{upper})", f">={upper}", "not_graded"]


def stats(xs) -> dict:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    a = np.asarray(xs)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def share(n: int, d: int):
    return n / d if d else None


def rewrite_flags(raw: str | None, question: str, max_chars: int, max_words: int) -> list[str]:
    t = (raw or "").strip()
    flags = []
    if not t:
        return ["empty"]
    if PREFACE.search(t):
        flags.append("preface")
    if EXPLAIN.search(t):
        flags.append("explanation")
    if QUOTES.search(t):
        flags.append("quotes")
    if "\n" in t:
        flags.append("multiline")
    if MARKDOWN.search(t):
        flags.append("markdown")
    if len(t) > max_chars:
        flags.append("long_chars")
    if len(t.split()) > max_words:
        flags.append("long_words")
    if normalize_answer(t) == normalize_answer(question or ""):
        flags.append("same_as_question")
    return flags


def grade_calls_for_docs(calls: list[dict], scores: list, mappable: bool) -> list[dict | None]:
    """평가된 문서 순서에 grade 호출을 대응. 대응을 보장할 수 없으면 None 들."""
    out: list[dict | None] = [None] * len(scores)
    cs = sorted((c for c in calls if c.get("kind") == "grade"), key=lambda c: c.get("idx", 0))
    graded = [i for i, s in enumerate(scores) if s is not None]
    if mappable and cs and len(cs) == len(graded):
        for i, c in zip(graded, cs, strict=True):
            out[i] = c
    return out


# ---------------------------------------------------------------- analysis
def collect(van: dict, full: dict, order: list[str], ctx: dict) -> dict:
    upper, lower = ctx["upper"], ctx["lower"]
    docs, calls, rewrites, webs, queries = [], [], [], [], []
    for q in order:
        v, f = van["rows"][q], full["rows"][q]
        g = group_of(v, f)
        act = f.get("corrective_action")
        fv = full_view(f)
        golds = gold_norms(f.get("gold_answers") or [])
        f_calls = full["calls_by_query"].get(q, [])
        graded = [s for s in fv["scores"] if s is not None]
        queries.append(
            {
                "query_id": q,
                "group": g,
                "action": act,
                "max_score": max(graded) if graded else None,
                "mean_score": float(np.mean(graded)) if graded else None,
                "n_graded": len(graded),
                "n_ge_upper": sum(s >= upper for s in graded),
                "n_zero": sum(s == 0.0 for s in graded),
            }
        )
        mapped = grade_calls_for_docs(f_calls, fv["scores"], ctx["mappable"])
        for r, (i, s) in enumerate(zip(fv["ret_ids"], fv["scores"], strict=True)):
            c = mapped[r]
            docs.append(
                {
                    "query_id": q,
                    "group": g,
                    "action": act,
                    "rank": r + 1,
                    "doc_id": i,
                    "score": s,
                    "band": band(s, upper, lower),
                    "grade_out_tokens": None if c is None else c.get("out_tokens"),
                    "grade_output": None if c is None else c.get("output_text"),
                }
            )
        for c in f_calls:
            if c.get("kind") not in GRADE_KINDS:
                continue
            o = c.get("output_text")
            calls.append(
                {
                    "query_id": q,
                    "group": g,
                    "action": act,
                    "kind": c.get("kind"),
                    "idx": c.get("idx"),
                    "out_tokens": c.get("out_tokens"),
                    "prompt_tokens": c.get("prompt_tokens"),
                    "finish_reason": c.get("finish_reason"),
                    "wall": c.get("wall"),
                    "error": c.get("error"),
                    "output_text": o,
                    "output_has_digit": None if o is None else bool(re.search(r"\d", o)),
                }
            )
        if act in (None, "correct"):
            continue
        raw = fv["rewritten"]
        if raw is None:  # 기존 crag pipeline: rewrite 호출 출력 원문 (record_texts 일 때만)
            rc = [c for c in f_calls if c.get("kind") == "rewrite"]
            raw = rc[-1].get("output_text") if rc else None
        t = (raw or "").strip()
        sent = ctx["web_by_query"].get(t)
        flags = (
            rewrite_flags(raw, f.get("question") or "", ctx["max_chars"], ctx["max_words"])
            if raw is not None
            else []
        )
        rewrites.append(
            {
                "query_id": q,
                "group": g,
                "action": act,
                "question": f.get("question"),
                "rewrite_raw": raw,
                "rewrite_available": raw is not None,
                "chars": len(t) if raw is not None else None,
                "words": len(t.split()) if raw is not None else None,
                "lines": len(t.splitlines()) if raw is not None else None,
                "found_in_web_cache": None if raw is None else sent is not None,
                "flags": ";".join(flags),
            }
        )
        by_q = {str(d["id"]): d for d in (sent or [])}
        for r, i in enumerate(fv["web_ids"]):
            d = by_q.get(i)
            text = d.get("text") if d else ctx["web_by_id"].get(i)
            webs.append(
                {
                    "query_id": q,
                    "group": g,
                    "action": act,
                    "rank": r + 1,
                    "doc_id": i,
                    "text_found": text is not None,
                    "chars": None if text is None else len(text),
                    "empty_text": None if text is None else not text.strip(),
                    "title": ((d or {}).get("metadata") or {}).get("title"),
                    "gold_string": has_gold(text, golds),
                    "text_head": None if text is None else text[:200],
                }
            )
        if not fv["web_ids"]:
            webs.append(
                {"query_id": q, "group": g, "action": act, "rank": 0, "doc_id": None,
                 "text_found": None, "no_web_docs": True}
            )  # fmt: skip
    return {"docs": docs, "calls": calls, "rewrites": rewrites, "webs": webs, "queries": queries}


def by_group(rows: list[dict], with_all: bool = True) -> dict[str, list[dict]]:
    out = {g: [r for r in rows if r["group"] == g] for g in GROUPS}
    if with_all:
        out["all"] = rows
    return out


def score_distribution(docs, upper, lower) -> list[dict]:
    rows = []
    for g, ds in by_group(docs).items():
        graded = [d for d in ds if d["score"] is not None]
        c = Counter(d["band"] for d in ds)
        for b in band_order(upper, lower):
            rows.append(
                {
                    "group": g,
                    "band": b,
                    "n_docs": c.get(b, 0),
                    "share_of_graded": None
                    if b == "not_graded"
                    else share(c.get(b, 0), len(graded)),
                    "basis": "로그",
                }
            )
    return rows


def score_values(docs, top: int = 10) -> list[dict]:
    rows = []
    for g, ds in by_group(docs).items():
        c = Counter(d["score"] for d in ds if d["score"] is not None)
        n = sum(c.values())
        for val, k in c.most_common(top):
            rows.append(
                {"group": g, "score": val, "n_docs": k, "share": share(k, n), "basis": "로그"}
            )
    return rows


def score_by_action(docs, queries, upper, lower) -> list[dict]:
    rows = []
    for g, qs in by_group(queries).items():
        for act in ACTIONS:
            qa = [q for q in qs if q["action"] == act]
            ids = {q["query_id"] for q in qa}
            ds = [d for d in (docs if g == "all" else [d for d in docs if d["group"] == g])
                  if d["query_id"] in ids and d["score"] is not None]  # fmt: skip
            c = Counter(d["band"] for d in ds)
            row = {
                "group": g,
                "action": act,
                "n_queries": len(qa),
                "n_graded_docs": len(ds),
                "mean_score": stats([d["score"] for d in ds])["mean"],
                "mean_max_score": stats([q["max_score"] for q in qa])["mean"],
                "mean_n_ge_upper": stats([q["n_ge_upper"] for q in qa])["mean"],
                "queries_all_zero": sum(
                    q["n_graded"] > 0 and q["n_zero"] == q["n_graded"] for q in qa
                ),
            }
            for b in band_order(upper, lower)[:-1]:
                row[f"share_{b}"] = share(c.get(b, 0), len(ds))
            row["basis"] = "로그"
            rows.append(row)
    return rows


def grade_tokens(calls, long_tokens: int) -> list[dict]:
    rows = []
    for g, cs in by_group(calls).items():
        toks = [c["out_tokens"] for c in cs]
        known = [t for t in toks if t is not None]
        txt = [c for c in cs if c["output_has_digit"] is not None]
        s = stats(known)
        rows.append(
            {
                "group": g,
                "n_calls": len(cs),
                "n_out_tokens_known": len(known),
                "mean": s["mean"],
                "p50": s["p50"],
                "p95": s["p95"],
                "max": s["max"],
                f"share_gt_{long_tokens}_tokens": share(
                    sum(t > long_tokens for t in known), len(known)
                ),
                "finish_length": sum(c["finish_reason"] == "length" for c in cs),
                "n_errors": sum(c["error"] is not None for c in cs),
                "n_output_text": len(txt),
                "share_no_digit": share(sum(not c["output_has_digit"] for c in txt), len(txt)),
            }
        )
    return rows


def grade_tokens_by_band(docs, upper, lower) -> list[dict]:
    rows = []
    for g, ds in by_group(docs).items():
        for b in band_order(upper, lower)[:-1]:
            t = [d["grade_out_tokens"] for d in ds if d["band"] == b]
            s = stats(t)
            rows.append({"group": g, "band": b, **s, "basis": "로그*"})
    return rows


def rewrite_summary(rewrites) -> list[dict]:
    rows = []
    for g, rs in by_group(rewrites).items():
        av = [r for r in rs if r["rewrite_available"]]
        row = {
            "group": g,
            "n_corrected": len(rs),
            "n_rewrite_available": len(av),
            "chars_mean": stats([r["chars"] for r in av])["mean"],
            "chars_p95": stats([r["chars"] for r in av])["p95"],
            "words_mean": stats([r["words"] for r in av])["mean"],
            "words_max": stats([r["words"] for r in av])["max"],
            "share_multiline_log": share(sum(r["lines"] > 1 for r in av), len(av)),
            "share_not_in_web_cache": share(
                sum(r["found_in_web_cache"] is False for r in av), len(av)
            ),
            "share_any_flag": share(sum(bool(r["flags"]) for r in av), len(av)),
        }
        for k in REWRITE_FLAGS:
            row[f"share_{k}"] = share(sum(k in r["flags"].split(";") for r in av), len(av))
        rows.append(row)
    return rows


def web_summary(webs) -> list[dict]:
    rows = []
    for g, ws in by_group(webs).items():
        for act in ("ambiguous", "incorrect", "all"):
            wa = [w for w in ws if act == "all" or w["action"] == act]
            qids = {w["query_id"] for w in wa}
            docs = [w for w in wa if w.get("doc_id") is not None]
            found = [w for w in docs if w["text_found"]]
            per_q = defaultdict(list)
            for w in docs:
                per_q[w["query_id"]].append(w)
            known_q = [q for q, ds in per_q.items() if all(d["text_found"] for d in ds)]
            s = stats([w["chars"] for w in found])
            rows.append(
                {
                    "group": g,
                    "action": act,
                    "n_queries": len(qids),
                    "queries_no_web_docs": sum(1 for w in wa if w.get("no_web_docs")),
                    "n_web_docs": len(docs),
                    "docs_per_query": share(len(docs), len(qids)),
                    "n_text_found": len(found),
                    "chars_mean": s["mean"],
                    "chars_p50": s["p50"],
                    "chars_p95": s["p95"],
                    "share_empty_text": share(sum(w["empty_text"] for w in found), len(found)),
                    "est_share_docs_gold": share(
                        sum(bool(w["gold_string"]) for w in found), len(found)
                    ),
                    "est_share_queries_any_gold": share(
                        sum(any(d["gold_string"] for d in per_q[q]) for q in known_q), len(known_q)
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------- report
def table_for(rows, cols, groups=("regression", "full_correct")):
    return md_table(
        [{k: md_cell(r.get(k)) for k in cols} for r in rows if r["group"] in groups], cols
    )


def build_report(m: dict, t: dict, a) -> str:
    upper, lower = m["upper"], m["lower"]
    bands = band_order(upper, lower)
    n = m["group_sizes"]
    dist = defaultdict(dict)
    for r in t["score_distribution"]:
        dist[r["group"]][r["band"]] = r
    dist_rows = [
        {"group": GROUP_KO[g], "n_graded_docs": sum(dist[g][b]["n_docs"] for b in bands[:-1]),
         **{b: f"{dist[g][b]['n_docs']} ({100 * (dist[g][b]['share_of_graded'] or 0):.1f}%)"
            for b in bands[:-1]}}
        for g in ("regression", "full_correct", "both_wrong", "all")
    ]  # fmt: skip
    long_cols = ["query_id", "group", "action", "idx", "out_tokens", "finish_reason", "output_text"]
    L = [
        "# CRAG 로그 진단: grading / rewrite / web 문서",
        "",
        f"- Vanilla: `{m['vanilla_dir']}`",
        f"- Full: `{m['full_dir']}`",
        f"- 공통 query {m['n_common']}개 (오류·warmup 제외). upper={upper}, lower={lower}",
        f"- 집단: 회귀 {n['regression']} / Full 정답 {n['full_correct']} / "
        f"둘 다 오답 {n['both_wrong']}",
        "- 새 추론·Tavily 호출 없이 기존 로그만 읽어 계산했다.",
        "",
        "## 근거 종류",
        "",
        "- **[로그]** 로그에 기록된 값을 센 것 (점수, action, out_tokens, finish_reason, "
        "rewrite 원문·길이, web 문서 수·글자 수). 집계 자체는 확정 사실이다.",
        "- **[로그\\*]** 로그 값이지만 grade 호출과 문서를 "
        "'순차 평가에서 호출 순서 = 문서 순서' 로 "
        f"대응시킨 것. 현재: {m['availability']['mapping']}",
        "- **[추정]** 규칙 기반 참고 지표: 정답 문자열 포함 여부, rewrite 이상 징후 정규식, "
        f"'긴 출력' 기준 (> {a.long_tokens} 토큰). 원인의 확정 판정이 아니다.",
        "- 로그 사실이어도 원인은 아니다. 예: 회귀에 0.0 점수가 많다는 것은 사실이지만, 0.0 이 "
        "파싱 실패인지 모델의 판단인지는 출력 원문이 있어야 구분된다.",
        "",
        "## 데이터 가용성",
        "",
        f"- grade 출력 원문 (record_texts): {m['availability']['record_texts_full']}, "
        f"원문 있는 grade 호출 {m['availability']['n_grade_output_text']}개",
        f"- grade 호출 out_tokens 기록: {m['availability']['n_out_tokens_known']} / "
        f"{m['availability']['n_grade_calls']}",
        f"- web 원문 출처: {m['availability']['web_sources'] or '없음'}",
        "",
        "## 1. Grading 점수 분포 [로그]",
        "",
        "문서 단위. 괄호는 평가된 문서 중 비율.",
        "",
        md_table(dist_rows),
        "점수 값 빈도 (상위, 회귀 / Full 정답):",
        "",
        table_for(t["score_values"], ["group", "score", "n_docs", "share"]),
        "## 2. Grading 호출 출력 토큰 수 [로그]",
        "",
        table_for(
            t["grade_tokens"],
            [
                "group",
                "n_calls",
                "n_out_tokens_known",
                "mean",
                "p50",
                "p95",
                "max",
                f"share_gt_{a.long_tokens}_tokens",
                "finish_length",
                "share_no_digit",
            ],
            ("regression", "full_correct", "both_wrong", "all"),
        ),  # fmt: skip
        f"- `share_gt_{a.long_tokens}_tokens` 의 기준값은 [추정] (숫자 하나 출력은 보통 몇 토큰). "
        "`share_no_digit` 은 출력 원문이 있을 때만 [로그].",
        "",
        "점수 구간별 grade 출력 토큰 [로그\\*]:",
        "",
        table_for(t["grade_tokens_by_band"], ["group", "band", "n", "mean", "p50", "p95", "max"]),
        f"긴 출력 사례 (상위 {min(10, len(t['long_grade_calls']))}개, 전체는 "
        "`long_grade_calls.csv`):",
        "",
        md_table(
            [
                {k: md_cell(str(r.get(k))[:160]) for k in long_cols}
                for r in t["long_grade_calls"][:10]
            ],
            long_cols,
        ),  # fmt: skip
        "## 3. Corrective action 별 Grading 점수 특성 [로그]",
        "",
        table_for(
            t["score_by_action"],
            [
                "group",
                "action",
                "n_queries",
                "mean_score",
                "mean_max_score",
                "mean_n_ge_upper",
                "queries_all_zero",
                "share_0.0",
                f"share_>={upper}",
            ],
        ),  # fmt: skip
        "## 4. Query rewrite 이상 징후",
        "",
        "글자·단어·줄 수, web 기록 일치는 [로그], `share_<징후>` 는 정규식 [추정].",
        "",
        table_for(
            t["rewrite_summary"],
            [
                "group",
                "n_corrected",
                "n_rewrite_available",
                "chars_mean",
                "chars_p95",
                "words_max",
                "share_multiline_log",
                "share_not_in_web_cache",
                "share_any_flag",
            ]
            + [f"share_{k}" for k in REWRITE_FLAGS],
            ("regression", "full_correct", "both_wrong", "all"),
        ),  # fmt: skip
        "징후 정의: " + "; ".join(f"`{k}` {v}" for k, v in REWRITE_FLAGS.items()),
        "",
        "징후가 있는 rewrite 예시 (회귀):",
        "",
        md_table(
            [
                {
                    k: md_cell(str(r.get(k))[:160])
                    for k in ("query_id", "action", "rewrite_raw", "flags")
                }
                for r in t["rewrite_cases"]
                if r["group"] == "regression" and r["flags"]
            ][:10],
            ["query_id", "action", "rewrite_raw", "flags"],
        ),  # fmt: skip
        "## 5. Web 문서 길이와 정답 문자열",
        "",
        "문서 수·글자 수·빈 본문은 [로그], `est_*` (정답 문자열) 은 [추정].",
        "",
        table_for(
            t["web_summary"],
            [
                "group",
                "action",
                "n_queries",
                "queries_no_web_docs",
                "n_web_docs",
                "n_text_found",
                "chars_mean",
                "chars_p50",
                "chars_p95",
                "share_empty_text",
                "est_share_docs_gold",
                "est_share_queries_any_gold",
            ],
        ),  # fmt: skip
        "## 6. 회귀 vs Full 정답 비교 요약",
        "",
        md_table(m["comparison"], ["metric", "basis", "regression", "full_correct"]),
        "두 집단의 차이는 기술 통계이다. 집단 크기가 작고 action 구성이 다르므로 (회귀는 "
        "Incorrect 비중이 클 수 있음) action 별 표 (3, 5) 와 함께 읽는다.",
        "",
        "## 서버 실행 후 확인할 항목",
        "",
        "1. 집단 크기 (회귀 55건 등) 가 기존 분석과 같은지",
        "2. 회귀 쪽 0.0 비율이 높다면: grade 출력 원문이 없으면 파싱 실패 여부는 미확정. "
        "점수 구간별 출력 토큰 (0.0 문서의 토큰이 길면 파싱 실패 의심) 을 참고",
        "3. 긴 출력 사례의 `output_text` (있으면) 를 읽어 설명문 출력인지 확인",
        "4. rewrite 징후 사례를 원문으로 읽어 정규식 오탐 (예: 따옴표가 원래 제목인 경우) 제외",
        "5. web 정답 문자열 [추정] 은 별칭·표기 차이를 놓칠 수 있으므로 표본을 원문으로 확인",
    ]
    return "\n".join(L) + "\n"


def comparison(t: dict, upper: float, long_tokens: int) -> list[dict]:
    def get(rows, g, key, **match):
        for r in rows:
            if r["group"] == g and all(r.get(k) == v for k, v in match.items()):
                return r.get(key)
        return None

    def fmt(x):
        return None if x is None else round(float(x), 4)

    specs = [
        ("점수 0.0 비율 (문서)", "로그", "score_distribution", "share_of_graded", {"band": "0.0"}),
        (f"점수 >= {upper} 비율 (문서)", "로그", "score_distribution", "share_of_graded",
         {"band": f">={upper}"}),
        ("grade 출력 토큰 평균", "로그", "grade_tokens", "mean", {}),
        ("grade 출력 토큰 p95", "로그", "grade_tokens", "p95", {}),
        (f"grade 출력 > {long_tokens} 토큰 비율", "추정 기준", "grade_tokens",
         f"share_gt_{long_tokens}_tokens", {}),
        ("rewrite 평균 글자 수", "로그", "rewrite_summary", "chars_mean", {}),
        ("rewrite 이상 징후 비율", "추정", "rewrite_summary", "share_any_flag", {}),
        ("web 문서 평균 글자 수", "로그", "web_summary", "chars_mean", {"action": "all"}),
        ("web 에 정답 문자열 있는 query 비율", "추정", "web_summary", "est_share_queries_any_gold",
         {"action": "all"}),
    ]  # fmt: skip
    return [
        {
            "metric": name,
            "basis": basis,
            "regression": fmt(get(t[tab], "regression", key, **match)),
            "full_correct": fmt(get(t[tab], "full_correct", key, **match)),
        }
        for name, basis, tab, key, match in specs
    ]


# ---------------------------------------------------------------- main
def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--vanilla", help="Vanilla run 폴더 (집단 구분용)")
    ap.add_argument("--full", help="Full CRAG run 폴더")
    ap.add_argument("--web-cache", action="append", default=[], help="추가 web 기록 jsonl")
    ap.add_argument("--upper", type=float)
    ap.add_argument("--lower", type=float)
    ap.add_argument("--long-tokens", type=int, default=8, help="'긴 grade 출력' 기준 (추정)")
    ap.add_argument("--top-long", type=int, default=20)
    ap.add_argument("--rewrite-max-chars", type=int, default=100)
    ap.add_argument("--rewrite-max-words", type=int, default=15)
    ap.add_argument("--name", default=None)
    ap.add_argument(
        "--out-root", default=str(ROOT / "results" / "crag_system" / "analysis" / "log_diag")
    )
    a = ap.parse_args(argv)
    preset = PRESETS.get(a.preset, {}) if a.preset else {}
    v_dir, f_dir = (
        resolve(a.vanilla or preset.get("vanilla")),
        resolve(a.full or preset.get("full")),
    )
    if not v_dir or not f_dir:
        ap.error("--preset 또는 --vanilla / --full 이 필요")
    van, full = load_run("vanilla", v_dir), load_run("full", f_dir)
    crag_cfg = full["cfg"].get("crag") or {}
    upper = a.upper if a.upper is not None else crag_cfg.get("upper_threshold", 0.7)
    lower = a.lower if a.lower is not None else crag_cfg.get("lower_threshold", 0.3)
    order = [q for q in van["rows"] if q in full["rows"]]

    web_paths = [
        p for p in run_web_sources(full) + [resolve(x) for x in a.web_cache] if p and p.exists()
    ]
    web_by_query, web_by_id = load_web(web_paths)
    gcfg = full["cfg"].get("grading") or {}
    strategy, parallel = gcfg.get("strategy", "full"), bool(gcfg.get("parallel"))
    mappable = strategy in ("full", "lazy", "lazy_hit", "lazy_rest") and not parallel
    ctx = {
        "upper": upper,
        "lower": lower,
        "mappable": mappable,
        "web_by_query": web_by_query,
        "web_by_id": web_by_id,
        "max_chars": a.rewrite_max_chars,
        "max_words": a.rewrite_max_words,
    }
    data = collect(van, full, order, ctx)
    long_calls = sorted(
        (
            c
            for c in data["calls"]
            if c["out_tokens"] is not None and c["out_tokens"] > a.long_tokens
        ),
        key=lambda c: -c["out_tokens"],
    )[: a.top_long]
    tables = {
        "score_distribution": score_distribution(data["docs"], upper, lower),
        "score_values": score_values(data["docs"]),
        "score_by_action": score_by_action(data["docs"], data["queries"], upper, lower),
        "grade_tokens": grade_tokens(data["calls"], a.long_tokens),
        "grade_tokens_by_band": grade_tokens_by_band(data["docs"], upper, lower),
        "long_grade_calls": long_calls,
        "rewrite_cases": data["rewrites"],
        "rewrite_summary": rewrite_summary(data["rewrites"]),
        "web_docs": data["webs"],
        "web_summary": web_summary(data["webs"]),
    }
    sizes = Counter(q["group"] for q in data["queries"])
    mapped_n = sum(d["grade_out_tokens"] is not None for d in data["docs"])
    meta = {
        "created": datetime.now().isoformat(),
        "note": "읽기 전용 로그 진단. [추정] 지표는 참고용",
        "vanilla_dir": str(v_dir),
        "full_dir": str(f_dir),
        "upper": upper,
        "lower": lower,
        "n_common": len(order),
        "group_sizes": {g: sizes.get(g, 0) for g in GROUPS},
        "errors": {"vanilla": van["errors"], "full": full["errors"]},
        "availability": {
            "record_texts_full": bool((full["cfg"].get("run") or {}).get("record_texts")),
            "n_grade_calls": len(data["calls"]),
            "n_out_tokens_known": sum(c["out_tokens"] is not None for c in data["calls"]),
            "n_grade_output_text": sum(c["output_text"] is not None for c in data["calls"]),
            "web_sources": [str(p) for p in web_paths],
            "mapping": (
                f"대응 {mapped_n}개 문서 (strategy={strategy}, parallel={parallel})"
                if mappable
                else f"대응 불가 (strategy={strategy}, parallel={parallel}): "
                "구간별 토큰 표는 비어 있음"
            ),
        },
        "criteria": {
            "long_tokens": a.long_tokens,
            "rewrite_max_chars": a.rewrite_max_chars,
            "rewrite_max_words": a.rewrite_max_words,
        },
    }
    meta["comparison"] = comparison(tables, upper, a.long_tokens)

    name = a.name or a.preset or "custom"
    out = Path(a.out_root) / name / datetime.now().strftime("%Y%m%d_%H%M%S")
    for d in (v_dir, f_dir):  # 결과를 run 폴더 안에 쓰지 않는다
        if out.resolve().is_relative_to(d.resolve()):
            raise ValueError(f"출력 폴더가 run 폴더 안에 있음: {out}")
    out.mkdir(parents=True, exist_ok=False)
    for k, rows in tables.items():
        write_csv(out / f"{k}.csv", rows)
    with open(out / "diag_meta.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    with open(out / "report.md", "x", encoding="utf-8") as f:
        f.write(build_report(meta, tables, a))
    print(f"common={len(order)} groups={meta['group_sizes']}")
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
