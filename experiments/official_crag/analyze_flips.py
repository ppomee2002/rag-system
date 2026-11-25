"""Flip 분석 (읽기 전용): 기준 run (예: Vanilla) 과 비교 run (예: CRAG) 에서
정답 여부가 뒤집힌 문항의 원인을 Grader 오판정 / Evidence 손실 / 외부 지식 변화 /
Prompt 영향 / 평가 차이 로 잠정 분류한다.

    uv run python experiments/official_crag/analyze_flips.py \
        --baseline <Vanilla run 폴더> --treatment <CRAG run 폴더> \
        [--flip-ids <Flip-20 id 파일 (txt/csv/json)>] --name flip20

  HARM = 기준 정답 -> 비교 오답, GAIN = 기준 오답 -> 비교 정답 (run 에 기록된 acc 기준)
  --flip-ids 를 주면 그 문항만 분석한다 (기존 Flip-20 목록). 목록에 있지만 두 run 에서 뒤집히지 않은
  문항은 `not_a_flip` 으로 따로 보고한다. 없으면 뒤집힌 문항 전부.

입력 형식 (자동 판별)
  crag_system runner 결과 (queries.jsonl 의 query_id / contexts / extra.grading.scores_full)
  official_crag MODE B 결과 (schema official_crag.query.v1: evidence / grading_events / correction)
문서 원문: retriever.docs_path (chunk jsonl), web 원문: web_cache.jsonl + replay 원본
(experiments/crag_system/analyze_regression.py 의 로더와 지표를 그대로 쓴다).

분류 규칙 (primary_cause, 위에서부터 처음 맞는 것. contributing 에는 함께 성립한 요인)
  공통
    평가 차이        run 의 acc (정규화 substring) 로는 뒤집혔지만 공식 match
                     (대소문자 구분 substring) 로는 뒤집히지 않음 (또는 그 반대)
    판정 보류        검색 결과가 두 run 에서 다름 (D1) 또는 문서 원문이 없어
                     문자열 지표 판정 불가 (D2)
  HARM
    Prompt 영향      CRAG 최종 context 에 정답 문자열이 있는데 오답 (A1)
    Grader 오판정    Vanilla context 에 있던 정답 문자열 문서가 CRAG 에서 빠졌고 (E1),
                     그 문서의 grader 점수가 유지 기준 미만
                     (correct: upper, ambiguous: lower, incorrect: 전부 폐기)
    Evidence 손실    E1 인데 정답 문서 점수가 유지 기준 이상 (정책 구조로 제거) 이거나
                     평가되지 않음 (budget)
    외부 지식 변화   Vanilla context 에 정답 문자열이 없었고 (모델 지식 / 별칭으로 정답)
                     CRAG 가 web 문서로 context 를 바꾼 경우
    Prompt 영향      위에 해당 없고 web 없이 context 만 달라진 경우
  GAIN
    외부 지식 변화   Vanilla context 에 정답 문자열이 없고 web 문서에 있음 (web 이 근거를 추가)
    Grader 오판정(정) 정답 문서는 유지되고 정답 없는 문서가 제거됨 (방해 문서 제거)
    Prompt 영향      정답 문서가 양쪽에 있고 제거된 문서 없음 (context 형태·순서만 다름)
    평가 차이        정답 문자열이 어느 context 에도 없음 (모델 지식·우연 일치 가능)
문자열 지표는 확정 근거가 아니다. 결과 CSV 의 manual_cause / manual_note 에 수동 확인을 기록한다.
Prompt 영향과 근거 손실을 분리하려면 같은 context 로 생성만 바꾸는 재실행이 필요하다 (README 10.2).

출력 (results/official_crag/analysis/flips/<name>/<ts>/, 이미 있으면 실패)
  report.md            요약 + Incorrect HARM 우선 사례 카드 + 나머지 HARM + GAIN
  flip_cases.csv       문항별 사실·지표·분류·수동 확인 열
  flip_docs.csv        문항 x 문서 (FAISS / web): 순위, grader 점수, 유지 여부, 정답 문자열 여부
  cause_summary.csv    방향 x 판정 x 원인 건수
  analysis_meta.json   입력, 데이터 가용성, 규칙
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "crag_system"))
from analyze_regression import (  # noqa: E402
    analyze_query,
    full_view,
    load_chunk_texts,
    load_web,
    md_cell,
    resolve,
    run_web_sources,
)
from analyze_runs import load_run, md_table, read_jsonl, write_csv  # noqa: E402

from rag.crag_official.generation import official_match  # noqa: E402

CAUSES = {
    "grader": "Grader 오판정",
    "evidence": "Evidence 손실",
    "external": "외부 지식 변화",
    "prompt": "Prompt 영향",
    "evaluation": "평가 차이",
    "hold": "판정 보류 (데이터)",
    "other": "기타 (수동 확인)",
}
MODE_B_SCHEMA = "official_crag.query.v1"


# ---------------------------------------------------------------- loading
def _is_mode_b(run_dir: Path) -> bool:
    for r in read_jsonl(run_dir / "queries.jsonl"):
        return r.get("schema") == MODE_B_SCHEMA
    return False


def mode_b_row(r: dict) -> dict:
    """MODE B 행 -> crag_system 행 형태 (analyze_regression.full_view 가 읽는 필드)."""
    ev = r.get("evidence") or {}
    corr = r.get("correction") or {}
    ret = [str(i) for i in r.get("retrieval_doc_ids") or []]
    scores: list = [None] * len(ret)
    for e in r.get("grading_events") or []:
        rank = e.get("rank")
        if rank is not None and 0 <= rank < len(scores):
            scores[rank] = e.get("score")
    kept = [str(i) for i in ev.get("kept_retrieved_ids") or []]
    web = [str(i) for i in corr.get("web_doc_ids") or []]
    added = [str(i) for i in ev.get("added_ids") or []]
    return {
        "query_id": str(r.get("question_id")),
        "question": r.get("question"),
        "gold_answers": r.get("ground_truth") or [],
        "answer": r.get("prediction") or "",
        "acc": (r.get("metrics") or {}).get("acc", r.get("is_correct")),
        "corrective_action": (r.get("decision") or {}).get("action"),
        "retrieval_doc_ids": ret,
        "contexts": kept + added,
        "warmup": r.get("warmup"),
        "error": r.get("error"),
        "request_index": r.get("request_index"),
        "extra": {
            "retrieval_doc_ids": ret,
            "grading": {"scores_full": scores, "strategy": (r.get("budget") or {}).get("name")},
            "web_doc_ids": web,
            "rewritten_query": corr.get("query"),
        },
        "_policy_id": r.get("policy_id"),
        "_thresholds": r.get("thresholds"),
    }


def load_any(name: str, run_dir: Path) -> dict:
    """crag_system / MODE B 결과를 analyze_runs.load_run 과 같은 형태로."""
    if not _is_mode_b(run_dir):
        run = load_run(name, run_dir)
        run["format"] = "crag_system"
        return run
    import yaml

    rows = [mode_b_row(r) for r in read_jsonl(run_dir / "queries.jsonl")]
    measured = [r for r in rows if not r.get("warmup")]
    ok = {r["query_id"]: r for r in measured if not r.get("error")}
    calls: dict[str, list] = {}
    for c in read_jsonl(run_dir / "calls.jsonl"):
        q = str(c.get("question_id"))
        if q in ok and c.get("request_index") == ok[q].get("request_index"):
            calls.setdefault(q, []).append(c)
    cfg_p = run_dir / "config.yaml"
    cfg = (yaml.safe_load(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else None) or {}
    return {
        "name": name,
        "dir": run_dir,
        "cfg": cfg,
        "rows": ok,
        "n_measured": len(measured),
        "errors": [r["query_id"] for r in measured if r.get("error")],
        "calls_by_query": calls,
        "format": "mode_b",
    }


def read_flip_ids(path: Path) -> dict[str, str | None]:
    """id -> 지정된 방향 (harm/gain, 없으면 None).

    txt / csv (query_id|question_id[,direction]) / json.
    """
    text = path.read_text(encoding="utf-8-sig")
    out: dict[str, str | None] = {}
    if path.suffix == ".json":
        data = json.loads(text)
        items = data.get("ids", data) if isinstance(data, dict) else data
        for x in items:
            if isinstance(x, dict):
                q = x.get("query_id") or x.get("question_id")
                out[str(q)] = (x.get("direction") or x.get("flip") or "").lower() or None
            else:
                out[str(x)] = None
        return out
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines and "," in lines[0]:
        head = [h.strip().lower() for h in lines[0].split(",")]
        qc = next(head.index(k) for k in ("query_id", "question_id", "id") if k in head)
        dc = next((head.index(k) for k in ("direction", "flip", "type") if k in head), None)
        for ln in lines[1:]:
            cells = [c.strip() for c in ln.split(",")]
            d = cells[dc].lower() if dc is not None and dc < len(cells) else None
            out[cells[qc]] = d or None
        return out
    return {ln: None for ln in lines}


def thresholds(run: dict, a) -> tuple[float, float, str]:
    if a.upper is not None and a.lower is not None:
        return a.upper, a.lower, "cli"
    if run["format"] == "mode_b":
        for r in run["rows"].values():
            t = r.get("_thresholds")
            if t:
                return t["upper"], t["lower"], "mode_b row"
    crag = run["cfg"].get("crag") or {}
    return crag.get("upper_threshold", 0.7), crag.get("lower_threshold", 0.3), "crag_system cfg"


# ---------------------------------------------------------------- classification
def keep_threshold(action: str | None, upper: float, lower: float) -> float | None:
    """crag_rule 기준 FAISS 문서 유지 기준. incorrect 는 전부 폐기 (무한대)."""
    return {"correct": upper, "ambiguous": lower, "incorrect": float("inf")}.get(action)


def classify(direction: str, case: dict, docs: list[dict], upper: float, lower: float) -> dict:
    sig = set(filter(None, case["signals"].split(";")))
    action = case["action"]
    contrib: list[str] = []
    faiss = [d for d in docs if d["source"] == "faiss"]
    gold_docs = [d for d in faiss if d.get("gold_string")]
    removed_gold = [d for d in gold_docs if not d.get("kept_in_full")]
    g_v, g_f = case["str_gold_in_vanilla_ctx"], case["str_gold_in_full_ctx"]
    web_used = case["n_web_docs"] > 0 or action in ("ambiguous", "incorrect")
    if case["metric_disagree"]:
        contrib.append("evaluation")
    if "D3_weak_gold" in sig or ("A3_partial_overlap" in sig and direction == "harm"):
        contrib.append("evaluation")

    def done(primary: str, detail: str) -> dict:
        rest = [c for c in dict.fromkeys(contrib) if c != primary]
        return {
            "primary_cause": primary,
            "primary_cause_ko": CAUSES[primary],
            "cause_detail": detail,
            "contributing": ";".join(rest),
        }

    if case["metric_disagree"]:
        return done(
            "evaluation", "acc (정규화) 와 공식 match (대소문자 구분) 의 뒤집힘 여부가 다름"
        )
    if "D1_retrieval_differs" in sig or "D2_text_missing" in sig:
        return done("hold", "검색 결과 불일치 또는 문서 원문 없음")

    if direction == "harm":
        if "C1_web_empty" in sig:
            contrib.append("external")
        elif "C2_web_no_gold" in sig or "C4_rewrite_low_overlap" in sig:
            contrib.append("external")
        if "A1_gold_in_ctx_answer_wrong" in sig:
            if "C6_context_grew" in sig:
                contrib.append("evidence")
            return done("prompt", "최종 context 에 정답 문자열이 있으나 오답 (생성·prompt 영향)")
        if "E1_gold_ctx_removed" in sig and removed_gold:
            contrib.append("evidence")
            thr = keep_threshold(action, upper, lower)
            graded = [d["grade_score"] for d in removed_gold if d["grade_score"] is not None]
            if len(graded) < len(removed_gold):
                return done("evidence", "정답 문서가 평가되지 않고 제거됨 (budget / lazy)")
            best = max(graded)
            if thr is not None and best < thr:
                where = {
                    "incorrect": f"정답 문서 점수 {best:.3f} < lower {lower} -> Incorrect 로 FAISS 전부 폐기",  # noqa: E501
                    "ambiguous": f"정답 문서 점수 {best:.3f} < lower {lower} -> Ambiguous 필터에서 제거",  # noqa: E501
                    "correct": f"정답 문서 점수 {best:.3f} < upper {upper} -> Correct 필터에서 제거",  # noqa: E501
                }[action]
                return done("grader", where)
            return done("evidence", f"정답 문서 점수 {best:.3f} 가 유지 기준 이상인데 정책상 제거")
        if g_v is False and web_used:
            return done(
                "external", "Vanilla 근거 문자열 없이 정답 -> CRAG 가 web 문서로 context 교체"
            )
        if g_v is False:
            return done("prompt", "근거 문자열 없이 정답이던 문항에서 context 만 달라짐")
        return done("other", "신호 조합이 규칙에 해당하지 않음")

    # gain
    if g_v is False and "C3_web_has_gold" in sig:
        return done("external", "Vanilla context 에 없던 정답 문자열을 web 문서가 추가")
    if g_v and g_f:
        dropped_nongold = [
            d for d in faiss if d.get("gold_string") is False and not d.get("kept_in_full")
        ]
        if dropped_nongold and not removed_gold:
            return done(
                "grader", "정답 문서 유지 + 정답 없는 문서 제거 (방해 문서 제거, 정상 판정)"
            )
        return done("prompt", "정답 문서가 양쪽 context 에 있고 형태·순서만 다름")
    if not g_v and not g_f:
        contrib.append("evaluation")
        return done("evaluation", "정답 문자열이 어느 context 에도 없음 (모델 지식·우연 일치 가능)")
    if g_v and not g_f:
        return done("evaluation", "정답 문서가 제거됐는데 정답 (모델 지식·우연 일치 가능)")
    return done("other", "신호 조합이 규칙에 해당하지 않음")


# ---------------------------------------------------------------- report
def case_card(c: dict, docs: list[dict]) -> list[str]:
    L = [
        f"#### {c['direction'].upper()} · {c['action']} · `{c['query_id']}` — "
        f"**{c['primary_cause_ko']}**",
        "",
        f"- 질문: {md_cell(c['question'])}",
        f"- 정답: {md_cell(c['gold'])}",
        f"- 기준 답: {md_cell(c['vanilla_answer'])} (acc {c['vanilla_acc']}, "
        f"공식 match {c['baseline_official_match']})",
        f"- 비교 답: {md_cell(c['full_answer'])} (acc {c['full_acc']}, "
        f"공식 match {c['treatment_official_match']})",
        f"- 분류 근거: {c['cause_detail']}"
        + (f" | 함께 성립: {c['contributing']}" if c["contributing"] else ""),
        f"- 정답 문자열: 기준 context {c['str_gold_in_vanilla_ctx']}, 비교 context "
        f"{c['str_gold_in_full_ctx']}, 출처 {c['str_gold_source_full']}",
        f"- web 문서 {c['n_web_docs']}개, rewrite: {md_cell(c['rewritten_query'])} "
        f"(질문 내용어 보존율 {c['str_rewrite_recall']})",
        f"- 신호: `{c['signals']}`",
        "",
        "| 출처 | 순위 | doc | grader 점수 | 구간 | 유지 | 기준 ctx | 정답 문자열 | 본문 앞부분 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in docs:
        L.append(
            "| "
            + " | ".join(
                str(x)
                for x in (
                    d["source"],
                    d.get("rank", ""),
                    d["doc_id"],
                    "" if d.get("grade_score") is None else f"{d['grade_score']:.3f}",
                    d.get("grade_band", ""),
                    d.get("kept_in_full"),
                    d.get("in_vanilla_ctx", ""),
                    d.get("gold_string"),
                    md_cell((d.get("text_head") or "")[:120]),
                )
            )
            + " |"
        )
    return [*L, ""]


def build_report(meta: dict, cases: list[dict], docs_by_q: dict, summary: list[dict]) -> str:
    harm_inc = [c for c in cases if c["direction"] == "harm" and c["action"] == "incorrect"]
    harm_rest = [c for c in cases if c["direction"] == "harm" and c["action"] != "incorrect"]
    gain = [c for c in cases if c["direction"] == "gain"]
    by = Counter((c["direction"], c["primary_cause_ko"]) for c in cases)
    L = [
        "# Flip 원인 분석 (잠정 분류)",
        "",
        f"- 기준: `{meta['baseline_dir']}` ({meta['baseline_format']})",
        f"- 비교: `{meta['treatment_dir']}` ({meta['treatment_format']})",
        f"- 공통 문항 {meta['n_common']} | HARM {meta['n_harm']} / GAIN {meta['n_gain']} "
        f"(acc 기준) | 분석 대상 {len(cases)}"
        + (f" (Flip 목록 {meta['flip_ids']['n']}개)" if meta.get("flip_ids") else ""),
        f"- threshold upper={meta['upper']}, lower={meta['lower']} ({meta['threshold_source']})",
        "",
        "> 분류는 로그 사실과 정답 문자열 지표로 만든 **잠정 분류**이다. 문자열이 있어도 근거가 아닐 수 "  # noqa: E501
        "있고 없어도 근거일 수 있다. `flip_cases.csv` 의 `manual_cause` 에 수동 확인을 기록한 뒤 확정한다.",  # noqa: E501
        "",
        "## 요약: 방향 x 원인",
        "",
        md_table(
            [{"direction": d, "cause": k, "n": n} for (d, k), n in sorted(by.items())],
        ),
        "## 방향 x 판정 x 원인",
        "",
        md_table(summary),
    ]
    if (meta.get("flip_ids") or {}).get("not_a_flip"):
        L += [
            "## Flip 목록에 있으나 두 run 에서 뒤집히지 않은 문항",
            "",
            md_table(meta["flip_ids"]["not_a_flip"]),
        ]
    L += ["## 1. Incorrect 판정 HARM (우선 분석)", ""]
    for c in harm_inc:
        L += case_card(c, docs_by_q[c["query_id"]])
    if not harm_inc:
        L += ["(없음)", ""]
    L += ["## 2. 나머지 HARM", ""]
    for c in harm_rest:
        L += case_card(c, docs_by_q[c["query_id"]])
    if not harm_rest:
        L += ["(없음)", ""]
    L += ["## 3. GAIN", ""]
    for c in gain:
        L += case_card(c, docs_by_q[c["query_id"]])
    if not gain:
        L += ["(없음)", ""]
    L += [
        "## 수동 확인 체크리스트 (Incorrect HARM)",
        "",
        "1. 기준 context 의 정답 문자열 문서가 실제 근거인가 (동명이인·숫자 우연 일치 아님)?",
        "2. 그 문서의 grader 점수가 낮은 이유: 문서가 질문 대상과 다른 표기·별칭인가, grader 가 관련 문서를 "  # noqa: E501
        "못 알아본 것인가 (공식 T5 점수와 비교하려면 MODE B T5 run 의 같은 문항)?",
        "3. web 문서가 정답을 담지 못한 이유: rewrite 가 질문 대상을 잃었는가, 검색 결과가 다른 인물·사물인가?",  # noqa: E501
        "4. 기준 답이 근거 없이 정답이면 (정답 문자열 없음) 모델 지식으로 맞힌 것이므로 CRAG 의 손실은 "  # noqa: E501
        "context 교체 (외부 지식 변화) 로 본다.",
        "",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", required=True, help="기준 run (예: Vanilla)")
    ap.add_argument("--treatment", required=True, help="비교 run (예: Full CRAG)")
    ap.add_argument("--flip-ids", default=None)
    ap.add_argument("--docs-path", default=None)
    ap.add_argument("--web-cache", action="append", default=[])
    ap.add_argument("--upper", type=float, default=None)
    ap.add_argument("--lower", type=float, default=None)
    ap.add_argument("--name", default="custom")
    ap.add_argument(
        "--out-root", default=str(ROOT / "results" / "official_crag" / "analysis" / "flips")
    )
    a = ap.parse_args(argv)

    b_dir, t_dir = resolve(a.baseline), resolve(a.treatment)
    base, treat = load_any("baseline", b_dir), load_any("treatment", t_dir)
    upper, lower, tsrc = thresholds(treat, a)
    order = [q for q in base["rows"] if q in treat["rows"]]

    def acc(r):
        return int(bool(r.get("acc")))

    flips = {}
    for q in order:
        x, y = acc(base["rows"][q]), acc(treat["rows"][q])
        if x != y:
            flips[q] = "harm" if x else "gain"
    flip_meta = None
    targets = list(flips)
    if a.flip_ids:
        wanted = read_flip_ids(Path(a.flip_ids))
        not_flip = []
        targets = []
        for q, d in wanted.items():
            got = flips.get(q)
            if q not in base["rows"] or q not in treat["rows"]:
                not_flip.append({"query_id": q, "listed": d, "observed": "missing_in_run"})
            elif got is None:
                not_flip.append({"query_id": q, "listed": d, "observed": "same_acc"})
            elif d and d != got:
                not_flip.append({"query_id": q, "listed": d, "observed": got})
                targets.append(q)
            else:
                targets.append(q)
        flip_meta = {"file": a.flip_ids, "n": len(wanted), "not_a_flip": not_flip}

    docs_path = resolve(
        a.docs_path
        or (treat["cfg"].get("retriever") or {}).get("docs_path")
        or (base["cfg"].get("retriever") or {}).get("docs_path")
    )
    if docs_path is None and treat["format"] == "mode_b":
        fp = (treat["cfg"].get("inputs") or {}).get("frozen_path")
        docs_path = None if not fp else resolve(fp)
    need: set[str] = set()
    for q in targets:
        need |= {str(i) for i in base["rows"][q].get("contexts") or []}
        need |= set(full_view(treat["rows"][q])["ret_ids"])
    texts = load_chunk_texts(docs_path, need) if docs_path and docs_path.suffix == ".jsonl" else {}
    if treat["format"] == "mode_b" and len(texts) < len(need):
        texts.update(frozen_texts(treat, need - set(texts)))
    web_paths = run_web_sources(treat) + [resolve(p) for p in a.web_cache]
    if treat["format"] == "mode_b":
        ws = treat["cfg"].get("web_search") or {}
        web_paths += [resolve(p) for p in ws.get("replay_paths") or []]
        web_paths += [
            resolve(p) / "web_cache.jsonl" for p in ws.get("replay_from_runs") or [] if resolve(p)
        ]
    web_paths = list(dict.fromkeys(p for p in web_paths if p and p.exists()))
    web_by_query, web_by_id = load_web(web_paths)
    ctx = {
        "upper": upper,
        "lower": lower,
        "texts": texts,
        "web_by_query": web_by_query,
        "web_by_id": web_by_id,
        "grade_mappable": False,  # grade 출력 원문은 쓰지 않음 (점수는 로그 값)
    }

    cases, docs_by_q = [], {}
    for q in targets:
        v, f = base["rows"][q], treat["rows"][q]
        c, d = analyze_query(
            q, v, f, base["calls_by_query"].get(q, []), treat["calls_by_query"].get(q, []), ctx
        )
        golds = f.get("gold_answers") or v.get("gold_answers") or []
        om_v = official_match(v.get("answer") or "", golds)
        om_f = official_match(f.get("answer") or "", golds)
        direction = flips.get(q) or ("harm" if c["vanilla_acc"] else "gain")
        c.update(
            direction=direction,
            baseline_official_match=om_v,
            treatment_official_match=om_f,
            metric_disagree=(om_v != om_f) != (c["vanilla_acc"] != c["full_acc"]),
            treatment_policy=f.get("_policy_id"),
        )
        c.update(classify(direction, c, d, upper, lower))
        c["manual_cause"] = ""
        c["manual_note"] = ""
        c.pop("provisional_bucket", None)
        c.pop("manual_label", None)
        cases.append(c)
        docs_by_q[q] = d
    rank = {"harm": 0, "gain": 1}
    act_rank = {"incorrect": 0, "ambiguous": 1, "correct": 2}
    cases.sort(key=lambda c: (rank[c["direction"]], act_rank.get(c["action"], 3), c["query_id"]))

    summ = Counter((c["direction"], str(c["action"]), c["primary_cause_ko"]) for c in cases)
    summary = [
        {"direction": d, "action": act, "cause": k, "n": n}
        for (d, act, k), n in sorted(summ.items())
    ]
    meta = {
        "created": datetime.now().isoformat(),
        "note": "읽기 전용. 문자열 지표 기반 잠정 분류 (확정 아님)",
        "baseline_dir": str(b_dir),
        "treatment_dir": str(t_dir),
        "baseline_format": base["format"],
        "treatment_format": treat["format"],
        "n_common": len(order),
        "n_harm": sum(v == "harm" for v in flips.values()),
        "n_gain": sum(v == "gain" for v in flips.values()),
        "upper": upper,
        "lower": lower,
        "threshold_source": tsrc,
        "flip_ids": flip_meta,
        "availability": {
            "docs_path": str(docs_path) if docs_path else None,
            "chunk_ids_needed": len(need),
            "chunk_ids_found": len(texts),
            "web_sources": [str(p) for p in web_paths if p and p.exists()],
        },
        "causes": CAUSES,
    }
    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    for d in (b_dir, t_dir):
        if out.resolve().is_relative_to(d.resolve()):
            raise ValueError(f"출력 폴더가 run 폴더 안에 있음: {out}")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "flip_cases.csv", cases)
    write_csv(out / "flip_docs.csv", [x for c in cases for x in docs_by_q[c["query_id"]]])
    write_csv(out / "cause_summary.csv", summary)
    with open(out / "analysis_meta.json", "x", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    with open(out / "report.md", "x", encoding="utf-8") as fh:
        fh.write(build_report(meta, cases, docs_by_q, summary))
    print(f"common={len(order)} harm={meta['n_harm']} gain={meta['n_gain']} analyzed={len(cases)}")
    print(f"saved -> {out}")
    return out


def frozen_texts(run: dict, ids: set[str]) -> dict[str, str]:
    """MODE B: 고정 입력 파일 (frozen jsonl) 의 문서 원문."""
    fp = resolve((run["cfg"].get("inputs") or {}).get("frozen_path"))
    out: dict[str, str] = {}
    if not fp or not fp.exists():
        return out
    for r in read_jsonl(fp):
        for d in r.get("docs") or []:
            i = str(d.get("id"))
            if i in ids:
                out[i] = d.get("text")
    return out


if __name__ == "__main__":
    main()
