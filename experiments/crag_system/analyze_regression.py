"""Vanilla 정답 -> CRAG 오답 (회귀) 사례의 원인 지표 분석. 읽기 전용.

기존 run 폴더의 파일만 읽는다. LLM / Tavily 호출, run 폴더 쓰기는 하지 않는다.
결과는 새 폴더 results/crag_system/analysis/regression/<name>/<timestamp>/ 에만 쓴다.

    uv run python experiments/crag_system/analyze_regression.py --preset a1-smoke
    # 또는
    uv run python experiments/crag_system/analyze_regression.py \
        --vanilla results/crag_system/smoke/SMOKE-A1-VANILLA/20261110_103018 \
        --full results/crag_system/smoke/SMOKE-A1-FULL-RETRY/20261110_111726

읽는 파일
  <run>/queries.jsonl, calls.jsonl, config.yaml, run_meta.json, (있으면) web_cache.jsonl
  retriever.docs_path (chunk jsonl, 필요한 id 만 읽음)
  web replay 원본 (config / run_meta 에 기록된 경로)

지표의 두 종류 (report.md 와 CSV 에 구분해서 표기)
  [로그] 로그에 기록된 사실: corrective action, grader 점수, 유지·제거된 문서 id, web 문서 수,
         rewrite 문장, 생성 호출의 finish_reason / 토큰 수.
  [문자열] 정답 문자열이 문서·답변에 들어 있는지 (정규화 후 단어 경계 일치) 같은 참고 지표.
         문자열이 있어도 근거가 아닐 수 있고 (동명이인, 숫자 우연 일치), 없어도 근거일 수 있다
         (별칭, 표기 차이). 확정 판정이 아니며, 사례별 수동 확인 (manual_label 열) 이 필요하다.
  provisional_bucket 은 위 지표로 만든 잠정 분류이며 오류 원인의 확정이 아니다.

출력
  report.md                 요약과 서버에서 확인할 항목
  regression_cases.csv      회귀 사례별 로그 사실 + 문자열 지표 + 신호 + 잠정 분류 + 수동 확인 열
  regression_docs.csv       회귀 사례 x 문서 (FAISS / web): 순위, 검색 점수, grader 점수, 유지 여부,
                            정답 문자열 여부, 본문 앞부분
  signal_summary.csv        신호별 빈도: 회귀 사례 vs Full 정답 사례 (신호가 회귀에만 많은지 대조)
  bucket_summary.csv        잠정 분류 x corrective action
  action_summary.csv        Full corrective action 별 정확도·회귀·개선 (전체 공통 query)
  gold_doc_grading.csv      전체 FAISS 문서: 정답 문자열 여부 x grader 점수 구간 x 유지 여부
  analysis_meta.json        입력 경로, 데이터 가용성, 임계값
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_runs import NUM_ONLY, load_run, md_table, read_jsonl, write_csv  # noqa: E402

from rag.eval.metrics import f1_score, normalize_answer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    "a1-smoke": {
        "vanilla": "results/crag_system/smoke/SMOKE-A1-VANILLA/20261110_103018",
        "full": "results/crag_system/smoke/SMOKE-A1-FULL-RETRY/20261110_111726",
    }
}
SNIPPET = 200
REWRITE_RECALL_MIN = 0.5  # 질문 내용어 중 rewrite 에 남은 비율이 이보다 낮으면 신호
CONTEXT_GROWTH = 1.5  # Full 최종 context 글자 수가 Vanilla 의 이 배수 초과면 신호
STOPWORDS = set(
    "what who whom whose which when where why how is are was were be been do does did of in on "
    "at to for by with from and or as it its this that these those".split()
)
REFUSAL = re.compile(
    r"\b(unknown|not (?:mentioned|provided|specified|stated|available|given|found|clear)|"
    r"no (?:information|mention|answer|relevant)|cannot|can ?not|can't|unable|"
    r"does(?: not|n't) (?:say|mention|provide|contain|specify|state)|"
    r"not enough|insufficient|i don't know|n/?a)\b",
    re.IGNORECASE,
)

# 신호: (그룹, 근거 종류, 설명). 근거 종류: 로그 / 문자열 / 출력원문 (record_texts 필요)
SIGNALS = {
    "G1_gold_doc_below_upper": (
        "grading",
        "문자열",
        "정답 문자열이 있는 FAISS 문서의 grader 점수 < upper",
    ),
    "G2_gold_doc_below_lower": (
        "grading",
        "문자열",
        "정답 문자열이 있는 FAISS 문서의 grader 점수 < lower",
    ),
    "G3_nongold_kept_gold_dropped": (
        "grading",
        "문자열",
        "정답 문자열 없는 문서는 유지, 정답 문자열 있는 문서는 제거",
    ),
    "G4_incorrect_action": (
        "grading",
        "로그",
        "corrective action = incorrect (FAISS 문서 전부 폐기)",
    ),
    "G5_grade_output_unparsable": (
        "grading",
        "출력원문",
        "grade 출력에 숫자가 없어 parse_score 가 0 으로 처리",
    ),
    "E1_gold_ctx_removed": (
        "evidence",
        "문자열",
        "Vanilla context 에 있던 정답 문자열이 Full 의 유지 FAISS 문서에 없음",
    ),
    "E2_vanilla_ctx_all_dropped": (
        "evidence",
        "로그",
        "Vanilla context 문서 id 가 Full 최종 context 에 하나도 없음",
    ),
    "E3_no_gold_string_anywhere": (
        "evidence",
        "문자열",
        "Vanilla / Full 어느 context 에도 정답 문자열 없음",
    ),
    "C1_web_empty": ("correction", "로그", "correction 했는데 web 문서 0개"),
    "C2_web_no_gold": ("correction", "문자열", "web 문서 본문에 정답 문자열 없음"),
    "C3_web_has_gold": ("correction", "문자열", "web 문서 본문에 정답 문자열 있음"),
    "C4_rewrite_low_overlap": (
        "correction",
        "문자열",
        f"질문 내용어 중 rewrite 에 남은 비율 < {REWRITE_RECALL_MIN}",
    ),
    "C5_web_text_missing": (
        "correction",
        "로그",
        "web 문서 원문을 찾지 못함 (web_cache / replay 원본 없음)",
    ),
    "C6_context_grew": (
        "correction",
        "로그",
        f"Full 최종 context 글자 수 > Vanilla x{CONTEXT_GROWTH}",
    ),
    "A1_gold_in_ctx_answer_wrong": (
        "generation",
        "문자열",
        "Full 최종 context 에 정답 문자열이 있는데 오답",
    ),
    "A2_refusal_like": (
        "generation",
        "문자열",
        "답변이 '모름/언급 없음' 형태 (정규식)",
    ),
    "A3_partial_overlap": (
        "generation",
        "문자열",
        "오답이지만 정답과 토큰 F1 > 0 (별칭·표기 차이 등 평가 방식 영향 가능)",
    ),
    "A4_gen_truncated": ("generation", "로그", "생성 호출 finish_reason = length"),
    "A5_empty_answer": ("generation", "로그", "답변이 빈 문자열"),
    "A6_gen_call_error": ("generation", "로그", "생성 호출 오류 기록"),
    "D1_retrieval_differs": (
        "data",
        "로그",
        "Vanilla / Full 최초 검색 문서 id 가 다름 (비교 조건 불일치)",
    ),
    "D2_text_missing": ("data", "로그", "context 문서 원문을 찾지 못해 문자열 지표 판정 불가"),
    "D3_weak_gold": (
        "data",
        "문자열",
        "정답이 짧거나 숫자뿐이라 문자열 일치가 우연일 가능성 큼",
    ),
}


# ---------------------------------------------------------------- string indicators
def gold_norms(gold: list[str]) -> list[str]:
    return [g for g in (normalize_answer(x) for x in gold or []) if g]


def has_gold(text: str | None, golds: list[str]) -> bool | None:
    """정규화 후 단어 경계 일치. text 가 없으면 None (판정 불가).

    accuracy 지표 (부분 문자열 포함) 보다 엄격하다: '1' 이 '2010' 에 일치하지 않게 한다.
    """
    if text is None:
        return None
    t = f" {normalize_answer(text)} "
    return any(f" {g} " in t for g in golds)


def any_gold(texts: list[str | None], golds: list[str]) -> bool | None:
    """문서 여러 개 중 하나라도 정답 문자열을 포함하는지. 원문이 하나도 없으면 None."""
    found = [has_gold(t, golds) for t in texts]
    if not found:
        return False
    if any(found):
        return True
    return None if any(f is None for f in found) else False


def weak_gold(golds: list[str]) -> bool:
    return bool(golds) and all(len(g) <= 2 or g.replace(" ", "").isdigit() for g in golds)


def content_tokens(text: str) -> list[str]:
    return [t for t in normalize_answer(text or "").split() if t not in STOPWORDS]


def rewrite_recall(question: str, rewritten: str | None) -> float | None:
    q = set(content_tokens(question))
    if not q or rewritten is None:
        return None
    return len(q & set(normalize_answer(rewritten).split())) / len(q)


def score_band(s, upper: float, lower: float) -> str:
    if s is None:
        return "not_graded"
    return f">={upper}" if s >= upper else (f"<{lower}" if s < lower else f"{lower}-{upper}")


# ---------------------------------------------------------------- loading texts
def run_web_sources(run: dict) -> list[Path]:
    """run 의 web 문서 원문 위치: web_cache.jsonl + replay 원본 (config / run_meta 기록)."""
    paths = [Path(run["dir"]) / "web_cache.jsonl"]
    ws = (run["cfg"].get("crag") or {}).get("web_search") or {}
    paths += [Path(p) for p in ws.get("replay_paths") or []]
    paths += [Path(r) / "web_cache.jsonl" for r in ws.get("replay_from_runs") or []]
    meta_p = Path(run["dir"]) / "run_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        srcs = ((meta.get("comparability") or {}).get("web_search") or {}).get("sources") or []
        paths += [Path(s["path"]) for s in srcs if s.get("path")]
    out = []
    for p in paths:
        p = p if p.is_absolute() else ROOT / p
        if p.exists() and p not in out:
            out.append(p)
    return out


def load_web(paths: list[Path]) -> tuple[dict, dict]:
    """(검색어 -> [문서], 문서 id -> 본문). 같은 검색어는 처음 기록을 사용 (replay 와 같은 규칙)."""
    by_query, by_id = {}, {}
    for p in paths:
        for row in read_jsonl(p):
            q = (row.get("query") or "").strip()
            docs = row.get("docs") or []
            by_query.setdefault(q, docs)
            for d in docs:
                by_id.setdefault(str(d["id"]), d.get("text"))
    return by_query, by_id


def load_chunk_texts(path: Path | None, ids: set[str]) -> dict[str, str]:
    """chunk jsonl 에서 필요한 id 의 본문만 읽는다."""
    out: dict[str, str] = {}
    if not path or not path.exists() or not ids:
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            i = str(r.get("id"))
            if i in ids:
                out[i] = r.get("text")
                if len(out) == len(ids):
                    break
    return out


def resolve(p: str | Path | None) -> Path | None:
    if not p:
        return None
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- per query
def full_view(row: dict) -> dict:
    """crag_graph / crag 두 pipeline 의 결과 행에서 같은 형태의 정보를 꺼낸다."""
    extra = row.get("extra") or {}
    g = extra.get("grading") or {}
    ret_ids = [
        str(i) for i in (extra.get("retrieval_doc_ids") or row.get("retrieval_doc_ids") or [])
    ]
    scores = list(g.get("scores_full") or row.get("grader_scores") or [])
    scores += [None] * (len(ret_ids) - len(scores))
    ctx = [str(i) for i in row.get("contexts") or []]
    web_ids = extra.get("web_doc_ids")
    if web_ids is None:  # 기존 crag pipeline: 최초 검색에 없던 context 문서 = web 문서
        web_ids = [i for i in ctx if i not in set(ret_ids)]
    return {
        "ret_ids": ret_ids,
        "ret_scores": list(extra.get("retrieval_scores") or []),
        "scores": scores[: len(ret_ids)],
        "ctx": ctx,
        "kept_faiss": [i for i in ctx if i in set(ret_ids)],
        "web_ids": [str(i) for i in web_ids],
        "rewritten": extra.get("rewritten_query"),
        "strategy": g.get("strategy") or row.get("grading"),
    }


def gen_call(calls: list[dict]) -> dict:
    cs = [c for c in calls if c.get("kind") == "generate"]
    return cs[-1] if cs else {}


def grade_outputs(calls: list[dict], scores: list, mappable: bool) -> list:
    """grade 호출 출력 원문을 평가된 문서 순서에 맞춘다. 순서를 보장할 수 없으면 None 들."""
    outs = [None] * len(scores)
    cs = sorted((c for c in calls if c.get("kind") == "grade"), key=lambda c: c.get("idx", 0))
    graded = [i for i, s in enumerate(scores) if s is not None]
    if mappable and cs and len(cs) == len(graded):
        for i, c in zip(graded, cs, strict=True):
            outs[i] = c.get("output_text")
    return outs


def analyze_query(q: str, v: dict, f: dict, v_calls, f_calls, ctx: dict) -> tuple[dict, list]:
    upper, lower, texts = ctx["upper"], ctx["lower"], ctx["texts"]
    golds = gold_norms(f.get("gold_answers") or v.get("gold_answers") or [])
    fv = full_view(f)
    action = f.get("corrective_action")
    v_ctx = [str(i) for i in v.get("contexts") or []]
    v_ret = [str(i) for i in v.get("retrieval_doc_ids") or []]

    # web 문서 본문: 이 query 의 rewrite 검색어로 기록된 결과 우선, 없으면 id 로
    web_docs_by_q = {
        str(d["id"]): d.get("text")
        for d in ctx["web_by_query"].get((fv["rewritten"] or "").strip(), [])
    }

    def web_text(i):
        return web_docs_by_q.get(i, ctx["web_by_id"].get(i))

    def text(i):
        return texts.get(i) if i in texts else web_text(i)

    g_out = grade_outputs(f_calls, fv["scores"], ctx["grade_mappable"])
    docs = []
    for r, i in enumerate(fv["ret_ids"]):
        s = fv["scores"][r]
        t = texts.get(i)
        docs.append(
            {
                "query_id": q,
                "source": "faiss",
                "rank": r + 1,
                "doc_id": i,
                "retrieval_score": fv["ret_scores"][r] if r < len(fv["ret_scores"]) else None,
                "grade_score": s,
                "grade_band": score_band(s, upper, lower),
                "grade_output": g_out[r],
                "in_vanilla_ctx": i in v_ctx,
                "kept_in_full": i in fv["kept_faiss"],
                "gold_string": has_gold(t, golds),
                "text_head": None if t is None else t[:SNIPPET],
            }
        )
    for i in v_ctx:
        if i not in fv["ret_ids"]:
            t = texts.get(i)
            docs.append(
                {
                    "query_id": q,
                    "source": "vanilla_only",
                    "doc_id": i,
                    "in_vanilla_ctx": True,
                    "kept_in_full": False,
                    "gold_string": has_gold(t, golds),
                    "text_head": None if t is None else t[:SNIPPET],
                }
            )
    for r, i in enumerate(fv["web_ids"]):
        t = web_text(i)
        docs.append(
            {
                "query_id": q,
                "source": "web",
                "rank": r + 1,
                "doc_id": i,
                "kept_in_full": i in fv["ctx"],
                "gold_string": has_gold(t, golds),
                "text_head": None if t is None else t[:SNIPPET],
            }
        )

    g_vctx = any_gold([texts.get(i) for i in v_ctx], golds)
    g_kept = any_gold([texts.get(i) for i in fv["kept_faiss"]], golds)
    g_web = any_gold([web_text(i) for i in fv["web_ids"]], golds) if fv["web_ids"] else False
    g_fctx = any_gold([text(i) for i in fv["ctx"]], golds)
    if g_kept is None or g_web is None:
        gold_src = "unknown"
    else:
        gold_src = {(True, True): "both", (True, False): "faiss", (False, True): "web"}.get(
            (g_kept, g_web), "none"
        )
    faiss_docs = [d for d in docs if d["source"] == "faiss"]
    gold_scores = [d["grade_score"] for d in faiss_docs if d["gold_string"]]
    gold_graded = [s for s in gold_scores if s is not None]
    v_chars = sum(len(texts[i]) for i in v_ctx if i in texts) if v_ctx else 0
    f_chars = sum(len(text(i) or "") for i in fv["ctx"])
    f_txt_ok = all(text(i) is not None for i in fv["ctx"])
    v_txt_ok = all(i in texts for i in v_ctx)
    fg, vg = gen_call(f_calls), gen_call(v_calls)
    answer = f.get("answer") or ""
    rr = rewrite_recall(f.get("question") or "", fv["rewritten"]) if action != "correct" else None
    corrected = action not in (None, "correct")

    sig = []
    if gold_graded and max(gold_graded) < upper:
        sig.append("G1_gold_doc_below_upper")
    if gold_graded and max(gold_graded) < lower:
        sig.append("G2_gold_doc_below_lower")
    if any(d["kept_in_full"] and d["gold_string"] is False for d in faiss_docs) and any(
        d["gold_string"] and not d["kept_in_full"] for d in faiss_docs
    ):
        sig.append("G3_nongold_kept_gold_dropped")
    if action == "incorrect":
        sig.append("G4_incorrect_action")
    if any(o is not None and not re.search(r"\d", o) for o in g_out):
        sig.append("G5_grade_output_unparsable")
    if g_vctx and g_kept is False:
        sig.append("E1_gold_ctx_removed")
    if v_ctx and not set(v_ctx) & set(fv["ctx"]):
        sig.append("E2_vanilla_ctx_all_dropped")
    if g_vctx is False and g_fctx is False:
        sig.append("E3_no_gold_string_anywhere")
    if corrected:
        if not fv["web_ids"]:
            sig.append("C1_web_empty")
        elif g_web is None:
            sig.append("C5_web_text_missing")
        elif g_web:
            sig.append("C3_web_has_gold")
        else:
            sig.append("C2_web_no_gold")
        if rr is not None and rr < REWRITE_RECALL_MIN:
            sig.append("C4_rewrite_low_overlap")
    if v_txt_ok and f_txt_ok and v_chars and f_chars > CONTEXT_GROWTH * v_chars:
        sig.append("C6_context_grew")
    f_acc = bool(f.get("acc"))
    if g_fctx and not f_acc:
        sig.append("A1_gold_in_ctx_answer_wrong")
    if not f_acc and REFUSAL.search(answer):
        sig.append("A2_refusal_like")
    f1 = f1_score(answer, f.get("gold_answers") or [])
    if not f_acc and f1 > 0:
        sig.append("A3_partial_overlap")
    if fg.get("finish_reason") == "length":
        sig.append("A4_gen_truncated")
    if not answer.strip():
        sig.append("A5_empty_answer")
    if fg.get("error"):
        sig.append("A6_gen_call_error")
    if v_ret and fv["ret_ids"] and v_ret != fv["ret_ids"]:
        sig.append("D1_retrieval_differs")
    if g_vctx is None or g_fctx is None:
        sig.append("D2_text_missing")
    if weak_gold(golds):
        sig.append("D3_weak_gold")

    case = {
        "query_id": q,
        "question": f.get("question") or v.get("question"),
        "gold": " | ".join((f.get("gold_answers") or [])[:5]),
        "vanilla_acc": int(bool(v.get("acc"))),
        "full_acc": int(f_acc),
        "vanilla_answer": (v.get("answer") or "")[:300],
        "full_answer": answer[:300],
        # [로그] 사실
        "action": action,
        "grading_strategy": fv["strategy"],
        "grade_scores": json.dumps(fv["scores"]),
        "max_grade_score": max([s for s in fv["scores"] if s is not None], default=None),
        "n_vanilla_ctx": len(v_ctx),
        "n_full_ctx": len(fv["ctx"]),
        "n_faiss_kept": len(fv["kept_faiss"]),
        "n_web_docs": len(fv["web_ids"]),
        "vanilla_ctx_kept_ratio": (len(set(v_ctx) & set(fv["ctx"])) / len(v_ctx))
        if v_ctx
        else None,
        "retrieval_same": None if not (v_ret and fv["ret_ids"]) else v_ret == fv["ret_ids"],
        "rewritten_query": (fv["rewritten"] or "")[:200] if corrected else None,
        "vanilla_ctx_chars": v_chars if v_txt_ok else None,
        "full_ctx_chars": f_chars if f_txt_ok else None,
        "vanilla_gen_prompt_tokens": vg.get("prompt_tokens"),
        "full_gen_prompt_tokens": fg.get("prompt_tokens"),
        "full_gen_out_tokens": fg.get("out_tokens"),
        "full_gen_finish": fg.get("finish_reason"),
        # [문자열] 참고 지표
        "str_gold_in_vanilla_ctx": g_vctx,
        "str_gold_in_full_ctx": g_fctx,
        "str_gold_source_full": gold_src,
        "str_gold_doc_scores": json.dumps(gold_scores),
        "str_rewrite_recall": None if rr is None else round(rr, 3),
        "str_full_answer_f1": round(f1, 3),
        "signals": ";".join(sig),
        "provisional_bucket": bucket(sig, action),
        "manual_label": "",
        "manual_note": "",
    }
    return case, docs


def bucket(sig: list[str], action: str | None) -> str:
    """잠정 분류 (문자열 지표 기반, 확정 아님).

    우선순위: 판정 보류 > 생성 > 근거 제거 > 근거 없음.
    """
    s = set(sig)
    if "D2_text_missing" in s:
        return "보류: 문서 원문 없음"
    if "A1_gold_in_ctx_answer_wrong" in s:
        return "생성 문제 의심 (최종 context 에 정답 문자열 있음)"
    if "E1_gold_ctx_removed" in s:
        where = {
            "correct": "Correct 필터 (upper 미만 문서 제외)",
            "ambiguous": "Ambiguous 필터 + web 에 정답 문자열 없음",
            "incorrect": "Incorrect 폐기 + web 에 정답 문자열 없음",
        }.get(action, str(action))
        return f"근거 제거 의심: {where}"
    if "E3_no_gold_string_anywhere" in s:
        return "문자열 근거 없음 (모델 지식·별칭·평가 방식 가능)"
    return "기타 (수동 확인)"


# ---------------------------------------------------------------- aggregate
def signal_summary(cases_all: list[dict]) -> list[dict]:
    reg = [c for c in cases_all if c["vanilla_acc"] and not c["full_acc"]]
    ok = [c for c in cases_all if c["full_acc"]]
    rows = []
    for name, (group, basis, desc) in SIGNALS.items():

        def n(cs, name=name):
            return sum(name in c["signals"].split(";") for c in cs)

        by_act = Counter(c["action"] for c in reg if name in c["signals"].split(";"))
        rows.append(
            {
                "signal": name,
                "group": group,
                "basis": basis,
                "description": desc,
                "n_regression": n(reg),
                "rate_regression": n(reg) / len(reg) if reg else None,
                "n_full_correct": n(ok),
                "rate_full_correct": n(ok) / len(ok) if ok else None,
                "regression_correct": by_act.get("correct", 0),
                "regression_ambiguous": by_act.get("ambiguous", 0),
                "regression_incorrect": by_act.get("incorrect", 0),
            }
        )
    return rows


def action_summary(cases_all: list[dict]) -> list[dict]:
    rows = []
    for act in sorted({c["action"] for c in cases_all}, key=str):
        cs = [c for c in cases_all if c["action"] == act]

        def rate(key, cs=cs):
            vals = [c[key] for c in cs if c[key] is not None]
            return sum(map(bool, vals)) / len(vals) if vals else None

        rows.append(
            {
                "action": act,
                "n": len(cs),
                "vanilla_acc": sum(c["vanilla_acc"] for c in cs) / len(cs),
                "full_acc": sum(c["full_acc"] for c in cs) / len(cs),
                "regressions": sum(c["vanilla_acc"] and not c["full_acc"] for c in cs),
                "improvements": sum(not c["vanilla_acc"] and c["full_acc"] for c in cs),
                "str_gold_in_vanilla_ctx_rate": rate("str_gold_in_vanilla_ctx"),
                "str_gold_in_full_ctx_rate": rate("str_gold_in_full_ctx"),
            }
        )
    return rows


def gold_doc_grading(docs_all: list[dict]) -> list[dict]:
    t = defaultdict(lambda: [0, 0])
    for d in docs_all:
        if d["source"] != "faiss":
            continue
        k = (str(d["gold_string"]), d["grade_band"])
        t[k][0] += 1
        t[k][1] += bool(d["kept_in_full"])
    return [
        {"gold_string": k[0], "grade_band": k[1], "n_docs": n, "n_kept": kept}
        for k, (n, kept) in sorted(t.items())
    ]


def bucket_summary(cases: list[dict]) -> list[dict]:
    t = Counter((c["provisional_bucket"], c["action"]) for c in cases)
    return [
        {"provisional_bucket": b, "action": a, "n": n} for (b, a), n in sorted(t.items(), key=str)
    ]


def grade_format(run: dict, qids: list[str]) -> dict | None:
    outs = [
        c.get("output_text")
        for q in qids
        for c in run["calls_by_query"].get(q, [])
        if c.get("kind") == "grade" and c.get("output_text") is not None
    ]
    if not outs:
        return None
    return {
        "n": len(outs),
        "number_only_rate": sum(bool(NUM_ONLY.match(o)) for o in outs) / len(outs),
        "no_digit_rate": sum(re.search(r"\d", o) is None for o in outs) / len(outs),
        "non_numeric_examples": [o[:100] for o in outs if not NUM_ONLY.match(o)][:8],
    }


# ---------------------------------------------------------------- report
def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def md_cell(v):
    """표 칸 안의 | 와 줄바꿈이 Markdown 표를 깨지 않도록."""
    if not isinstance(v, str):
        return v
    return " ".join(v.split()).replace("|", "\\|")


def build_report(m: dict, tables: dict, cases: list[dict], n_examples: int) -> str:
    sig = {r["signal"]: r for r in tables["signal_summary"]}
    av = m["availability"]
    L = [
        "# CRAG 회귀 원인 지표 분석 (Vanilla 정답 -> Full 오답)",
        "",
        f"- Vanilla: `{m['vanilla_dir']}`",
        f"- Full: `{m['full_dir']}`",
        f"- 공통 query {m['n_common']}개 (오류·warmup 제외). "
        f"upper={m['upper']}, lower={m['lower']}",
        f"- 정확도: Vanilla {pct(m['acc_vanilla'])}, Full {pct(m['acc_full'])} | "
        f"회귀 (V 정답 -> F 오답) {m['n_regressions']}건, 개선 (V 오답 -> F 정답) "
        f"{m['n_improvements']}건",
        "",
        "## 해석 원칙",
        "",
        "- **[로그]** 지표는 로그에 기록된 사실 (action, 점수, 유지·제거된 문서 id, web 문서 수, "
        "finish_reason).",
        "- **[문자열]** 지표는 정답 문자열이 문서·답변에 있는지 보는 **참고 지표**이다. "
        "문자열이 있어도 근거가 아닐 수 있고 (동명이인·우연 일치), 없어도 근거일 수 있다 "
        "(별칭·표기 차이). 이것만으로 오류 원인을 확정하지 않는다.",
        "- `provisional_bucket` 은 신호 조합으로 만든 **잠정 분류**이다. 확정은 "
        "`regression_cases.csv` 의 `manual_label` 수동 확인 후에 한다.",
        "- 신호는 회귀 사례와 Full 정답 사례에서 함께 센다. 두 집단에서 비율이 비슷한 신호는 "
        "회귀 원인으로 보기 어렵다.",
        "",
        "## 데이터 가용성",
        "",
        f"- chunk 원문 (`{av['docs_path']}`): 필요한 {av['chunk_ids_needed']}개 중 "
        f"{av['chunk_ids_found']}개 찾음",
        f"- web 원문 출처: {av['web_sources'] or '없음'}",
        f"- grade / generate 출력 원문 (`run.record_texts`): Full={av['record_texts_full']}, "
        f"Vanilla={av['record_texts_vanilla']}",
        f"- grade 출력과 문서의 순서 대응: {av['grade_mappable_reason']}",
        f"- 최초 검색 결과가 Vanilla 와 다른 query: {av['retrieval_differs']}건",
        "",
        "## Full corrective action 별 (전체 공통 query)",
        "",
        md_table(tables["action_summary"]),
        "## 회귀 사례 잠정 분류 (확정 아님)",
        "",
        md_table(tables["bucket_summary"]),
        "## 신호 빈도: 회귀 vs Full 정답",
        "",
        md_table(
            tables["signal_summary"],
            [
                "signal",
                "basis",
                "n_regression",
                "rate_regression",
                "n_full_correct",
                "rate_full_correct",
                "regression_correct",
                "regression_ambiguous",
                "regression_incorrect",
            ],
        ),
        "## 1. Grading 오판 지표",
        "",
        "정답 문자열이 있는 FAISS 문서를 grader 가 낮게 평가했는지 본다. 문자열 포함은 관련성과 "
        "같지 않으므로 아래 표는 '정답 문자열 문서의 점수 분포' 이며 오판 비율이 아니다. "
        "(전체 공통 query 의 FAISS 문서, gold_string=None 은 원문 없음)",
        "",
        md_table(tables["gold_doc_grading"]),
        f"- 회귀 중 정답 문자열 문서 점수 < upper: "
        f"{sig['G1_gold_doc_below_upper']['n_regression']}건, "
        f"< lower: {sig['G2_gold_doc_below_lower']['n_regression']}건",
        f"- 정답 문자열 없는 문서 유지 + 있는 문서 제거: "
        f"{sig['G3_nongold_kept_gold_dropped']['n_regression']}건",
        "- grade 출력 형식: "
        + str(m["grade_format"] or "출력 원문 미기록 (record_texts=false): 판정 불가"),
        "",
        "## 2. 정답 근거 제거 지표",
        "",
        f"- Vanilla context 의 정답 문자열이 Full 유지 문서에서 사라짐 (E1): "
        f"{sig['E1_gold_ctx_removed']['n_regression']}건",
        f"- Vanilla context 문서 id 가 하나도 유지되지 않음 (E2, 로그): "
        f"{sig['E2_vanilla_ctx_all_dropped']['n_regression']}건",
        f"- 양쪽 context 모두 정답 문자열 없음 (E3): "
        f"{sig['E3_no_gold_string_anywhere']['n_regression']}건 — Vanilla 가 모델 지식이나 "
        "별칭으로 맞혔을 수 있어 문자열 지표로 설명되지 않는 사례",
        "",
        "## 3. Correction 이후 근거 변화",
        "",
        md_table(tables["gold_source"]),
        f"- web 0개 (C1): {sig['C1_web_empty']['n_regression']}건, web 원문 없음 (C5): "
        f"{sig['C5_web_text_missing']['n_regression']}건",
        f"- rewrite 가 질문 내용어를 적게 보존 (C4, recall<{REWRITE_RECALL_MIN}): "
        f"{sig['C4_rewrite_low_overlap']['n_regression']}건",
        f"- context 글자 수 증가 (C6): {sig['C6_context_grew']['n_regression']}건",
        "",
        "## 4. Generation 지표",
        "",
        f"- 최종 context 에 정답 문자열 있는데 오답 (A1): "
        f"{sig['A1_gold_in_ctx_answer_wrong']['n_regression']}건",
        f"- 거부·모름 형태 답변 (A2): {sig['A2_refusal_like']['n_regression']}건",
        f"- 부분 일치 (A3, F1>0): {sig['A3_partial_overlap']['n_regression']}건 — 평가 방식 "
        "(부분 문자열 accuracy) 영향 가능, 수동 확인 필요",
        f"- 생성 잘림 (A4): {sig['A4_gen_truncated']['n_regression']}건, 빈 답변 (A5): "
        f"{sig['A5_empty_answer']['n_regression']}건, 호출 오류 (A6): "
        f"{sig['A6_gen_call_error']['n_regression']}건",
        f"- 생성 prompt 토큰 평균 (회귀 사례): Vanilla {m['gen_prompt_tokens']['vanilla']}, "
        f"Full {m['gen_prompt_tokens']['full']}",
        "",
        "## 사례 예시 (잠정 분류별, 수동 확인용)",
        "",
    ]
    by_b = defaultdict(list)
    for c in cases:
        by_b[c["provisional_bucket"]].append(c)
    for b, cs in sorted(by_b.items()):
        L.append(f"### {b} ({len(cs)}건)")
        L.append("")
        L.append(
            md_table(
                [{k: md_cell(v) for k, v in c.items()} for c in cs[:n_examples]],
                ["query_id", "action", "gold", "vanilla_answer", "full_answer", "signals"],
            )
        )
    L += [
        "## 서버 실행 후 확인할 항목",
        "",
        "이 보고서의 수치는 스크립트가 로그에서 계산한 값이다. 아래는 사람이 확인해야 한다.",
        "",
        "1. 상단 정확도·회귀 수가 기존 보고 수치 (Vanilla 78.5%, Full 55.5%, 회귀 55건) 와 "
        "같은지. 다르면 공통 query 집합 (오류·warmup 제외) 차이를 먼저 확인",
        "2. `retrieval_same` 이 False 인 query 가 있으면 두 run 의 검색 조건이 달랐던 것이므로 "
        "해당 사례는 원인 분석에서 분리",
        "3. 잠정 분류별로 `regression_cases.csv` 를 몇 건씩 열어 `regression_docs.csv` 의 문서 "
        "본문을 읽고 `manual_label` 에 확정 원인 기록 (문자열 일치가 실제 근거인지)",
        "4. G1/G2 사례: 정답 문자열 문서가 실제로 질문과 관련 있는데 낮게 평가됐는지 "
        "(grading 오판) "
        "아니면 관련 없는 문서에 우연히 문자열이 있었는지",
        "5. grade 출력 원문이 없으면 (record_texts=false) 점수 0.0 이 실제 0 인지 "
        "파싱 실패인지 구분 "
        "불가 -> 필요 시 record_texts=true 로 해당 query 만 재실행",
        "6. E3 사례: Vanilla 정답이 context 근거 없이 나온 것인지 (모델 지식), 별칭 때문에 문자열 "
        "지표가 놓친 것인지",
        "7. A3 사례: 답변이 사실상 정답인데 accuracy 지표가 놓친 것인지 (평가 방식 영향)",
    ]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--vanilla", help="Vanilla run 폴더")
    ap.add_argument("--full", help="Full CRAG run 폴더")
    ap.add_argument(
        "--docs-path", help="chunk jsonl (기본: Full run config 의 retriever.docs_path)"
    )
    ap.add_argument("--web-cache", action="append", default=[], help="추가 web 기록 jsonl")
    ap.add_argument("--upper", type=float, help="기본: Full run config")
    ap.add_argument("--lower", type=float, help="기본: Full run config")
    ap.add_argument("--examples", type=int, default=5, help="report 의 분류별 예시 수")
    ap.add_argument("--name", default=None)
    ap.add_argument(
        "--out-root", default=str(ROOT / "results" / "crag_system" / "analysis" / "regression")
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

    docs_path = resolve(a.docs_path or (full["cfg"].get("retriever") or {}).get("docs_path"))
    need = set()
    for q in order:
        need |= {str(i) for i in van["rows"][q].get("contexts") or []}
        need |= set(full_view(full["rows"][q])["ret_ids"])
    texts = load_chunk_texts(docs_path, need)
    web_paths = run_web_sources(full) + [resolve(p) for p in a.web_cache]
    web_by_query, web_by_id = load_web([p for p in web_paths if p and p.exists()])

    gcfg = full["cfg"].get("grading") or {}
    strategy = gcfg.get("strategy", "full")
    parallel = bool(gcfg.get("parallel"))
    mappable = strategy in ("full", "lazy", "lazy_hit", "lazy_rest") and not parallel
    has_text = grade_format(full, order) is not None
    if not has_text:
        map_reason = "출력 원문 미기록"
    elif not mappable:
        map_reason = f"대응 불가 (strategy={strategy}, parallel={parallel}): 집계만 사용"
    else:
        map_reason = "호출 순서 = 문서 순서 (순차 개별 평가)"

    ctx = {
        "upper": upper,
        "lower": lower,
        "texts": texts,
        "web_by_query": web_by_query,
        "web_by_id": web_by_id,
        "grade_mappable": mappable,
    }
    cases_all, docs_by_q = [], {}
    for q in order:
        c, d = analyze_query(
            q,
            van["rows"][q],
            full["rows"][q],
            van["calls_by_query"].get(q, []),
            full["calls_by_query"].get(q, []),
            ctx,
        )
        cases_all.append(c)
        docs_by_q[q] = d
    reg = [c for c in cases_all if c["vanilla_acc"] and not c["full_acc"]]
    reg_docs = [d for c in reg for d in docs_by_q[c["query_id"]]]
    corrected_reg = [c for c in reg if c["action"] not in (None, "correct")]
    gold_source = [
        {"action": a_, "str_gold_source_full": s, "n": n}
        for (a_, s), n in sorted(
            Counter((c["action"], c["str_gold_source_full"]) for c in corrected_reg).items(),
            key=str,
        )
    ]

    def mean_tok(key):
        vals = [c[key] for c in reg if c[key] is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    n = len(cases_all)
    meta = {
        "created": datetime.now().isoformat(),
        "note": "읽기 전용 분석. 문자열 지표는 참고용이며 확정 판정 아님",
        "vanilla_dir": str(v_dir),
        "full_dir": str(f_dir),
        "upper": upper,
        "lower": lower,
        "n_common": n,
        "missing": {
            "vanilla_only": sorted(set(van["rows"]) - set(full["rows"])),
            "full_only": sorted(set(full["rows"]) - set(van["rows"])),
            "errors_vanilla": van["errors"],
            "errors_full": full["errors"],
        },
        "acc_vanilla": sum(c["vanilla_acc"] for c in cases_all) / n if n else None,
        "acc_full": sum(c["full_acc"] for c in cases_all) / n if n else None,
        "n_regressions": len(reg),
        "n_improvements": sum(not c["vanilla_acc"] and c["full_acc"] for c in cases_all),
        "gen_prompt_tokens": {
            "vanilla": mean_tok("vanilla_gen_prompt_tokens"),
            "full": mean_tok("full_gen_prompt_tokens"),
        },
        "grade_format": grade_format(full, order),
        "availability": {
            "docs_path": str(docs_path) if docs_path else None,
            "docs_path_exists": bool(docs_path and docs_path.exists()),
            "chunk_ids_needed": len(need),
            "chunk_ids_found": len(texts),
            "web_sources": [str(p) for p in web_paths if p and p.exists()],
            "record_texts_full": bool((full["cfg"].get("run") or {}).get("record_texts")),
            "record_texts_vanilla": bool((van["cfg"].get("run") or {}).get("record_texts")),
            "grade_mappable_reason": map_reason,
            "retrieval_differs": sum(c["retrieval_same"] is False for c in cases_all),
        },
        "thresholds": {
            "rewrite_recall_min": REWRITE_RECALL_MIN,
            "context_growth": CONTEXT_GROWTH,
        },
        "signals": {
            k: {"group": g, "basis": b, "description": d} for k, (g, b, d) in SIGNALS.items()
        },
    }
    tables = {
        "regression_cases": reg,
        "regression_docs": reg_docs,
        "signal_summary": signal_summary(cases_all),
        "bucket_summary": bucket_summary(reg),
        "action_summary": action_summary(cases_all),
        "gold_doc_grading": gold_doc_grading([d for ds in docs_by_q.values() for d in ds]),
        "gold_source": gold_source,
    }

    name = a.name or a.preset or "custom"
    out = Path(a.out_root) / name / datetime.now().strftime("%Y%m%d_%H%M%S")
    for d in (v_dir, f_dir):  # 결과를 run 폴더 안에 쓰지 않는다
        if out.resolve().is_relative_to(d.resolve()):
            raise ValueError(f"출력 폴더가 run 폴더 안에 있음: {out}")
    out.mkdir(parents=True, exist_ok=False)
    for k, rows in tables.items():
        if k != "gold_source":
            write_csv(out / f"{k}.csv", rows)
    with open(out / "analysis_meta.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    with open(out / "report.md", "x", encoding="utf-8") as f:
        f.write(build_report(meta, tables, reg, a.examples))
    print(
        f"common={n} acc vanilla={pct(meta['acc_vanilla'])} full={pct(meta['acc_full'])} "
        f"regressions={len(reg)} improvements={meta['n_improvements']}"
    )
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
