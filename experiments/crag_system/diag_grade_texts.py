"""Grading 출력 원문 진단 (읽기 전용): record_texts=true 재실행 vs 기존 Full run.

    uv run python experiments/crag_system/diag_grade_texts.py \
        --rerun results/crag_system/pre/PRE-DIAG-GRADETEXT/<ts> --preset a1-smoke \
        --selection results/crag_system/workloads/diag_gradetext_15_selection.csv

문서 단위로 확인하는 것
  재실행: grade 출력 원문, 현재 parse_score 로 다시 파싱한 점수, 기록된 점수, 출력 토큰,
          finish_reason, 문서 ID, corrective action
  기존 Full: 같은 질문·같은 순위의 문서 ID 와 점수, action  (기존 run 은 출력 원문 없음)

해석 원칙
  - 재실행 출력에서 본 파싱 문제 (숫자 없음, 숫자 외 문장, 범위 밖 숫자) 는 재실행에 대한 사실이다.
    기존 Full run 에서도 같은 출력이 나왔다고 단정하지 않는다. 기존 run 에서 확인되는 것은
    '점수가 재실행과 같았는지' 뿐이다 (같은 점수라도 출력 원문이 같았는지는 알 수 없음).
  - grade 호출과 문서의 대응은 순차 평가 (strategy=full, parallel=false) 에서 호출 순서 = 문서 순서
    로 한다. 대응할 수 없으면 원문 열을 비우고 보고한다.

결과: results/crag_system/analysis/grade_text/<name>/<timestamp>/
  report.md, grade_docs.csv (문서 단위), grade_queries.csv (질문 단위), diag_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_regression import PRESETS, ROOT, full_view, md_cell, resolve  # noqa: E402
from analyze_runs import NUM_ONLY, load_run, md_table, write_csv  # noqa: E402
from diag_crag_logs import grade_calls_for_docs  # noqa: E402

from rag.grading.base import parse_score  # noqa: E402  (현재 파서를 읽기만 함)

FIRST_NUM = re.compile(r"\d*\.?\d+")
ISSUES = {
    "no_digit": "출력에 숫자가 없음 -> parse_score 가 0.0 으로 처리",
    "extra_text": "숫자 하나만이 아닌 출력 (설명·접두어 등)",
    "multi_number": "숫자가 2개 이상 (첫 숫자만 사용)",
    "out_of_range": "첫 숫자가 [0, 1] 밖 -> clip",
    "length": "finish_reason = length (출력 잘림)",
}


def output_issues(text: str | None, finish: str | None) -> list[str] | None:
    if text is None:
        return None
    out = []
    nums = FIRST_NUM.findall(text)
    if not nums:
        out.append("no_digit")
    if not NUM_ONLY.match(text):
        out.append("extra_text")
    if len(nums) > 1:
        out.append("multi_number")
    if nums and not 0.0 <= float(nums[0]) <= 1.0:
        out.append("out_of_range")
    if finish == "length":
        out.append("length")
    return out


def read_selection(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {r["query_id"]: r["stratum"] for r in csv.DictReader(f)}


def compare(rerun: dict, full: dict, strata: dict[str, str], mappable: bool):
    docs, queries = [], []
    for q, r in rerun["rows"].items():
        f = full["rows"].get(q)
        rv = full_view(r)
        fv = full_view(f) if f else None
        calls = grade_calls_for_docs(rerun["calls_by_query"].get(q, []), rv["scores"], mappable)
        n_diff = 0
        for k, (i, s) in enumerate(zip(rv["ret_ids"], rv["scores"], strict=True)):
            c = calls[k] or {}
            text = c.get("output_text")
            f_id = fv["ret_ids"][k] if fv and k < len(fv["ret_ids"]) else None
            f_s = fv["scores"][k] if fv and k < len(fv["scores"]) else None
            equal = None if fv is None else (f_id == i and f_s == s)
            n_diff += equal is False
            iss = output_issues(text, c.get("finish_reason"))
            docs.append(
                {
                    "query_id": q,
                    "stratum": strata.get(q),
                    "rank": k + 1,
                    "doc_id": i,
                    "full_doc_id": f_id,
                    "same_doc": None if fv is None else f_id == i,
                    "full_score": f_s,
                    "rerun_score": s,
                    "score_equal": equal,
                    "rerun_output": text,
                    "reparsed_score": None if text is None else parse_score(text),
                    "reparse_matches_recorded": None if text is None else parse_score(text) == s,
                    "out_tokens": c.get("out_tokens"),
                    "finish_reason": c.get("finish_reason"),
                    "issues": None if iss is None else ";".join(iss),
                    "full_action": f.get("corrective_action") if f else None,
                    "rerun_action": r.get("corrective_action"),
                }
            )
        queries.append(
            {
                "query_id": q,
                "stratum": strata.get(q),
                "in_full": f is not None,
                "full_action": f.get("corrective_action") if f else None,
                "rerun_action": r.get("corrective_action"),
                "action_same": None
                if f is None
                else f.get("corrective_action") == r.get("corrective_action"),
                "retrieval_same": None if fv is None else fv["ret_ids"] == rv["ret_ids"],
                "full_scores": json.dumps(fv["scores"]) if fv else None,
                "rerun_scores": json.dumps(rv["scores"]),
                "n_docs_score_diff": n_diff if fv else None,
                "full_acc": None if f is None else int(bool(f.get("acc"))),
                "rerun_acc": int(bool(r.get("acc"))),
                "full_answer": (f.get("answer") or "")[:200] if f else None,
                "rerun_answer": (r.get("answer") or "")[:200],
                "web_replay": json.dumps(r.get("web_replay")) if r.get("web_replay") else None,
            }
        )  # fmt: skip
    return docs, queries


def build_report(m: dict, docs: list[dict], queries: list[dict]) -> str:
    with_text = [d for d in docs if d["rerun_output"] is not None]
    by_stratum = []
    for s in sorted({d["stratum"] or "-" for d in docs}):
        ds = [d for d in with_text if (d["stratum"] or "-") == s]
        row = {"stratum": s, "n_docs_with_text": len(ds)}
        for k in ISSUES:
            row[k] = sum(k in (d["issues"] or "").split(";") for d in ds)
        row["score_0.0"] = sum(d["rerun_score"] == 0.0 for d in ds)
        row["score_equal_to_full"] = sum(d["score_equal"] is True for d in ds)
        by_stratum.append(row)
    outputs = Counter(d["rerun_output"] for d in with_text)
    odd = [d for d in with_text if d["issues"]]
    q_cols = ["query_id", "stratum", "full_action", "rerun_action", "action_same",
              "retrieval_same", "n_docs_score_diff", "full_acc", "rerun_acc"]  # fmt: skip
    L = [
        "# Grading 출력 원문 진단 (재실행 vs 기존 Full)",
        "",
        f"- 재실행: `{m['rerun_dir']}` (record_texts={m['record_texts_rerun']})",
        f"- 기존 Full: `{m['full_dir']}`",
        f"- 질문 {m['n_queries']}개, 문서 {m['n_docs']}개, "
        f"출력 원문 대응 {m['n_docs_with_text']}개 "
        f"({m['mapping']})",
        f"- 재실행 오류 행: {m['rerun_errors'] or '없음'}",
        "",
        "## 해석 원칙",
        "",
        "- 아래 파싱 문제는 **재실행 출력에 대한 사실**이다. 기존 Full run 은 출력 원문이 없으므로 "
        "같은 문제가 과거에도 있었다고 단정하지 않는다.",
        "- 기존 run 과 비교해 확인되는 것은 같은 문서에 같은 점수가 나왔는지 "
        "(`score_equal`) 뿐이다. "
        "점수가 같아도 출력 원문이 같았는지는 알 수 없다.",
        "",
        "## 기존 Full 과의 일치",
        "",
        f"- 문서 ID 가 같은 순위에서 일치: {m['same_doc']} / {m['n_docs_compared']}",
        f"- 점수 일치 (문서 ID 와 점수 모두): {m['score_equal']} / {m['n_docs_compared']}",
        f"- action 일치: {m['action_same']} / {m['n_queries_compared']}",
        f"- 정답 여부 일치: {m['acc_same']} / {m['n_queries_compared']}",
        "",
        md_table([{k: md_cell(q.get(k)) for k in q_cols} for q in queries], q_cols),
        "## 재실행 출력의 파싱 문제 (층별, 문서 수)",
        "",
        md_table(by_stratum),
        "정의: " + "; ".join(f"`{k}` {v}" for k, v in ISSUES.items()),
        "",
        "- 현재 parse_score 로 다시 파싱한 값과 기록된 점수가 다른 문서: "
        f"{m['reparse_mismatch']}개 "
        "(0 이 아니면 기록 경로 확인 필요)",
        "",
        "## 출력 원문 빈도 (상위 15)",
        "",
        md_table([{"output": md_cell(repr(o)), "n": n} for o, n in outputs.most_common(15)]),
        "## 파싱 문제가 있는 출력 예시 (최대 20)",
        "",
        md_table(
            [
                {
                    k: md_cell(str(d.get(k))[:200])
                    for k in (
                        "query_id",
                        "rank",
                        "doc_id",
                        "rerun_output",
                        "rerun_score",
                        "full_score",
                        "issues",
                    )
                }
                for d in odd[:20]
            ]
        ),  # fmt: skip
    ]
    return "\n".join(L) + "\n"


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rerun", required=True, help="record_texts=true 재실행 run 폴더")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--full", help="기존 Full run 폴더 (기본: preset)")
    ap.add_argument("--selection", help="select_diag_queries.py 의 _selection.csv (층 표시)")
    ap.add_argument("--name", default="gradetext")
    ap.add_argument(
        "--out-root", default=str(ROOT / "results" / "crag_system" / "analysis" / "grade_text")
    )
    a = ap.parse_args(argv)
    f_dir = resolve(a.full or (PRESETS.get(a.preset, {}) if a.preset else {}).get("full"))
    if not f_dir:
        ap.error("--full 또는 --preset 이 필요")
    r_dir = resolve(a.rerun)
    rerun, full = load_run("rerun", r_dir), load_run("full", f_dir)
    gcfg = rerun["cfg"].get("grading") or {}
    strategy, parallel = gcfg.get("strategy", "full"), bool(gcfg.get("parallel"))
    mappable = strategy in ("full", "lazy", "lazy_hit", "lazy_rest") and not parallel
    docs, queries = compare(rerun, full, read_selection(resolve(a.selection)), mappable)
    cmp_docs = [d for d in docs if d["score_equal"] is not None]
    cmp_q = [q for q in queries if q["in_full"]]
    meta = {
        "created": datetime.now().isoformat(),
        "rerun_dir": str(r_dir),
        "full_dir": str(f_dir),
        "record_texts_rerun": bool((rerun["cfg"].get("run") or {}).get("record_texts")),
        "mapping": f"strategy={strategy}, parallel={parallel}, "
        + ("호출 순서 = 문서 순서" if mappable else "대응 불가"),
        "rerun_errors": rerun["errors"],
        "n_queries": len(queries),
        "n_docs": len(docs),
        "n_docs_with_text": sum(d["rerun_output"] is not None for d in docs),
        "n_docs_compared": len(cmp_docs),
        "same_doc": sum(bool(d["same_doc"]) for d in cmp_docs),
        "score_equal": sum(bool(d["score_equal"]) for d in cmp_docs),
        "n_queries_compared": len(cmp_q),
        "action_same": sum(bool(q["action_same"]) for q in cmp_q),
        "acc_same": sum(q["full_acc"] == q["rerun_acc"] for q in cmp_q),
        "reparse_mismatch": sum(d["reparse_matches_recorded"] is False for d in docs),
        "issue_counts": {
            k: sum(k in (d["issues"] or "").split(";") for d in docs if d["issues"] is not None)
            for k in ISSUES
        },
        "note": "파싱 문제는 재실행 출력 기준. 기존 run 에서도 발생했다고 단정하지 않음",
    }
    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    for d in (r_dir, f_dir):
        if out.resolve().is_relative_to(d.resolve()):
            raise ValueError(f"출력 폴더가 run 폴더 안에 있음: {out}")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "grade_docs.csv", docs)
    write_csv(out / "grade_queries.csv", queries)
    with open(out / "diag_meta.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    with open(out / "report.md", "x", encoding="utf-8") as f:
        f.write(build_report(meta, docs, queries))
    print(
        f"queries={len(queries)} docs={len(docs)} with_text={meta['n_docs_with_text']} "
        f"score_equal={meta['score_equal']}/{meta['n_docs_compared']}"
    )
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
