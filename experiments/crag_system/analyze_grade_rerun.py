"""Grader 원본 출력 재실행 (record_texts=true) 분석: 파싱 실패, 재현성, 분기 변화 기여. 읽기 전용.

    uv run python experiments/crag_system/analyze_grade_rerun.py \
        --rerun r1=results/crag_system/pre/PRE-DIAG-GRADER-AWQ-R1/<ts> \
        [--rerun r2=results/crag_system/pre/PRE-DIAG-GRADER-AWQ-R2/<ts>] \
        --preset a1-smoke \
        [--selection results/crag_system/analysis/grader_diff/<ts>/srvinc_loccor_selected.csv] \
        [--local-profile results/integrated/profile_20261102_235932/profile.jsonl]

기존 run 폴더는 읽기만 한다. 결과는 새 폴더
results/crag_system/analysis/grade_rerun/<name>/<timestamp>/ 에만 쓴다 (이미 있으면 실패).

질문 (사용자 요청 번호)
  3. 기존 Full 의 0점 출력을 출력 토큰 수 (2 / 3 / 그 외) 로 나누고,
     재실행에서 같은 문서의 실제 문자열 -> token_strings.csv.
     재실행 출력이 기존 출력과 같다고 볼 근거 (점수·토큰 수 일치) 를 같이 표시.
  4. parse_score 가 숫자 없는 응답을 0 으로 처리한 사례 -> parse_summary.csv, output_class 열
  5. 기존 Full 점수 vs 재실행 점수 (문서·질문 단위), 재실행끼리 (r1 vs r2) -> repro.csv
  6. 파싱 정책별 action 재계산 (LLM 호출 없음) -> action_transitions.csv
     + 출력 토큰 수 -> 출력 유형 대응을 기존 Full 200 문항에 적용한 [추정]
       -> extrapolation.csv

근거 표시: [로그] 기록된 값, [재실행] 재실행 원문에서 본 사실, [추정] 토큰 수 대응 등으로 만든 값.
재실행 원문을 기존 run 의 원문으로 간주하는 것은 '같은 문서에서 점수와 출력 토큰 수가 모두 같다' 는
조건에서만 하고, 그 비율을 함께 보고한다. 정확도 (acc) 는 generation 을 다시 해야 바뀌므로
분기 변화는 action 수준까지만 계산한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_regression import PRESETS, ROOT, full_view, resolve  # noqa: E402
from analyze_runs import NUM_ONLY, load_run, md_table, read_jsonl, write_csv  # noqa: E402
from diag_crag_logs import grade_calls_for_docs  # noqa: E402

from rag.grading.base import decide_action, parse_score  # noqa: E402  (읽기만 함)

FIRST_NUM = re.compile(r"\d*\.?\d+")
WORD = re.compile(r"[a-z]+")
POLICIES = {
    "current": "parse_score 그대로 (숫자 없음 -> 0.0)",
    "missing": "숫자 없는 출력은 평가하지 않은 것으로 (None, 판정에서 제외)",
    "yes_no": "숫자 없는 출력 중 yes 로 시작 -> 1.0, no 로 시작 -> 0.0, 그 외 -> None",
}


# ---------------------------------------------------------------- 출력 분류
def output_class(text: str | None) -> str | None:
    """재실행 grade 출력 원문 -> 유형. None 이면 원문 없음."""
    if text is None:
        return None
    t = text.strip()
    if NUM_ONLY.match(t):
        v = float(FIRST_NUM.search(t).group())
        return "numeric_zero" if v == 0 else "numeric"
    if not FIRST_NUM.search(t):
        w = WORD.findall(t.lower())
        first = w[0] if w else ""
        if first == "yes":
            return "no_digit_yes"
        if first == "no":
            return "no_digit_no"
        return "no_digit_other" if t else "empty"
    return "number_in_text"


def policy_score(policy: str, text: str | None, recorded: float | None) -> float | None:
    if text is None or policy == "current":
        return recorded
    cls = output_class(text)
    if not cls or not (cls.startswith("no_digit") or cls == "empty"):
        return parse_score(text)
    if policy == "missing":
        return None
    if cls == "no_digit_yes":
        return 1.0
    if cls == "no_digit_no":
        return 0.0
    return None


def decide(scores: list[float | None], upper: float, lower: float) -> str:
    graded = [s for s in scores if s is not None]
    if not graded:
        return "incorrect_all_missing"  # decide_action 은 평가 문서가 없으면 incorrect
    return decide_action(scores, upper, lower)


def tok_band(n) -> str:
    if n is None:
        return "?"
    return str(n) if n in (2, 3, 4) else ">4"


# ---------------------------------------------------------------- 수집
def read_selection(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {r["query_id"]: r.get("stratum") or "-" for r in csv.DictReader(f)}


def read_local(path: Path | None, pipeline: str = "crag_full") -> dict[str, list]:
    if not path or not path.exists():
        return {}
    return {
        r["query_id"]: r.get("grader_scores") or []
        for r in read_jsonl(path)
        if r.get("pipeline") == pipeline and r.get("repeat", 0) == 0
    }


def mappable(run: dict) -> bool:
    g = run["cfg"].get("grading") or {}
    return g.get("strategy", "full") == "full" and not g.get("parallel")


def doc_rows(name: str, rerun: dict, full: dict, strata: dict, local: dict) -> list[dict]:
    out = []
    mr, mf = mappable(rerun), mappable(full)
    for q, r in rerun["rows"].items():
        f = full["rows"].get(q)
        rv = full_view(r)
        rc = grade_calls_for_docs(rerun["calls_by_query"].get(q, []), rv["scores"], mr)
        fv = full_view(f) if f else None
        fc = (
            grade_calls_for_docs(full["calls_by_query"].get(q, []), fv["scores"], mf) if f else None
        )
        loc = local.get(q) or []
        for i, doc in enumerate(rv["ret_ids"]):
            c = rc[i] or {}
            text = c.get("output_text")
            rec = rv["scores"][i]
            fs = fv["scores"][i] if fv and i < len(fv["scores"]) else None
            fcall = (fc[i] if fc else None) or {}
            out.append(
                {
                    "rerun": name,
                    "query_id": q,
                    "stratum": strata.get(q, "control" if strata else "-"),
                    "rank": i,
                    "doc_id": doc,
                    "doc_same_as_full": bool(
                        fv and i < len(fv["ret_ids"]) and fv["ret_ids"][i] == doc
                    ),
                    "full_score": fs,
                    "full_out_tokens": fcall.get("out_tokens"),
                    "full_action": f.get("corrective_action") if f else None,
                    "rerun_score": rec,
                    "rerun_out_tokens": c.get("out_tokens"),
                    "rerun_finish": c.get("finish_reason"),
                    "rerun_action": r.get("corrective_action"),
                    "output_text": text,
                    "output_class": output_class(text),
                    "reparsed": None if text is None else parse_score(text),
                    "reparse_matches_recorded": None if text is None else parse_score(text) == rec,
                    "score_same_as_full": None if fs is None or rec is None else fs == rec,
                    "tokens_same_as_full": (
                        None
                        if fcall.get("out_tokens") is None or c.get("out_tokens") is None
                        else fcall["out_tokens"] == c["out_tokens"]
                    ),
                    "local_score": loc[i] if i < len(loc) else None,
                }
            )
    return out


# ---------------------------------------------------------------- 표
def token_strings(docs: list[dict]) -> list[dict]:
    """기존 Full 0점 문서: 기존 출력 토큰 수 x 재실행 문자열."""
    groups: dict[tuple, Counter] = defaultdict(Counter)
    same: dict[tuple, int] = Counter()
    n: dict[tuple, int] = Counter()
    for d in docs:
        if d["full_score"] != 0 or d["output_text"] is None:
            continue
        key = (d["rerun"], tok_band(d["full_out_tokens"]))
        groups[key][d["output_text"].strip()[:80]] += 1
        n[key] += 1
        same[key] += bool(d["score_same_as_full"] and d["tokens_same_as_full"])
    rows = []
    for key in sorted(groups):
        for text, k in groups[key].most_common():
            rows.append(
                {
                    "rerun": key[0],
                    "full_out_tokens": key[1],
                    "n_docs_in_band": n[key],
                    "same_score_and_tokens_in_band": same[key],
                    "rerun_text": text,
                    "count": k,
                    "share": k / n[key],
                    "class": output_class(text),
                }
            )
    return rows


def parse_summary(docs: list[dict]) -> list[dict]:
    rows = []
    by = defaultdict(list)
    for d in docs:
        if d["output_text"] is None:
            continue
        by[(d["rerun"], "all")].append(d)
        by[(d["rerun"], f"stratum:{d['stratum']}")].append(d)
        by[
            (
                d["rerun"],
                f"full_tokens:{tok_band(d['full_out_tokens'])}|full_score0:{d['full_score'] == 0}",
            )
        ].append(d)
        if d["local_score"] is not None:
            lb = (
                "local>=0.7"
                if d["local_score"] >= 0.7
                else ("local0" if d["local_score"] == 0 else "local_mid")
            )
            by[(d["rerun"], f"{lb}")].append(d)
    for (rn, grp), ds in sorted(by.items()):
        cls = Counter(d["output_class"] for d in ds)
        zero = [d for d in ds if d["rerun_score"] == 0]
        zero_fail = sum(
            1
            for d in zero
            if (d["output_class"] or "").startswith("no_digit") or d["output_class"] == "empty"
        )
        rows.append(
            {
                "rerun": rn,
                "group": grp,
                "n_docs": len(ds),
                **{
                    f"n_{k}": cls.get(k, 0)
                    for k in (
                        "numeric",
                        "numeric_zero",
                        "no_digit_yes",
                        "no_digit_no",
                        "no_digit_other",
                        "empty",
                        "number_in_text",
                    )
                },
                "n_score0": len(zero),
                "n_score0_parse_failure": zero_fail,
                "share_score0_parse_failure": zero_fail / len(zero) if zero else None,
                "share_docs_parse_failure": sum(
                    (d["output_class"] or "").startswith("no_digit") or d["output_class"] == "empty"
                    for d in ds
                )
                / len(ds),
                "n_reparse_mismatch": sum(d["reparse_matches_recorded"] is False for d in ds),
                "n_finish_length": sum(d["rerun_finish"] == "length" for d in ds),
            }
        )
    return rows


def repro(docs: list[dict], reruns: dict, full: dict) -> list[dict]:
    rows = []
    for rn, run in reruns.items():
        ds = [d for d in docs if d["rerun"] == rn]
        qs = [q for q in run["rows"] if q in full["rows"]]
        rows.append(
            {
                "pair": f"{rn} vs full",
                "n_queries": len(qs),
                "retrieval_same": sum(
                    full_view(run["rows"][q])["ret_ids"] == full_view(full["rows"][q])["ret_ids"]
                    for q in qs
                ),
                "n_docs": len(ds),
                "score_same": sum(bool(d["score_same_as_full"]) for d in ds),
                "out_tokens_same": sum(bool(d["tokens_same_as_full"]) for d in ds),
                "action_same": sum(
                    run["rows"][q].get("corrective_action")
                    == full["rows"][q].get("corrective_action")
                    for q in qs
                ),
                "answer_same": sum(
                    (run["rows"][q].get("answer") or "").strip()
                    == (full["rows"][q].get("answer") or "").strip()
                    for q in qs
                ),
                "acc_same": sum(
                    run["rows"][q].get("acc") == full["rows"][q].get("acc") for q in qs
                ),
            }
        )
    names = list(reruns)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            da = {(d["query_id"], d["rank"]): d for d in docs if d["rerun"] == a}
            db = {(d["query_id"], d["rank"]): d for d in docs if d["rerun"] == b}
            keys = sorted(set(da) & set(db))
            qs = [q for q in reruns[a]["rows"] if q in reruns[b]["rows"]]
            rows.append(
                {
                    "pair": f"{a} vs {b}",
                    "n_queries": len(qs),
                    "n_docs": len(keys),
                    "score_same": sum(da[k]["rerun_score"] == db[k]["rerun_score"] for k in keys),
                    "out_tokens_same": sum(
                        da[k]["rerun_out_tokens"] == db[k]["rerun_out_tokens"] for k in keys
                    ),
                    "text_same": sum(da[k]["output_text"] == db[k]["output_text"] for k in keys),
                    "action_same": sum(
                        reruns[a]["rows"][q].get("corrective_action")
                        == reruns[b]["rows"][q].get("corrective_action")
                        for q in qs
                    ),
                    "answer_same": sum(
                        (reruns[a]["rows"][q].get("answer") or "").strip()
                        == (reruns[b]["rows"][q].get("answer") or "").strip()
                        for q in qs
                    ),
                }
            )
    return rows


def action_transitions(
    docs: list[dict], upper: float, lower: float
) -> tuple[list[dict], list[dict]]:
    by_q: dict[tuple, list[dict]] = defaultdict(list)
    for d in docs:
        by_q[(d["rerun"], d["query_id"])].append(d)
    per_q, counts = [], Counter()
    for (rn, q), ds in sorted(by_q.items()):
        ds = sorted(ds, key=lambda d: d["rank"])
        row = {
            "rerun": rn,
            "query_id": q,
            "stratum": ds[0]["stratum"],
            "full_action": ds[0]["full_action"],
            "rerun_action": ds[0]["rerun_action"],
        }
        for p in POLICIES:
            row[f"action_{p}"] = decide(
                [policy_score(p, d["output_text"], d["rerun_score"]) for d in ds], upper, lower
            )
        row["n_parse_failure"] = sum(
            (d["output_class"] or "").startswith("no_digit") or d["output_class"] == "empty"
            for d in ds
        )
        row["changed_by_parse_only"] = row["action_current"] != row["action_yes_no"]
        per_q.append(row)
        for p in ("missing", "yes_no"):
            counts[(rn, p, row["action_current"], row[f"action_{p}"])] += 1
    summary = [
        {"rerun": rn, "policy": p, "from_current": a, "to": b, "n_queries": n}
        for (rn, p, a, b), n in sorted(counts.items())
    ]
    return per_q, summary


def extrapolate(
    docs: list[dict], full: dict, vanilla: dict | None, upper, lower, min_share: float
) -> tuple[list[dict], dict]:
    """[추정] 재실행에서 '기존 0점 & 기존 출력 토큰 수 k' 문서의 출력 유형 다수값이
    min_share 이상이면, 기존 Full 200 문항의 같은 (0점, k 토큰) 문서에
    그 유형의 yes_no 점수를 적용해 action 을 다시 계산."""
    mapping, info = {}, {}
    band_cls: dict[str, Counter] = defaultdict(Counter)
    for d in docs:
        if (
            d["full_score"] == 0
            and d["output_class"]
            and d["score_same_as_full"]
            and d["tokens_same_as_full"]
        ):
            band_cls[tok_band(d["full_out_tokens"])][d["output_class"]] += 1
    value_of = {"no_digit_yes": 1.0, "no_digit_no": 0.0, "numeric_zero": 0.0}
    for band, cnt in band_cls.items():
        cls, k = cnt.most_common(1)[0]
        share = k / sum(cnt.values())
        applied = share >= min_share and cls in value_of
        info[band] = {
            "n": sum(cnt.values()),
            "majority": cls,
            "share": share,
            "applied": applied,
            "value": value_of.get(cls) if applied else None,  # None = 기존 점수 유지
        }
        if applied:
            mapping[band] = value_of[cls]
    rows, trans = [], Counter()
    mf = mappable(full)
    for q, f in full["rows"].items():
        fv = full_view(f)
        fc = grade_calls_for_docs(full["calls_by_query"].get(q, []), fv["scores"], mf)
        new = []
        for s, c in zip(fv["scores"], fc, strict=True):
            band = tok_band((c or {}).get("out_tokens"))
            new.append(
                mapping[band] if s == 0 and band in mapping and mapping[band] is not None else s
            )
        a0, a1 = f.get("corrective_action"), decide(new, upper, lower)
        v = (vanilla or {}).get("rows", {}).get(q) if vanilla else None
        regress = bool(v and v.get("acc") == 1 and f.get("acc") == 0)
        trans[(a0, a1, regress)] += 1
        rows.append(
            {"query_id": q, "full_action": a0, "estimated_action": a1, "regression": regress}
        )
    summary = [
        {"full_action": a0, "estimated_action": a1, "server_regression": r, "n_queries": n}
        for (a0, a1, r), n in sorted(trans.items(), key=lambda x: tuple(map(str, x[0])))
    ]
    return summary, {"band_majority": info, "mapping": mapping, "per_query": rows}


# ---------------------------------------------------------------- main
def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--rerun", action="append", required=True, help="name=run_dir (record_texts=true)"
    )
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--full", help="기존 Full run (기본: preset)")
    ap.add_argument("--vanilla", help="기존 Vanilla run (기본: preset, 회귀 표시용)")
    ap.add_argument("--selection", help="층 표시 csv (query_id, stratum). 없는 질문은 control")
    ap.add_argument("--local-profile", help="로컬 integrated profile.jsonl (로컬 grader 점수 열)")
    ap.add_argument("--extrapolate-min-share", type=float, default=0.9)
    ap.add_argument("--name", default="awq")
    ap.add_argument(
        "--out-root", default=str(ROOT / "results" / "crag_system" / "analysis" / "grade_rerun")
    )
    a = ap.parse_args(argv)
    preset = PRESETS.get(a.preset, {}) if a.preset else {}
    f_dir = resolve(a.full or preset.get("full"))
    v_dir = resolve(a.vanilla or preset.get("vanilla"))
    if not f_dir:
        ap.error("--preset 또는 --full 이 필요")
    full = load_run("full", f_dir)
    vanilla = load_run("vanilla", v_dir) if v_dir and v_dir.exists() else None
    reruns = {}
    for spec in a.rerun:
        nm, _, d = spec.partition("=")
        reruns[nm] = load_run(nm, resolve(d))
    crag = full["cfg"].get("crag") or {}
    upper, lower = crag.get("upper_threshold", 0.7), crag.get("lower_threshold", 0.3)
    strata = read_selection(resolve(a.selection))
    local = read_local(resolve(a.local_profile))

    problems = []
    for nm, run in reruns.items():
        if not (run["cfg"].get("run") or {}).get("record_texts"):
            problems.append(f"{nm}: run.record_texts 가 true 가 아님 (원문 없음)")
        if not mappable(run):
            problems.append(f"{nm}: grading 이 순차 full 이 아님 -> 호출-문서 대응 불가")
        g0, g1 = run["cfg"].get("grading"), full["cfg"].get("grading")
        if g0 != g1:
            problems.append(f"{nm}: grading 설정이 기존 Full 과 다름 ({g0} vs {g1})")
        if (run["cfg"].get("llm") or {}).get("model") != (full["cfg"].get("llm") or {}).get(
            "model"
        ):
            problems.append(f"{nm}: llm.model 이 기존 Full 과 다름")
        if run["errors"]:
            problems.append(
                f"{nm}: 오류 행 {len(run['errors'])}개 (분석에서 제외): {run['errors'][:5]}"
            )

    docs = [d for nm, run in reruns.items() for d in doc_rows(nm, run, full, strata, local)]
    ts = token_strings(docs)
    ps = parse_summary(docs)
    rp = repro(docs, reruns, full)
    per_q, trans = action_transitions(docs, upper, lower)
    first = docs and [d for d in docs if d["rerun"] == next(iter(reruns))]
    ex_summary, ex_info = extrapolate(
        first or [], full, vanilla, upper, lower, a.extrapolate_min_share
    )

    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "grade_docs.csv", docs)
    write_csv(out / "token_strings.csv", ts)
    write_csv(out / "parse_summary.csv", ps)
    write_csv(out / "repro.csv", rp)
    write_csv(out / "action_by_query.csv", per_q)
    write_csv(out / "action_transitions.csv", trans)
    write_csv(out / "extrapolation.csv", ex_summary)
    write_csv(out / "extrapolation_per_query.csv", ex_info["per_query"])

    n_text = sum(d["output_text"] is not None for d in docs)
    lines = [
        f"# Grader 원문 재실행 분석 ({a.name})",
        "",
        f"- 기존 Full: `{f_dir}`" + (f" / Vanilla: `{v_dir}`" if vanilla else ""),
        *[
            f"- 재실행 {nm}: `{run['dir']}` (정상 {len(run['rows'])}행, 오류 {len(run['errors'])})"
            for nm, run in reruns.items()
        ],
        f"- 문서 {len(docs)}개 중 원문 있음 {n_text}개. 임계값 upper={upper}, lower={lower}",
        "",
        "## 전제 검사",
        *(
            [f"- **{p}**" for p in problems]
            or ["- 문제 없음 (record_texts, 순차 full grading, 같은 grading 설정·모델, 오류 없음)"]
        ),
        "",
        "## 5. 재현성 [로그]",
        md_table(rp),
        "## 3. 기존 0점 출력: 기존 출력 토큰 수 별 재실행 문자열 [재실행]",
        "같은 문서에서 점수·출력 토큰 수가 기존과 모두 같은 수가 "
        "`same_score_and_tokens_in_band` 이다. "
        "이 비율이 높을 때만 재실행 문자열을 기존 출력으로 볼 수 있다.",
        "",
        md_table(ts[:60]),
        "## 4. parse_score 가 숫자 없는 출력을 0 으로 처리한 사례 [재실행]",
        "no_digit_* / empty = 숫자 없음 -> parse_score 0.0 (파싱 실패). "
        "numeric_zero = 모델이 0 을 출력 (실제 0점).",
        "",
        md_table(
            [
                r
                for r in ps
                if r["group"] == "all"
                or r["group"].startswith("full_tokens")
                or r["group"].startswith("local")
            ]
        ),
        "층별 표는 parse_summary.csv.",
        "",
        "## 6. 파싱 정책별 action 변화 (LLM 호출 없음)",
        *[f"- `{k}`: {v}" for k, v in POLICIES.items()],
        "",
        md_table(trans),
        f"질문 단위: action_by_query.csv (parse 만으로 action 이 바뀐 질문 "
        f"{sum(r['changed_by_parse_only'] for r in per_q)}개 / {len(per_q)}).",
        "",
        "## 6-추정. 기존 Full 200 문항에 적용 [추정]",
        f"재실행 {next(iter(reruns))} 에서 (기존 0점, 기존 출력 토큰 수) 구간별 출력 유형 다수값 "
        f"(점수·토큰이 기존과 같은 문서만, 다수값 비율 >= {a.extrapolate_min_share} 일 때만 적용):",
        "",
        md_table(
            [{"full_out_tokens": k, **v} for k, v in sorted(ex_info["band_majority"].items())]
        ),
        md_table(ex_summary),
        "정확도는 generation 을 다시 해야 바뀌므로 "
        "여기서는 action 변화와 그 질문의 기존 회귀 여부만 센다.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "full": str(f_dir),
        "vanilla": str(v_dir) if vanilla else None,
        "reruns": {nm: str(run["dir"]) for nm, run in reruns.items()},
        "selection": a.selection,
        "local_profile": a.local_profile,
        "problems": problems,
        "thresholds": {"upper": upper, "lower": lower},
        "extrapolation": {k: v for k, v in ex_info.items() if k != "per_query"},
    }
    with open(out / "analysis_meta.json", "x", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2, default=str)
    print(f"saved -> {out}")
    for p in problems:
        print("WARNING:", p)
    return out


if __name__ == "__main__":
    main()
