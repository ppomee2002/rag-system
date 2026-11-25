"""C2: Correction 발생 질문의 정확도·비용 분석 + Grading 중단 영향 (읽기 전용).

공식 T5 Evaluator (official@official_t5 run) 의 판정으로 Correction 발생 질문을 정하고, 같은 질문·
같은 고정 검색 입력·같은 generator·같은 평가 기준에서 정책을 질문 단위로 짝지어 비교한다.

    uv run python experiments/official_crag/analyze_correction.py \
        --official <T5-OFF run> --vanilla <VAN run> \
        [--legacy <LL-RULE 또는 T5-RULE run>] [--keep-all <T5-KEEP run>] \
        [--extra NAME=<run>] [--question-ids <ID 파일>] --name <이름>

Correction 발생 범위 (논문 CRAG, Yan et al. 2024 §4 / 공식 CRAG_Inference.py):
  Correct    : 내부 문서 knowledge refinement 만. 외부 검색 없음          -> Correction 아님
  Incorrect  : 내부 문서 버리고 web search (+ refinement)                  -> Correction
  Ambiguous  : 내부 refinement + web search 결합                           -> Correction
  => 기본 Correction 집합 = Incorrect ∪ Ambiguous (외부 검색이 실행된 질문).
     Incorrect 만 (strict) 결과도 따로 낸다.

결과 (이미 있으면 실패): results/official_crag/analysis/<name>/
  summary.json, report.md
  per_question.csv            질문별 경로·정책별 정답 여부·Vanilla 대비 변화·correction 유효성
  path_breakdown.csv          T5 경로 (correct/ambiguous/incorrect) x 정책: 정확도·실행 비용
  early_stop_summary.csv      grading 중단 규칙별: 판정 변화·evidence 변화·절감 비용·품질 영향 범위
  early_stop_per_question.csv 질문 x 규칙 (C2 상태 복원용)
  verification_candidates.csv evidence 가 바뀌어 generation 재실행으로 확인해야 하는 질문

주의
  - C1 공통 문항 (완료 실험 200문항) 은 기존 공통 성공 문항 (선택된 집합).
    이 결과를 전체 PopQA 정확도로 표현하지 않는다.
  - Grading 중단 시 최종 답 변화는 generation 을 다시 돌려야 확정된다. 여기서는 evidence 변화와
    그로 인한 정답 변화 가능 범위 (상한) 만 낸다.
  - Vanilla 대비 차이는 (official 정책의 경우) refinement + correction 의 결합 효과다.
    keep_all@official_t5 run 이 있으면 keep_all - vanilla 가 "내부 문서 유지 + web 추가" 효과다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze  # noqa: E402
from freeze_inputs import read_ids  # noqa: E402

from rag.evidence.manifest import compare_manifests, comparison_view  # noqa: E402
from rag.evidence.policies import get_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PATHS = ("correct", "ambiguous", "incorrect")
CORRECTION_PATHS = ("ambiguous", "incorrect")
METRICS = ("acc", "official_match")
SUBSET_CAVEAT = (
    "이 질문 집합은 기존 공통 성공 문항 등 선택된 집합일 수 있다. 결과를 전체 PopQA 의 일반적인 "
    "정확도로 표현하지 말 것 (MODE B 는 공식 논문 수치의 재현도 아님)."
)
IGNORE_IN_COMPARE = ("question_ids_sha256", "n_questions")


# ---------------------------------------------------------------- 입력
def load(d: str | Path, role: str) -> dict:
    d = Path(d)
    run = analyze.load_run(d)
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) or {}
    man_path = d / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else None
    comp = (
        man["comparison"]
        if man
        else comparison_view(
            {**cfg, "experiment_mode": run["meta"].get("experiment_mode", "B_controlled")},
            run["meta"].get("inputs"),
            None,
        )
    )
    run.update(role=role, cfg=cfg, manifest=man, comparison=comp)
    if man:
        run["eval_sha"] = (man.get("evaluation") or {}).get("sha256")
    return run


def check_role(run: dict) -> list[str]:
    ok = list(run["ok"].values())
    probs = []
    if run["comparison"].get("experiment_mode") != "B_controlled":
        probs.append(
            f"{run['role']}: MODE B run 이 아님 ({run['comparison'].get('experiment_mode')})"
        )
    pol = {r.get("evidence_policy") for r in ok}
    if run["role"] == "official":
        if pol != {"official"} or {r.get("grader_type") for r in ok} != {"official_t5"}:
            probs.append(f"official: official@official_t5 run 이 아님 ({pol})")
        if not all(r.get("official_policy_semantics") for r in ok):
            probs.append("official: official_policy_semantics=false")
        if any((r.get("budget") or {}).get("name") != "full_grading" for r in ok):
            probs.append("official: full_grading 이 아님")
    want = {"vanilla": "vanilla", "legacy": "crag_rule", "keep_all": "keep_all"}.get(run["role"])
    if want and pol != {want}:
        probs.append(f"{run['role']}: evidence_policy={pol} (기대: {want})")
    return probs


def is_mock(run: dict) -> bool:
    ev = run["meta"].get("evaluator") or {}
    return (ev.get("backend") or {}).get("backend") == "mock"


# ---------------------------------------------------------------- 계산
def correct(run: dict, q: str, metric: str) -> bool:
    return bool(run["ok"][q]["metrics"][metric])


def paired(a: dict, b: dict, qs: list[str], metric: str) -> dict:
    """a = 기준 (vanilla), b = 비교. 질문 단위 짝."""
    n = len(qs)
    both = sum(correct(a, q, metric) and correct(b, q, metric) for q in qs)
    improved = sum(not correct(a, q, metric) and correct(b, q, metric) for q in qs)
    degraded = sum(correct(a, q, metric) and not correct(b, q, metric) for q in qs)
    neither = n - both - improved - degraded
    return {
        "n": n,
        "base_acc": (both + degraded) / n if n else None,
        "compare_acc": (both + improved) / n if n else None,
        "improved": improved,  # 기준 오답 -> 비교 정답
        "degraded": degraded,  # 기준 정답 -> 비교 오답
        "both_correct": both,
        "both_wrong": neither,
        "delta_acc": (improved - degraded) / n if n else None,
        "mcnemar_p": analyze.mcnemar_p(degraded, improved),
    }


def cost(rows: list[dict]) -> dict:
    def m(f):
        return analyze.mean([f(r) for r in rows])

    def st(s):
        return m(lambda r: (r.get("stage_timings") or {}).get(s))

    def calls(r):
        return sum(v.get("n", 0) for v in (r.get("llm_calls") or {}).values())

    return {
        "e2e_s": m(lambda r: r.get("e2e_latency")),
        "grading_s": st("grading"),
        "refinement_s": st("refinement"),
        "rewrite_s": st("rewrite"),
        "web_search_s": st("web_search"),
        "generation_s": st("generation"),
        "evaluator_compute_s": m(lambda r: r.get("evaluator_compute_seconds")),
        "gen_in_tokens": m(lambda r: r.get("generation_input_tokens")),
        "gen_out_tokens": m(lambda r: r.get("generation_output_tokens")),
        "llm_calls": m(calls),
        "web_search_rate": m(lambda r: float(bool((r.get("correction") or {}).get("performed")))),
        "evidence_units": m(lambda r: (r.get("evidence") or {}).get("n_units")),
    }


def outcome(base_ok: bool, cmp_ok: bool) -> str:
    if base_ok and cmp_ok:
        return "both_correct"
    if cmp_ok:
        return "improved"
    if base_ok:
        return "degraded"
    return "both_wrong"


C2_CATEGORY = {
    "improved": "correction_improved",  # Vanilla 오답 -> CRAG 정답
    "both_correct": "correction_no_change_correct",
    "both_wrong": "correction_no_change_wrong",
    "degraded": "correction_degraded",  # Vanilla 정답 -> CRAG 오답
}


# ---------------------------------------------------------------- grading 중단
def evidence_signature(policy, action: str, scores: dict[int, float], n: int, upper, lower):
    """판정 + 평가된 점수로 정해지는 evidence 구성 (내부 문서 집합, web 사용).

    refine (official) 은 refinement 를 모든 검색 문서에 대해 실행한다고 가정한다 (grading 중단이
    refinement 범위를 줄이지 않음). 이 가정이 깨지면 evidence 가 더 바뀔 수 있다.
    """
    mode = policy.internal_mode(action)
    if mode == "refine":
        internal: tuple = ("refine_all_docs",)
    elif mode == "all":
        internal = tuple(range(n))
    elif mode == "none":
        internal = ()
    else:
        cut = upper if mode == "filter_upper" else lower
        internal = tuple(r for r in sorted(scores) if scores[r] >= cut)
    return internal, policy.uses_web(action)


def stop_point(rule: str, snaps: list[dict], n_total: int) -> int:
    """규칙에 따라 평가를 멈출 문서 수 k (1..n_total)."""
    if rule == "first_pass":
        first = next(
            (s["first_pass_rank"] for s in snaps if s["first_pass_rank"] is not None), None
        )
        return n_total if first is None else first + 1
    return min(int(rule.split(":")[1]), n_total)


def _sum(xs):
    return None if any(x is None for x in xs) else sum(xs)


def early_stop(run: dict, qs: list[str], label: str) -> tuple[list[dict], list[dict], list[dict]]:
    per_q, cands = [], []
    rows = [run["ok"][q] for q in qs]
    n_max = max((len(r.get("grading_events") or []) for r in rows), default=0)
    rules = ["first_pass"] + [f"max_docs:{k}" for k in range(1, n_max)]
    for r in rows:
        events = sorted(r.get("grading_events") or [], key=lambda e: e["eval_order"])
        snaps = r.get("grading_snapshots") or []
        if not events or not snaps:
            continue
        n_total = snaps[-1]["n_total"]
        if len(events) < n_total:
            continue  # 원 run 이 이미 중단됨 -> 오프라인 재구성 불가
        policy = get_policy(r["evidence_policy"])
        th = r.get("thresholds") or {}
        up, lo = th.get("upper"), th.get("lower")
        final = r["correction_path"]
        all_scores = {e["rank"]: e["score"] for e in events}
        sig_final = evidence_signature(policy, final, all_scores, n_total, up, lo)
        for rule in rules:
            k = stop_point(rule, snaps, n_total)
            prov = snaps[k - 1]["provisional_action"]
            sc = {e["rank"]: e["score"] for e in events[:k]}
            changed_ev = evidence_signature(policy, prov, sc, n_total, up, lo) != sig_final
            rest = events[k:]
            rec = {
                "run": label,
                "question_id": r["question_id"],
                "rule": rule,
                "n_total": n_total,
                "stop_after": k,
                "stopped_early": k < n_total,
                "final_action": final,
                "provisional_action": prov,
                "action_final_flag": snaps[k - 1]["action_final"],
                "decision_changed": prov != final,
                "evidence_changed": changed_ev,
                "saved_docs": len(rest),
                "saved_grading_latency_ms": sum(e["latency"] for e in rest) * 1000.0,
                "saved_compute_ms": (
                    None
                    if _sum([e.get("compute_seconds") for e in rest]) is None
                    else _sum([e.get("compute_seconds") for e in rest]) * 1000.0
                ),
                "saved_input_tokens": _sum([e.get("input_tokens") for e in rest]),
                "saved_output_tokens": _sum([e.get("output_tokens") for e in rest]),
                "total_grading_latency_ms": sum(e["latency"] for e in events) * 1000.0,
                "full_run_correct": bool(r["metrics"]["acc"]),
                "answer_change": "unverified_requires_generation"
                if changed_ev
                else "none_expected_same_evidence",
                "assumes_full_refinement": policy.internal_mode(prov) == "refine"
                or policy.internal_mode(final) == "refine",
            }
            per_q.append(rec)
            if changed_ev:
                cands.append(
                    {
                        "run": label,
                        "question_id": r["question_id"],
                        "rule": rule,
                        "final_action": final,
                        "provisional_action": prov,
                        "full_run_correct": rec["full_run_correct"],
                        "how_to_verify": (
                            "official 정책 + 부분 grading 은 runner 가 거부 (공식 evidence 정의 밖): "  # noqa: E501
                            "판정 강제 재실행 기능 필요 (미구현)"
                            if r["evidence_policy"] == "official"
                            else f"--set budget.stop_rule={rule} --set budget.retention=policy "
                            "(같은 정책·grader) 로 실제 run 후 per_question 비교"
                        ),
                    }
                )
    summ = []
    for rule in rules:
        xs = [x for x in per_q if x["rule"] == rule]
        if not xs:
            continue
        lat_tot = sum(x["total_grading_latency_ms"] for x in xs)
        lat_saved = sum(x["saved_grading_latency_ms"] for x in xs)
        ev = [x for x in xs if x["evidence_changed"]]
        trans: dict[str, int] = {}
        for x in xs:
            if x["decision_changed"]:
                key = f"{x['final_action']}->{x['provisional_action']}"
                trans[key] = trans.get(key, 0) + 1
        summ.append(
            {
                "run": label,
                "rule": rule,
                "n_questions": len(xs),
                "n_stopped_early": sum(x["stopped_early"] for x in xs),
                "n_decision_changed": sum(x["decision_changed"] for x in xs),
                "decision_transitions": json.dumps(trans, ensure_ascii=False),
                "n_evidence_changed": len(ev),
                "n_evidence_same": len(xs) - len(ev),
                "saved_docs_total": sum(x["saved_docs"] for x in xs),
                "saved_grading_latency_ms_total": lat_saved,
                "saved_grading_latency_frac": lat_saved / lat_tot if lat_tot else None,
                "saved_input_tokens_total": _sum([x["saved_input_tokens"] for x in xs]),
                "saved_output_tokens_total": _sum([x["saved_output_tokens"] for x in xs]),
                # 같은 evidence -> 같은 prompt -> temperature 0 에서 같은 답 (가정, 실측 아님)
                "potential_loss_upper_bound": sum(x["full_run_correct"] for x in ev),
                "potential_gain_upper_bound": sum(not x["full_run_correct"] for x in ev),
                "quality_change": "unverified (generation 재실행 필요)" if ev else "none_expected",
            }
        )
    return summ, per_q, cands


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True, help="official@official_t5 run (Correction 기준)")
    ap.add_argument("--vanilla", required=True)
    ap.add_argument("--legacy", default=None, help="crag_rule run (Legacy CRAG)")
    ap.add_argument("--keep-all", default=None)
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=RUN_DIR")
    ap.add_argument("--question-ids", default=None, help="분석할 질문 ID 파일 (txt/csv/json)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--allow-mismatch", action="store_true")
    a = ap.parse_args(argv)

    off = load(a.official, "official")
    van = load(a.vanilla, "vanilla")
    others = []
    if a.legacy:
        others.append(load(a.legacy, "legacy"))
    if a.keep_all:
        others.append(load(a.keep_all, "keep_all"))
    for spec in a.extra:
        nm, _, d = spec.partition("=")
        others.append(load(d, nm))

    problems = check_role(off) + check_role(van)
    excluded: dict[str, str] = {}
    for o in others:
        p = check_role(o)
        if p:
            excluded[o["role"]] = "; ".join(p)

    # 비교 조건 (고정 입력, generator, prompt, 평가 기준)
    def diffs(x, y):
        out = [
            d
            for d in compare_manifests(x["comparison"], y["comparison"])
            if not d.split(":")[0].endswith(IGNORE_IN_COMPARE)
        ]
        if x.get("eval_sha") and y.get("eval_sha") and x["eval_sha"] != y["eval_sha"]:
            out.append("evaluation script sha256 다름")
        return out

    d = diffs(off, van)
    if d:
        problems.append("official vs vanilla 조건 다름: " + "; ".join(d))
    for o in others:
        if o["role"] in excluded:
            continue
        d = diffs(off, o)
        if d:
            excluded[o["role"]] = "조건 다름: " + "; ".join(d)
    if problems and not a.allow_mismatch:
        raise SystemExit("비교 불가: " + " | ".join(problems))
    if a.allow_mismatch:
        excluded = {k: v for k, v in excluded.items() if not v.startswith("조건 다름")}
    comps = [o for o in others if o["role"] not in excluded]

    # 질문 집합
    if a.question_ids:
        ids = read_ids(Path(a.question_ids))
        missing = [q for q in ids if q not in off["rows"] or q not in van["rows"]]
        missing += [q for o in comps for q in ids if q not in o["rows"]]
        if missing and not a.allow_mismatch:
            raise SystemExit(
                f"질문 ID 가 run 에 없음 ({len(set(missing))}개): {sorted(set(missing))[:10]}"
            )
    else:
        ids = sorted(off["rows"])
    runs = [off, van, *comps]
    dropped = {q: [r["role"] for r in runs if q not in r["ok"]] for q in ids}
    dropped = {q: v for q, v in dropped.items() if v}
    qs = [q for q in ids if q not in dropped]

    corr = [q for q in qs if off["ok"][q]["correction_path"] in CORRECTION_PATHS]
    strict = [q for q in qs if off["ok"][q]["correction_path"] == "incorrect"]
    by_path = {p: [q for q in qs if off["ok"][q]["correction_path"] == p] for p in PATHS}

    out = ROOT / "results" / "official_crag" / "analysis" / a.name
    out.mkdir(parents=True, exist_ok=False)

    summary: dict = {
        "runs": {r["role"]: r["dir"] for r in runs},
        "excluded_runs": excluded,
        "warnings": problems,
        "mock_evaluator": is_mock(off),
        "correction_definition": {
            "default": "official T5 판정 in {ambiguous, incorrect} (web search 실행)",
            "strict": "official T5 판정 == incorrect",
            "not_correction": "correct (internal refinement 만)",
        },
        "subset_caveat": SUBSET_CAVEAT,
        "frozen_caveat": ((off["meta"].get("inputs") or {}).get("manifest") or {}).get("caveat"),
        "n_requested": len(ids),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "n_total": len(qs),
        "n_correction": len(corr),
        "correction_rate": len(corr) / len(qs) if qs else None,
        "n_correction_strict": len(strict),
        "path_counts": {p: len(v) for p, v in by_path.items()},
        "correction_invalid": sum(
            1 for q in corr if not (off["ok"][q].get("correction") or {}).get("valid")
        ),
        "metrics": {},
    }
    for metric in METRICS:
        mres: dict = {}
        for set_name, sq in (
            ("correction", corr),
            ("correction_strict", strict),
            ("all", qs),
            *((f"path_{p}", v) for p, v in by_path.items()),
        ):
            mres[set_name] = {"official_vs_vanilla": paired(van, off, sq, metric)}
            for o in comps:
                mres[set_name][f"{o['role']}_vs_vanilla"] = paired(van, o, sq, metric)
                mres[set_name][f"{o['role']}_vs_official"] = paired(off, o, sq, metric)
        summary["metrics"][metric] = mres
    # C2 분류 (correction 집합, acc 기준)
    cats = {v: 0 for v in C2_CATEGORY.values()}
    cats_invalid = {v: 0 for v in C2_CATEGORY.values()}
    for q in corr:
        c = C2_CATEGORY[outcome(correct(van, q, "acc"), correct(off, q, "acc"))]
        cats[c] += 1
        if not (off["ok"][q].get("correction") or {}).get("valid"):
            cats_invalid[c] += 1
    summary["c2_categories_acc"] = cats
    summary["c2_categories_acc_web_invalid"] = cats_invalid
    summary["c2_attribution_note"] = (
        "official vs vanilla 차이는 refinement + correction 결합 효과. keep_all@official_t5 가 있으면 "  # noqa: E501
        "keep_all vs vanilla 가 '내부 문서 유지 + web 추가' 효과"
    )
    # keep_all / legacy 의 판정 경로가 official 과 같은 질문인지 (같은 grader 면 같아야 함)
    for o in comps:
        tab: dict[str, int] = {}
        for q in qs:
            k = f"{off['ok'][q]['correction_path']}->{o['ok'][q].get('correction_path')}"
            tab[k] = tab.get(k, 0) + 1
        summary[f"path_crosstab_official_vs_{o['role']}"] = tab

    # per question
    pq = []
    for q in qs:
        r = off["ok"][q]
        c = r.get("correction") or {}
        row = {
            "question_id": q,
            "official_path": r["correction_path"],
            "is_correction": r["correction_path"] in CORRECTION_PATHS,
            "web_performed": c.get("performed"),
            "web_valid": c.get("valid"),
            "web_status": c.get("status"),
            "n_web_docs": len(c.get("web_doc_ids") or []),
            "correction_query": c.get("query"),
            "first_pass_rank": ((r.get("grading_snapshots") or [{}])[-1]).get("first_pass_rank"),
        }
        for rr in runs:
            x = rr["ok"][q]
            row[f"{rr['role']}:acc"] = int(bool(x["metrics"]["acc"]))
            row[f"{rr['role']}:official_match"] = int(bool(x["metrics"]["official_match"]))
            row[f"{rr['role']}:path"] = x.get("correction_path")
            row[f"{rr['role']}:e2e_s"] = x.get("e2e_latency")
        for rr in [off, *comps]:
            row[f"{rr['role']}_vs_vanilla:acc"] = outcome(
                correct(van, q, "acc"), correct(rr, q, "acc")
            )
        row["c2_category"] = (
            C2_CATEGORY[row["official_vs_vanilla:acc"]] if row["is_correction"] else "no_correction"
        )
        pq.append(row)
    analyze.write_csv(out / "per_question.csv", pq)

    # path x policy
    pb = []
    for set_name, sq in (*((p, v) for p, v in by_path.items()), ("correction", corr), ("all", qs)):
        for rr in runs:
            rows = [rr["ok"][q] for q in sq]
            pb.append(
                {
                    "t5_path": set_name,
                    "policy": rr["role"],
                    "policy_id": rows[0].get("policy_id") if rows else None,
                    "n": len(sq),
                    "acc": analyze.mean([float(bool(x["metrics"]["acc"])) for x in rows]),
                    "official_match": analyze.mean(
                        [float(bool(x["metrics"]["official_match"])) for x in rows]
                    ),
                    **cost(rows),
                }
            )
    analyze.write_csv(out / "path_breakdown.csv", pb)

    # grading 중단
    es_sum, es_pq, cands = [], [], []
    for rr in [off, *comps]:
        if not any(x.get("grading_events") for x in rr["ok"].values()):
            continue
        s, p, c = early_stop(rr, qs, rr["role"])
        es_sum += s
        es_pq += p
        cands += c
    analyze.write_csv(out / "early_stop_summary.csv", es_sum)
    analyze.write_csv(out / "early_stop_per_question.csv", es_pq)
    analyze.write_csv(out / "verification_candidates.csv", cands)
    summary["early_stop"] = es_sum

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out / "report.md").write_text(report(a.name, summary, runs), encoding="utf-8")
    print(f"saved -> {out}")
    return out


def _f(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else (f"{v:.3f}" if isinstance(v, float) else str(v))


def report(name: str, s: dict, runs: list[dict]) -> str:
    L = [f"# Correction 분석 (C2): {name}", ""]
    if s["mock_evaluator"]:
        L += ["**MOCK evaluator run — 정확도 결과로 사용하지 말 것**", ""]
    L += [f"> {s['subset_caveat']}", ""]
    if s.get("frozen_caveat"):
        L += [f"> 고정 입력 manifest: {s['frozen_caveat']}", ""]
    if s["warnings"]:
        L += ["**경고 (--allow-mismatch)**: " + " | ".join(s["warnings"]), ""]
    if s["excluded_runs"]:
        L += ["제외된 run: " + "; ".join(f"{k} ({v})" for k, v in s["excluded_runs"].items()), ""]
    L += [
        "Correction 정의: 공식 T5 판정이 Ambiguous 또는 Incorrect (외부 검색 실행). "
        "Correct 는 internal refinement 만 하므로 Correction 아님. strict = Incorrect 만.",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 전체 질문 | {s['n_total']} (요청 {s['n_requested']}, 오류로 제외 {s['n_dropped']}) |",
        f"| Correction 발생 | {s['n_correction']} ({_f(s['correction_rate'], True)}) |",
        f"| 경로 C / A / I | {s['path_counts']['correct']} / {s['path_counts']['ambiguous']} / "
        f"{s['path_counts']['incorrect']} |",
        f"| web 결과 무효 (correction 중) | {s['correction_invalid']} |",
        "",
    ]
    for metric in METRICS:
        m = s["metrics"][metric]
        L += [
            f"## {metric}",
            "",
            "| 집합 | 비교 | n | 기준 acc | 비교 acc | 개선 | 악화 | 둘다 정답 | 둘다 오답 | McNemar p |",  # noqa: E501
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for set_name, comps in m.items():
            for cname, p in comps.items():
                L.append(
                    f"| {set_name} | {cname} | {p['n']} | {_f(p['base_acc'])} | "
                    f"{_f(p['compare_acc'])} | {p['improved']} | {p['degraded']} | "
                    f"{p['both_correct']} | {p['both_wrong']} | {_f(p['mcnemar_p'])} |"
                )
        L.append("")
    L += [
        "## C2 분류 (Correction 질문, acc, official vs vanilla)",
        "",
        "| 분류 | 질문 수 | 그중 web 무효 |",
        "|---|---|---|",
    ]
    for k, v in s["c2_categories_acc"].items():
        L.append(f"| {k} | {v} | {s['c2_categories_acc_web_invalid'][k]} |")
    L += ["", s["c2_attribution_note"], "", "## Grading 중단 (오프라인 재구성)", ""]
    L += [
        "| run | 규칙 | 조기중단 | 판정변화 | evidence변화 | 절감 latency | 손실 상한 | 이득 상한 |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in s["early_stop"]:
        L.append(
            f"| {e['run']} | {e['rule']} | {e['n_stopped_early']} | {e['n_decision_changed']} | "
            f"{e['n_evidence_changed']} | {_f(e['saved_grading_latency_frac'], True)} | "
            f"{e['potential_loss_upper_bound']} | {e['potential_gain_upper_bound']} |"
        )
    L += [
        "",
        "- 판정·evidence 변화와 절감 비용은 full run 의 문서별 기록으로 정확히 계산된다.",
        "- 정답 변화는 **미확정** (generation 재실행 필요). 손실/이득 상한 = evidence 가 바뀐 질문 중 "  # noqa: E501
        "full run 정답/오답 수. evidence 가 같으면 temperature 0 에서 답이 같다고 가정 (실측 아님).",  # noqa: E501
        "- official 정책은 refinement 를 모든 문서에 대해 실행한다고 가정 (grading 중단이 refinement "  # noqa: E501
        "범위를 줄이면 evidence 가 더 바뀐다).",
        "",
        "runs: " + ", ".join(f"{r['role']}=`{r['dir']}`" for r in runs),
    ]
    return "\n".join(L)


if __name__ == "__main__":
    main()
