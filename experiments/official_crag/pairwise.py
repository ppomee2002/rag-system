"""조건 간 Pairwise 비교 (읽기 전용): `accuracy` 와 `official_match` 를 **각각** 기준으로.

    uv run python experiments/official_crag/pairwise.py \
        --run VAN=<run> --run CRAG=<run> --run T5-OFF=<run> [--run VAN-OP=<run> ...] \
        --baseline VAN [--all-pairs] --name c1_pairwise

두 지표
  accuracy       rag.eval.metrics.accuracy: 소문자·구두점·관사 정규화 후 정답 문자열 포함
                 (기존 지표)
  official_match 공식 eval.py match: 정규화 없이 대소문자 구분 포함
  두 값 모두 답변·정답 목록에서 다시 계산한다 (run 에 기록된 acc 와 다르면 meta 에 개수 보고).

출력 (results/official_crag/analysis/pairwise/<name>/<ts>/, 이미 있으면 실패)
  pairwise_accuracy.csv / pairwise_official_match.csv
      쌍별 n, 각 정확도, 차이 (B-A), 둘 다 정답 / 둘 다 오답, HARM (A 만 정답) / GAIN (B 만 정답),
      McNemar 정확 이항 p (양측), 문항 bootstrap 95% CI (seed 고정)
  metric_agreement.csv  run 별 두 지표가 다른 문항 수 (대소문자·구두점 차이)
  flips.csv             쌍별로 어느 한 지표에서라도 뒤집힌 문항
                        (지표별 방향, 한 지표에서만 뒤집혔는지)
  special_cases.csv     뒤집힌 문항에서 **틀린 쪽 답** 을 대상으로 한 특수 사례 후보와 근거
                        (동명이인 / 동명 작품 / 정답 alias·표기 / 데이터셋 정답 의심), 수동 확인 열
  report.md, analysis_meta.json

특수 사례 후보 규칙 (모두 "후보". manual_label 로 확정)
  alias_surface   틀린 답이 정답과 토큰이 겹치거나 (F1>0), 정답의 일부·악센트만 다른 표기이거나,
                  두 지표가 이 답에서 서로 다름 (대소문자·구두점)
  homonym_person / homonym_work / homonym_other
                  틀린 답의 핵심 어구가 질문 대상 이름을 포함한 문서에 있고, 그 문서들에
                  동명 구분 표지 ("may refer to", "(film)", "(born ...)", 대상 이름 뒤 서로 다른
                  괄호 설명) 가 있으며
                  그 문서에 정답 문자열이 없음. 사람/작품 구분은 PopQA 질문 템플릿의 관계로 정함
  dataset_error   틀린 답의 핵심 어구가 질문 대상을 언급한 문서에 있고 (동명 표지 없음),
                  어떤 문서에도
                  정답 문자열이 없음 -> 정답 목록이 문서와 다를 가능성. 또는 정답 목록이 비어 있음
threshold 는 이 스크립트에서 다루지 않는다 (결과를 보고 조정하지 않는다: README 11.6).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_flips import (  # noqa: E402
    ROOT,
    frozen_texts,
    full_view,
    load_any,
    load_chunk_texts,
    load_web,
    md_table,
    resolve,
    run_web_sources,
    write_csv,
)
from analyze_regression import gold_norms, has_gold  # noqa: E402

from rag.crag_official.generation import official_match  # noqa: E402
from rag.eval.metrics import accuracy, f1_score, normalize_answer  # noqa: E402

METRICS = ("accuracy", "official_match")

# PopQA 질문 템플릿 (Mallen et al. 2023) -> (관계, 대상 종류)
TEMPLATES = [
    (r"^What is (?P<s>.+)'s occupation\?$", "occupation", "person"),
    (r"^In what city was (?P<s>.+) born\?$", "place_of_birth", "person"),
    (r"^Who is the father of (?P<s>.+)\?$", "father", "person"),
    (r"^Who is the mother of (?P<s>.+)\?$", "mother", "person"),
    (r"^What is the religion of (?P<s>.+)\?$", "religion", "person"),
    (r"^What genre is (?P<s>.+)\?$", "genre", "work"),
    (r"^Who was the producer of (?P<s>.+)\?$", "producer", "work"),
    (r"^Who was the director of (?P<s>.+)\?$", "director", "work"),
    (r"^Who was the screenwriter for (?P<s>.+)\?$", "screenwriter", "work"),
    (r"^Who was the composer of (?P<s>.+)\?$", "composer", "work"),
    (r"^Who is the author of (?P<s>.+)\?$", "author", "work"),
    (r"^In what country is (?P<s>.+)\?$", "country", "other"),
    (r"^What is the capital of (?P<s>.+)\?$", "capital_of", "other"),
    (r"^What is (?P<s>.+) the capital of\?$", "capital", "other"),
    (r"^What color is (?P<s>.+)\?$", "color", "other"),
    (r"^What sport does (?P<s>.+) play\?$", "sport", "other"),
]
DISAMBIG = re.compile(
    r"may (?:also )?refer to|\(disambiguation\)|\((?:\d{4} )?(?:film|album|song|single|novel|book|"
    r"play|opera|band|tv series|television series|video game|musical|poem|painting|ep)\)|"
    r"\(born \d|\((?:politician|footballer|actor|actress|singer|writer|painter|musician|"
    r"composer|cricketer|boxer|athlete)\)",
    re.IGNORECASE,
)
STOP = set(
    "a an the of in on at to for by with from and or is was were be been who what which where "
    "when how his her its their he she it they this that these those answer question".split()
)


# ---------------------------------------------------------------- statistics
def mcnemar_p(b: int, c: int) -> float | None:
    from math import comb

    n = b + c
    if n == 0:
        return None
    p = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * p)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple:
    if n_boot <= 0 or len(x) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    d = y.astype(float) - x.astype(float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pair_stats(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    harm = int(((x == 1) & (y == 0)).sum())
    gain = int(((x == 0) & (y == 1)).sum())
    lo, hi = bootstrap_ci(x, y, n_boot, seed)
    n = len(x)
    return {
        "n": n,
        "a_rate": float(x.mean()) if n else None,
        "b_rate": float(y.mean()) if n else None,
        "diff_b_minus_a": float(y.mean() - x.mean()) if n else None,
        "ci95_low": lo,
        "ci95_high": hi,
        "both_correct": int(((x == 1) & (y == 1)).sum()),
        "both_wrong": int(((x == 0) & (y == 0)).sum()),
        "harm_a_only": harm,
        "gain_b_only": gain,
        "mcnemar_p": mcnemar_p(harm, gain),
    }


# ---------------------------------------------------------------- special cases
def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return normalize_answer("".join(ch for ch in s if not unicodedata.combining(ch)))


def parse_question(q: str) -> tuple[str | None, str | None, str]:
    for pat, prop, kind in TEMPLATES:
        m = re.match(pat, (q or "").strip())
        if m:
            return m.group("s"), prop, kind
    return None, None, "unknown"


def answer_spans(answer: str, question: str) -> list[str]:
    """틀린 답의 핵심 어구: 대문자로 시작하는 연속 단어 / 숫자 (질문에 있는 말 제외).

    답이 5단어 이하면 답 전체도 후보."""
    qn = fold(question)
    spans = re.findall(
        r"(?:[A-Z][\w'’.-]*|\d[\d,.]*)(?:\s+(?:[A-Z][\w'’.-]*|of|de|von|van|\d+))*", answer or ""
    )  # noqa: E501
    out = []
    for sp in spans:
        f = fold(sp)
        if f and f not in STOP and f" {f} " not in f" {qn} " and len(f) > 2:
            out.append(sp)
    core = [t for t in fold(answer).split() if t not in STOP]
    if 0 < len(core) <= 5 and fold(answer) not in qn:
        out.append(answer.strip())
    return list(dict.fromkeys(out))


def review_case(question: str, golds: list[str], answer: str, docs: list[tuple[str, str]]) -> dict:
    """틀린 답 하나의 특수 사례 후보. docs = [(id, text)] (그 run 의 검색·web 문서)."""
    subj, prop, kind = parse_question(question)
    gn = gold_norms(golds)
    flags: list[str] = []
    notes: list[str] = []
    if not golds:
        flags.append("dataset_error")
        notes.append("정답 목록이 비어 있음")
    # alias / 표기
    f1 = f1_score(answer, golds) if golds else 0.0
    a_acc, a_om = accuracy(answer, golds), official_match(answer, golds)
    fa = fold(answer)
    accent = any(fold(g) and fold(g) in fa for g in golds) and not a_acc
    part = any(
        fa and fold(g) and (fa in fold(g)) and len(fa) >= 3 for g in golds
    )  # 답이 정답의 일부 (예: Obama / Barack Obama)
    if bool(a_acc) != bool(a_om):
        flags.append("alias_surface")
        notes.append(f"두 지표가 다름 (accuracy {a_acc:.0f}, official_match {a_om})")
    elif accent or part or (f1 > 0 and not a_acc):
        flags.append("alias_surface")
        notes.append(
            "악센트·기호만 다른 표기"
            if accent
            else ("답이 정답의 일부" if part else f"정답과 토큰 일부 겹침 (F1 {f1:.2f})")
        )
    # 문서 근거
    sn = fold(subj) if subj else None
    about = [(i, t) for i, t in docs if t and sn and f" {sn} " in f" {fold(t)} "]
    spans = answer_spans(answer, question)
    support = [
        (i, sp) for i, t in about for sp in spans if fold(sp) and f" {fold(sp)} " in f" {fold(t)} "
    ]
    gold_any = any(has_gold(t, gn) for _, t in docs if t) if gn else False
    marker_docs = [i for i, t in about if DISAMBIG.search(t)]
    parens = set()
    if subj:
        for _, t in about:
            parens |= {
                m.strip().lower()
                for m in re.findall(re.escape(subj) + r"\s*\(([^)]{1,40})\)", t, re.IGNORECASE)
            }
    homonym_sign = bool(marker_docs) or len(parens) >= 2
    sup_ids = sorted({i for i, _ in support})
    sup_wo_gold = [i for i in sup_ids if not has_gold(dict(docs).get(i), gn)]
    if support and homonym_sign and sup_wo_gold:
        flags.append(f"homonym_{kind if kind in ('person', 'work') else 'other'}")
        notes.append(
            f"답 어구가 대상 이름 문서 {sup_wo_gold} 에 있음 (정답 없음), 동명 표지: "
            + (f"문서 {marker_docs}" if marker_docs else "")
            + (f" 괄호 설명 {sorted(parens)}" if len(parens) >= 2 else "")
        )
    elif support and not gold_any and not homonym_sign:
        flags.append("dataset_error")
        notes.append(f"대상 이름 문서 {sup_ids} 가 답을 지지하고 어떤 문서에도 정답 문자열 없음")
    primary = next(
        (
            f
            for f in (
                "alias_surface",
                "homonym_person",
                "homonym_work",
                "homonym_other",
                "dataset_error",
            )
            if f in flags
        ),
        "none",
    )
    return {
        "subject": subj,
        "relation": prop,
        "subject_kind": kind,
        "candidate": primary,
        "candidate_flags": ";".join(dict.fromkeys(flags)),
        "evidence_note": " | ".join(notes),
        "answer_spans": " ; ".join(spans),
        "n_docs_about_subject": len(about),
        "gold_in_any_doc": gold_any,
        "token_f1": round(f1, 3),
    }


# ---------------------------------------------------------------- main
def parse_runs(items: list[str]) -> dict[str, Path]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--run NAME=DIR 형식: {it}")
        k, v = it.split("=", 1)
        if k in out:
            raise SystemExit(f"run 이름 중복: {k}")
        out[k] = resolve(v)
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="NAME=run 폴더 (2개 이상)")
    ap.add_argument("--baseline", default=None, help="기본: 첫 --run")
    ap.add_argument("--all-pairs", action="store_true")
    ap.add_argument("--docs-path", default=None)
    ap.add_argument("--web-cache", action="append", default=[])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="custom")
    ap.add_argument(
        "--out-root", default=str(ROOT / "results" / "official_crag" / "analysis" / "pairwise")
    )
    a = ap.parse_args(argv)
    dirs = parse_runs(a.run)
    if len(dirs) < 2:
        raise SystemExit("--run 이 2개 이상 필요")
    runs = {k: load_any(k, d) for k, d in dirs.items()}
    names = list(runs)
    base = a.baseline or names[0]
    if base not in runs:
        raise SystemExit(f"--baseline {base} 가 --run 에 없음")
    pairs = (
        list(itertools.combinations(names, 2))
        if a.all_pairs
        else [(base, n) for n in names if n != base]
    )

    # 문항별 두 지표 (답변·정답에서 재계산)
    scores: dict[str, dict[str, dict[str, int]]] = {}
    recorded_mismatch = {}
    for k, run in runs.items():
        sc = {}
        mism = 0
        for q, r in run["rows"].items():
            g = r.get("gold_answers") or []
            ans = r.get("answer") or ""
            acc = int(accuracy(ans, g))
            sc[q] = {"accuracy": acc, "official_match": int(official_match(ans, g))}
            if r.get("acc") is not None and int(bool(r["acc"])) != acc:
                mism += 1
        scores[k] = sc
        recorded_mismatch[k] = mism

    tables: dict[str, list[dict]] = {m: [] for m in METRICS}
    flips: list[dict] = []
    for ra, rb in pairs:
        common = sorted(set(scores[ra]) & set(scores[rb]))
        for m in METRICS:
            x = np.array([scores[ra][q][m] for q in common], dtype=int)
            y = np.array([scores[rb][q][m] for q in common], dtype=int)
            tables[m].append({"a": ra, "b": rb, **pair_stats(x, y, a.bootstrap, a.seed)})
        for q in common:
            fl = {}
            for m in METRICS:
                xa, yb = scores[ra][q][m], scores[rb][q][m]
                fl[m] = None if xa == yb else ("harm" if xa else "gain")
            if fl["accuracy"] is None and fl["official_match"] is None:
                continue
            only = None
            if (fl["accuracy"] is None) != (fl["official_match"] is None):
                only = "accuracy" if fl["accuracy"] else "official_match"
            flips.append(
                {
                    "a": ra,
                    "b": rb,
                    "query_id": q,
                    "flip_accuracy": fl["accuracy"],
                    "flip_official_match": fl["official_match"],
                    "flip_in_one_metric_only": only,
                    "a_action": runs[ra]["rows"][q].get("corrective_action"),
                    "b_action": runs[rb]["rows"][q].get("corrective_action"),
                }
            )

    # 특수 사례 후보: 뒤집힌 문항에서 틀린 쪽 답
    need: dict[str, set[str]] = {}
    targets: list[tuple[dict, str]] = []
    for f in flips:
        d = f["flip_accuracy"] or f["flip_official_match"]
        wrong = f["b"] if d == "harm" else f["a"]
        targets.append((f, wrong))
        fv = full_view(runs[wrong]["rows"][f["query_id"]])
        need.setdefault(wrong, set()).update(fv["ret_ids"])
    texts: dict[str, str] = {}
    web_paths: list[Path] = [resolve(p) for p in a.web_cache]
    for k, run in runs.items():
        dp = resolve(a.docs_path or (run["cfg"].get("retriever") or {}).get("docs_path"))
        ids = need.get(k, set()) - set(texts)
        if dp and dp.exists() and dp.suffix == ".jsonl" and ids:
            texts.update(load_chunk_texts(dp, ids))
        if run["format"] == "mode_b" and ids - set(texts):
            texts.update(frozen_texts(run, ids - set(texts)))
        web_paths += run_web_sources(run)
        if run["format"] == "mode_b":
            ws = run["cfg"].get("web_search") or {}
            web_paths += [resolve(p) for p in ws.get("replay_paths") or []]
            web_paths += [
                resolve(p) / "web_cache.jsonl"
                for p in ws.get("replay_from_runs") or []
                if resolve(p)
            ]
    web_paths = list(dict.fromkeys(p for p in web_paths if p and p.exists()))
    web_by_query, web_by_id = load_web(web_paths)

    special = []
    seen = set()
    for f, wrong in targets:
        q = f["query_id"]
        r = runs[wrong]["rows"][q]
        key = (wrong, q)
        fv = full_view(r)
        by_q = {str(x["id"]): x.get("text") for x in web_by_query.get((fv["rewritten"] or ""), [])}
        ids = list(dict.fromkeys(fv["ret_ids"] + fv["web_ids"] + fv["ctx"]))
        docs = [(i, texts.get(i) or by_q.get(i) or web_by_id.get(i)) for i in ids]
        rev = review_case(
            r.get("question") or "", r.get("gold_answers") or [], r.get("answer") or "", docs
        )  # noqa: E501
        f["special_candidate"] = rev["candidate"]
        if key in seen:
            continue
        seen.add(key)
        special.append(
            {
                "run": wrong,
                "query_id": q,
                "in_pairs": ";".join(
                    f"{x['a']}>{x['b']}" for x, w in targets if x["query_id"] == q and w == wrong
                ),
                "question": r.get("question"),
                "gold": " | ".join((r.get("gold_answers") or [])[:6]),
                "answer": (r.get("answer") or "")[:300],
                "action": r.get("corrective_action"),
                **rev,
                "n_docs_text_missing": sum(t is None for _, t in docs),
                "manual_label": "",
                "manual_note": "",
            }
        )

    agree = []
    for k in names:
        sc = scores[k].values()
        n = len(scores[k])
        agree.append(
            {
                "run": k,
                "n": n,
                "accuracy": sum(s["accuracy"] for s in sc) / n if n else None,
                "official_match": sum(s["official_match"] for s in sc) / n if n else None,
                "acc1_om0": sum(s["accuracy"] and not s["official_match"] for s in sc),
                "acc0_om1": sum(s["official_match"] and not s["accuracy"] for s in sc),
                "recorded_acc_mismatch": recorded_mismatch[k],
            }
        )

    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    for d in dirs.values():
        if out.resolve().is_relative_to(d.resolve()):
            raise ValueError(f"출력 폴더가 run 폴더 안에 있음: {out}")
    out.mkdir(parents=True, exist_ok=False)
    for m in METRICS:
        write_csv(out / f"pairwise_{m}.csv", tables[m])
    write_csv(out / "metric_agreement.csv", agree)
    write_csv(out / "flips.csv", flips)
    write_csv(out / "special_cases.csv", special)
    meta = {
        "created": datetime.now().isoformat(),
        "runs": {k: str(v) for k, v in dirs.items()},
        "formats": {k: r["format"] for k, r in runs.items()},
        "pairs": pairs,
        "bootstrap": a.bootstrap,
        "seed": a.seed,
        "recorded_acc_mismatch": recorded_mismatch,
        "availability": {"chunk_texts": len(texts), "web_sources": [str(p) for p in web_paths]},
        "note": "읽기 전용. 특수 사례는 문자열 규칙 기반 후보 (manual_label 로 확정)",
    }
    with open(out / "analysis_meta.json", "x", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    cand = {}
    for s in special:
        cand[s["candidate"]] = cand.get(s["candidate"], 0) + 1
    L = [
        "# Pairwise 비교 (accuracy / official_match 각각)",
        "",
        "- run: " + ", ".join(f"`{k}`" for k in names) + f" | 기준: `{base}`",
        f"- bootstrap {a.bootstrap}회 (seed {a.seed}), McNemar 정확 이항 (양측). "
        "같은 문항 집합 (두 run 공통) 에서만 비교",
        "",
        "## accuracy (정규화 포함, 기존 지표)",
        "",
        md_table(tables["accuracy"]),
        "## official_match (공식 eval.py match, 대소문자 구분)",
        "",
        md_table(tables["official_match"]),
        "## 두 지표가 다른 문항 수 (run 별)",
        "",
        md_table(agree),
        "## 특수 사례 후보 (뒤집힌 문항의 틀린 답 기준)",
        "",
        md_table([{"candidate": k, "n": v} for k, v in sorted(cand.items())]),
        "후보 정의는 스크립트 docstring. `special_cases.csv` 의 `manual_label` 에 "
        "homonym_person / homonym_work / alias / dataset_error / none 중 확정값을 적는다.",
        "",
        "## 특수 사례 후보 목록",
        "",
        md_table(
            [s for s in special if s["candidate"] != "none"],
            ["run", "query_id", "candidate", "question", "gold", "answer", "evidence_note"],
        ),
    ]
    with open(out / "report.md", "x", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"pairs={len(pairs)} flips={len(flips)} special={len(special)}")
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
