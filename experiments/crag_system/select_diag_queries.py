"""Grading 출력 원문 진단용 대표 질문 15개 선정 (읽기 전용) -> sequential workload.

    uv run python experiments/crag_system/select_diag_queries.py --preset a1-smoke \
        --name diag_gradetext_15

층 (중복 없이, 위에서부터 순서대로 뽑음):
  inc_gold_removed  2  Incorrect 회귀 중 grading 점수가 전부 0.0 이고, Vanilla context 에
                       정답 문자열이 있던 문서가 제거된 사례 (정답 문자열 = [추정] 지표)
  inc_zero          4  Incorrect 회귀 중 grading 점수가 전부 0.0 인 사례 (위에서 뽑은 것 제외)
  cor               3  Correct 회귀
  amb               2  Ambiguous 회귀
  ctrl_correct      2  Full 정답 대조군 (action=correct)
  ctrl_incorrect    2  Full 정답 대조군 (action=incorrect)
  회귀 = Vanilla 정답 & Full 오답. 0 이 아닌 점수가 섞인 Incorrect 회귀는 선택하지 않는다.
  대조군 두 층은 한쪽이 모자라면 다른 쪽 (그다음 다른 action 의 Full 정답) 으로 채운다.
  그 외 층이 모자라면 기본적으로 실패한다 (--allow-shortage 로 있는 만큼만).

층 안에서는 query_id 정렬 후 seed 고정 무작위 추출. 선택한 질문이 Full run 설정의 질문 pool
(seed shuffle + limit, warmup 제외) 에 있는지 확인한다 (runner 가 workload 를 받는 조건).

결과 (이미 있으면 실패):
  results/crag_system/workloads/<name>.json             sequential workload (run.py --workload)
  results/crag_system/workloads/<name>_selection.csv    선정 근거
  results/crag_system/workloads/<name>_selection.json   층별 후보 수·부족 여부·입력 경로
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_regression import (  # noqa: E402
    PRESETS,
    ROOT,
    any_gold,
    full_view,
    gold_norms,
    load_chunk_texts,
    resolve,
)
from analyze_runs import load_run  # noqa: E402
from workload import make_workload, save_workload  # noqa: E402

STRATA = (
    ("inc_gold_removed", 2),
    ("inc_zero", 4),
    ("cor", 3),
    ("amb", 2),
    ("ctrl_correct", 2),
    ("ctrl_incorrect", 2),
)
CTRL = ("ctrl_correct", "ctrl_incorrect")


def candidates(van: dict, full: dict, texts: dict[str, str]) -> tuple[dict, list[dict]]:
    """층별 후보 query_id 와 질문별 근거 행."""
    cand = {k: [] for k, _ in STRATA}
    cand["ctrl_other"] = []
    info = []
    for q in sorted(set(van["rows"]) & set(full["rows"])):
        v, f = van["rows"][q], full["rows"][q]
        fv = full_view(f)
        graded = [s for s in fv["scores"] if s is not None]
        act = f.get("corrective_action")
        reg = bool(v.get("acc")) and not f.get("acc")
        all_zero = bool(graded) and all(s == 0.0 for s in graded)
        v_ctx = [str(i) for i in v.get("contexts") or []]
        gold_v = any_gold([texts.get(i) for i in v_ctx], gold_norms(f.get("gold_answers") or []))
        row = {
            "query_id": q,
            "question": f.get("question"),
            "gold": " | ".join((f.get("gold_answers") or [])[:5]),
            "vanilla_acc": int(bool(v.get("acc"))),
            "full_acc": int(bool(f.get("acc"))),
            "action": act,
            "scores_full": json.dumps(fv["scores"]),
            "all_scores_zero": all_zero,
            "retrieval_doc_ids": json.dumps(fv["ret_ids"]),
            "est_gold_in_vanilla_ctx": gold_v,
            "full_answer": (f.get("answer") or "")[:200],
            "vanilla_answer": (v.get("answer") or "")[:200],
        }
        info.append(row)
        if reg and act == "incorrect":
            if not all_zero:
                continue  # 0 이 아닌 점수가 섞인 Incorrect 회귀는 선택 대상 아님
            cand["inc_zero"].append(q)
            # 정답 문자열 문서가 Vanilla context 에 있었고 Full 에서 모두 제거됨
            # (Incorrect = 유지 FAISS 0개)
            if gold_v and not set(v_ctx) & set(fv["kept_faiss"]):
                cand["inc_gold_removed"].append(q)
        elif reg and act == "correct":
            cand["cor"].append(q)
        elif reg and act == "ambiguous":
            cand["amb"].append(q)
        elif f.get("acc"):
            key = {"correct": "ctrl_correct", "incorrect": "ctrl_incorrect"}.get(act, "ctrl_other")
            cand[key].append(q)
    return cand, info


def select(cand: dict, seed: int) -> tuple[dict[str, list[str]], dict]:
    rng = random.Random(seed)
    chosen: dict[str, list[str]] = {}
    used: set[str] = set()
    shortage = {}

    def take(pool, n):
        avail = [q for q in sorted(pool) if q not in used]
        picks = sorted(rng.sample(avail, min(n, len(avail))))
        used.update(picks)
        return picks

    for name, n in STRATA:
        if name in CTRL:
            continue
        chosen[name] = take(cand[name], n)
        if len(chosen[name]) < n:
            shortage[name] = {"need": n, "got": len(chosen[name])}
    need_ctrl = sum(n for k, n in STRATA if k in CTRL)
    for name, n in STRATA:
        if name in CTRL:
            chosen[name] = take(cand[name], n)
    # 대조군 부족분: 다른 대조군 층 -> 다른 action 의 Full 정답
    for name in CTRL:
        other = CTRL[1] if name == CTRL[0] else CTRL[0]
        n = dict(STRATA)[name]
        while len(chosen[name]) < n:
            extra = take(cand[other], 1) or take(cand["ctrl_other"], 1)
            if not extra:
                break
            chosen[name] += extra
    got_ctrl = sum(len(chosen[k]) for k in CTRL)
    if got_ctrl < need_ctrl:
        shortage["controls"] = {"need": need_ctrl, "got": got_ctrl}
    return chosen, shortage


def pool_ids(cfg: dict) -> list[str]:
    from runner import apply_split, load_examples, split_warmup

    _, pool = split_warmup(cfg, load_examples(cfg))
    return [ex.id for ex in apply_split(cfg, pool)]


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--vanilla")
    ap.add_argument("--full")
    ap.add_argument(
        "--docs-path", help="chunk jsonl (기본: Full run config 의 retriever.docs_path)"
    )
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-shortage", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "crag_system" / "workloads"))
    a = ap.parse_args(argv)
    preset = PRESETS.get(a.preset, {}) if a.preset else {}
    v_dir, f_dir = (
        resolve(a.vanilla or preset.get("vanilla")),
        resolve(a.full or preset.get("full")),
    )
    if not v_dir or not f_dir:
        ap.error("--preset 또는 --vanilla / --full 이 필요")
    van, full = load_run("vanilla", v_dir), load_run("full", f_dir)
    docs_path = resolve(a.docs_path or (full["cfg"].get("retriever") or {}).get("docs_path"))
    need = {str(i) for r in van["rows"].values() for i in r.get("contexts") or []}
    texts = load_chunk_texts(docs_path, need)

    cand, info = candidates(van, full, texts)
    chosen, shortage = select(cand, a.seed)
    if shortage and not a.allow_shortage:
        raise SystemExit(
            f"후보 부족: {shortage} (후보 수: { {k: len(v) for k, v in cand.items()} }). "
            "--allow-shortage 로 있는 만큼만 선택 가능"
        )
    ids = [q for name, _ in STRATA for q in chosen[name]]
    assert len(ids) == len(set(ids)), "중복 선택"
    pool = set(pool_ids(full["cfg"]))
    outside = [q for q in ids if q not in pool]
    if outside:
        raise SystemExit(f"Full run 설정의 질문 pool 에 없는 질문: {outside}")

    out_dir = Path(a.out_dir)
    wl = make_workload(ids, seed=a.seed, arrival="sequential", name=a.name)
    wl["source"] = {
        "selection": "select_diag_queries.py",
        "vanilla": str(v_dir),
        "full": str(f_dir),
    }
    wl_path = save_workload(wl, out_dir / f"{a.name}.json")
    by_id = {r["query_id"]: r for r in info}
    rows = [{"stratum": name, **by_id[q]} for name, _ in STRATA for q in chosen[name]]
    with open(out_dir / f"{a.name}_selection.csv", "x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "vanilla": str(v_dir),
        "full": str(f_dir),
        "docs_path": str(docs_path) if docs_path else None,
        "chunk_texts_found": len(texts),
        "seed": a.seed,
        "strata": {name: n for name, n in STRATA},
        "n_candidates": {k: len(v) for k, v in cand.items()},
        "n_incorrect_regression_nonzero_excluded": sum(
            1
            for r in info
            if r["vanilla_acc"]
            and not r["full_acc"]
            and r["action"] == "incorrect"
            and not r["all_scores_zero"]
        ),  # fmt: skip
        "chosen": chosen,
        "shortage": shortage,
        "workload": str(wl_path),
        "note": "inc_gold_removed 의 정답 문자열 판정은 [추정] 지표 (정규화 + 단어 경계 일치)",
    }
    with open(out_dir / f"{a.name}_selection.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False))
    for name, _ in STRATA:
        print(f"{name:18s} {len(chosen[name])}/{dict(STRATA)[name]}  {chosen[name]}")
    print(f"saved -> {wl_path} ({len(ids)} queries)")
    return wl_path


if __name__ == "__main__":
    main()
