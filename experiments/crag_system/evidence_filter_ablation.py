"""Evidence filtering ablation: 같은 검색 결과·grading 점수·web 문서로 generation 만 다시.

기존 CRAG 실행 코드 (CRAGGraph, Runner, grading) 는 바꾸지 않고 호출하지도 않는다.
기준 run (crag_graph, full grading) 의 기록에서 질문별로 다음을 그대로 가져온다.
  - FAISS top-k 문서 id (retrieval_doc_ids) -> 본문은 retriever.docs_path (chunk jsonl)
  - 문서별 grading 점수 (extra.grading.scores_full) 와 corrective action
  - web 문서: 기준 run 의 rewrite 검색어로 web 기록 (web_cache.jsonl / replay 원본) 조회
조건마다 바뀌는 것은 generation 에 넣는 context 를 고르는 규칙 하나뿐이다. grading / rewrite /
web search 호출은 하지 않으므로 점수·판정·correction 비율은 모든 조건에서 같다.

조건 (action 은 기준 run 의 판정 그대로)
                          correct            ambiguous               incorrect
  crag_rule (기준)        FAISS >= upper     FAISS >= lower + web    web 만
  no_correct_filter       FAISS top-k        (기준과 같음)           (기준과 같음)
  no_ambiguous_filter     (기준과 같음)      FAISS top-k + web       (기준과 같음)
  no_incorrect_discard    (기준과 같음)      (기준과 같음)           FAISS top-k + web
  keep_all                FAISS top-k        FAISS top-k + web       FAISS top-k + web
  vanilla_ctx             FAISS top-k        FAISS top-k             FAISS top-k   (= Vanilla RAG)
  - crag_rule 은 rag.grading.base.select_contexts 와 CRAGGraph 의 context 순서
    (FAISS 검색 순서 + web) 를 그대로 쓴다. no_correct_filter / no_incorrect_discard 는
    ablation.py 의 correct_keep_topk / incorrect_keep_faiss 와 같은 context 지만,
    그 실험과 달리 grading 을 다시 하지 않는다.
  - 규칙은 action 별로 따로 적용되므로 crag_rule -> keep_all 차이는 세 단일 규칙 효과의 합이다
    (질문마다 한 규칙만 영향).
  - 같은 질문에서 context 문서 목록이 같은 조건은 generation 1번을 공유한다
    (같은 입력 -> 같은 출력. 조건 간 차이에 generation 비결정성이 섞이지 않음).

재구성 검증 (질문 단위, 하나라도 실패하면 그 질문은 모든 조건에서 제외하고 보고)
  - 점수로 다시 계산한 action == 기록된 action (decide_action)
  - 재구성한 crag_rule context id == 기준 run 의 contexts id
  - web 문서 id == 기준 run 의 web_doc_ids, 문서 본문이 docs_path 에 있음
  generation 재현성: crag_rule 답변이 기준 run 답변과 같은 비율 (generation 비결정성의 크기)

Train / Eval
  학습하거나 고르는 파라미터가 없다 (threshold 는 기준 run 설정 그대로).
  --split 으로 기준 run 의 dataset.split.part 를 명시해야 하며 다르면 중단한다.
  설계 확인·디버깅은 train run, 보고 수치는 eval run 으로 한다.

    # 생성 (서버, vLLM 필요)
    uv run python experiments/crag_system/evidence_filter_ablation.py generate \
        --ref results/crag_system/pre/PRE-SV1-A/eval_seq --split eval --exp-id PRE-EVF-SV1-EVAL
    # 분석만 (LLM 호출 없음)
    uv run python experiments/crag_system/evidence_filter_ablation.py analyze \
        --run results/crag_system/pre/PRE-EVF-SV1-EVAL/<ts>

생성 결과: results/crag_system/pre/<exp_id>/<ts>/ (새 폴더, 기존 run 은 읽기만)
  run_meta.json, items.jsonl (질문별 재구성 결과·검증), generations.jsonl (조건 x 질문)
  --resume <run 폴더> 로 중단된 생성을 이어서 (이미 있는 (질문, context) 는 건너뜀)
분석 결과: results/crag_system/analysis/evidence_filter/<name>/<ts>/
  report.md, summary.csv, paired.csv, by_action.csv, per_query.csv, analysis.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag.eval.metrics import normalize_answer, score_all  # noqa: E402
from rag.grading.base import decide_action, select_contexts  # noqa: E402
from rag.llm.engine import LLMRequest  # noqa: E402
from rag.pipelines import prompts  # noqa: E402
from rag.retrieval.web_cache import read_jsonl  # noqa: E402
from rag.telemetry.run_meta import create_run_dir, git_info, write_new  # noqa: E402
from rag.types import Document  # noqa: E402

ACTIONS = ("correct", "ambiguous", "incorrect")
# action -> keep (FAISS top-k 유지) 여부. web 은 correction (ambiguous / incorrect) 일 때만
CONDITIONS = {
    "crag_rule": {"correct": False, "ambiguous": False, "incorrect": False, "web": True},
    "no_correct_filter": {"correct": True, "ambiguous": False, "incorrect": False, "web": True},
    "no_ambiguous_filter": {"correct": False, "ambiguous": True, "incorrect": False, "web": True},
    "no_incorrect_discard": {"correct": False, "ambiguous": False, "incorrect": True, "web": True},
    "keep_all": {"correct": True, "ambiguous": True, "incorrect": True, "web": True},
    "vanilla_ctx": {"correct": True, "ambiguous": True, "incorrect": True, "web": False},
}
BASE = "crag_rule"
COMPARISONS = (
    ("crag_rule", "no_correct_filter"), ("crag_rule", "no_ambiguous_filter"),
    ("crag_rule", "no_incorrect_discard"), ("crag_rule", "keep_all"),
    ("vanilla_ctx", "crag_rule"), ("vanilla_ctx", "keep_all"),
)  # fmt: skip


# ---------------------------------------------------------------- context rules
def build_context(cond: str, action: str, docs: list, scores: list, web: list, upper: float,
                  lower: float) -> list:  # fmt: skip
    """조건별 generation context. crag_rule 은 CRAGGraph (_decide / _web_search) 와 같은 결과."""
    rule = CONDITIONS[cond]
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    if rule[action]:
        faiss = list(docs)
    elif action == "incorrect":
        faiss = []
    else:
        faiss = select_contexts(docs, scores, action, upper, lower)
    use_web = rule["web"] and action != "correct"
    return faiss + (list(web) if use_web else [])


# ---------------------------------------------------------------- reference run
def ref_config(ref: Path) -> dict:
    return yaml.safe_load((ref / "config.yaml").read_text(encoding="utf-8"))


def web_sources(ref: Path, cfg: dict, extra: list[str]) -> list[Path]:
    w = (cfg.get("crag") or {}).get("web_search") or {}
    paths = [ref / "web_cache.jsonl"]
    paths += [Path(p) for p in w.get("replay_paths") or []]
    paths += [Path(r) / "web_cache.jsonl" for r in w.get("replay_from_runs") or []]
    paths += [Path(p) for p in extra]
    return [p for p in dict.fromkeys(paths) if p.exists()]


def load_web(paths: list[Path]) -> dict[tuple[str, int], list[Document]]:
    out: dict = {}
    for p in paths:
        for r in read_jsonl(p):
            out.setdefault(
                (r["query"].strip(), int(r["top_k"])),
                [Document(id=d["id"], text=d["text"], metadata=d.get("metadata") or {},
                          score=d.get("score")) for d in r["docs"]],
            )  # fmt: skip
    return out


def load_docs(path: Path, ids: set[str]) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            i = str(r["id"])
            if i in ids:
                out[i] = r["text"]
    return out


def load_items(ref: Path, split: str, docs_path: str | None = None,
               web_paths: tuple = ()) -> tuple[list[dict], dict]:  # fmt: skip
    """기준 run -> 질문별 재구성 항목 (검증 결과 포함) 과 메타."""
    cfg = ref_config(ref)
    part = ((cfg.get("dataset") or {}).get("split") or {}).get("part", "all")
    if part != split:
        raise SystemExit(f"기준 run 의 split 은 {part!r} 인데 --split {split!r} (데이터 누수 방지)")
    if cfg.get("pipeline", "crag_graph") != "crag_graph":
        raise SystemExit(f"crag_graph run 만 지원 (got {cfg.get('pipeline')})")
    if "ablation" in cfg:
        raise SystemExit("기준 run 이 ablation run 임 (기준은 일반 CRAG run)")
    c = cfg.get("crag") or {}
    upper, lower = c.get("upper_threshold", 0.7), c.get("lower_threshold", 0.3)
    web_k = int((c.get("web_search") or {}).get("top_k", 5))
    rows = [r for r in read_jsonl(ref / "queries.jsonl") if not r.get("warmup")]
    n_error = sum(1 for r in rows if r.get("error"))
    rows = [r for r in rows if not r.get("error")]
    dp = Path(docs_path or (cfg.get("retriever") or {})["docs_path"])
    if not dp.is_absolute() and not dp.exists():
        dp = ROOT / dp
    need = {i for r in rows for i in r.get("retrieval_doc_ids") or []}
    texts = load_docs(dp, need)
    wsrc = web_sources(ref, cfg, list(web_paths))
    web = load_web(wsrc)
    items = []
    for r in rows:
        ex = r.get("extra") or {}
        g = ex.get("grading") or {}
        ids = list(r.get("retrieval_doc_ids") or ex.get("retrieval_doc_ids") or [])
        scores = list(g.get("scores_full") or [])
        action = r.get("corrective_action")
        problems = []
        if g.get("strategy") not in ("full", "full_parallel"):
            problems.append(f"grading={g.get('strategy')}")
        if len(scores) != len(ids):
            problems.append("score/doc 수 불일치")
        missing = [i for i in ids if i not in texts]
        if missing:
            problems.append(f"문서 본문 없음 {len(missing)}")
        docs = [Document(id=i, text=texts.get(i, "")) for i in ids]
        if not problems and decide_action(scores, upper, lower) != action:
            problems.append("점수로 다시 계산한 action 불일치")
        wdocs: list = []
        if action in ("ambiguous", "incorrect"):
            rq = (ex.get("rewritten_query") or "").strip()
            got = web.get((rq, web_k))
            if got is None:
                problems.append("web 기록 없음")
            else:
                wdocs = got
                if [d.id for d in got] != list(ex.get("web_doc_ids") or []):
                    problems.append("web 문서 id 불일치")
        if not problems:
            ctx = build_context(BASE, action, docs, scores, wdocs, upper, lower)
            if [d.id for d in ctx] != list(r.get("contexts") or []):
                problems.append("crag_rule context 재구성 불일치")
        items.append(
            {
                "request_index": r["request_index"], "query_id": r["query_id"],
                "question": r["question"], "gold": r.get("gold_answers") or [],
                "action": action, "scores": scores, "doc_ids": ids, "docs": docs,
                "web": wdocs, "ref_answer": r.get("answer"), "ref_acc": r.get("acc"),
                "problems": problems,
            }
        )  # fmt: skip
    meta = {
        "ref": str(ref), "split": part, "upper": upper, "lower": lower, "web_top_k": web_k,
        "docs_path": str(dp), "web_sources": [str(p) for p in wsrc], "n_ref_rows": len(rows),
        "n_ref_errors": n_error, "llm": {k: v for k, v in (cfg.get("llm") or {}).items()
                                         if k != "api_key_env"},
        "grading": cfg.get("grading"),
        "config_sha256": hashlib.sha256((ref / "config.yaml").read_bytes()).hexdigest(),
    }  # fmt: skip
    return items, meta


def gold_in(texts: list[str], gold: list[str]) -> bool:
    """[문자열] 참고 지표: 정규화한 정답 문자열이 context 에 있는가 (근거 여부의 확정 아님)."""
    blob = " " + normalize_answer(" ".join(texts)) + " "
    return any(f" {normalize_answer(a)} " in blob for a in gold if normalize_answer(a))


# ---------------------------------------------------------------- generate
def generate(items: list[dict], meta: dict, engine, out: Path, conditions: list[str],
             done: set | None = None, limit: int | None = None) -> Path:  # fmt: skip
    done = done or set()
    ok = [it for it in items if not it["problems"]][: limit or None]
    gen_path = out / "generations.jsonl"
    with open(gen_path, "a", encoding="utf-8") as f:
        for n, it in enumerate(ok, 1):
            cache: dict[tuple, dict] = {}
            for cond in conditions:
                ctx = build_context(cond, it["action"], it["docs"], it["scores"], it["web"],
                                    meta["upper"], meta["lower"])  # fmt: skip
                ids = tuple(d.id for d in ctx)
                key = (it["query_id"], cond)
                if key in done:
                    continue
                shared = ids in cache
                if not shared:
                    prompt = prompts.GENERATE.format(
                        context=prompts.format_context(ctx), question=it["question"]
                    )
                    rec = {"error": None}
                    try:
                        resp = engine.complete(LLMRequest(prompt=prompt, kind="generate"))
                        rec.update(
                            answer=resp.text, prompt_tokens=resp.prompt_tokens,
                            out_tokens=resp.completion_tokens, finish_reason=resp.finish_reason,
                        )  # fmt: skip
                    except Exception as e:  # noqa: BLE001 기록하고 계속
                        rec.update(answer=None, error=f"{type(e).__name__}: {e}"[:500])
                    rec["prompt_chars"] = len(prompt)
                    cache[ids] = rec
                rec = cache[ids]
                s = score_all(rec["answer"], it["gold"]) if rec["answer"] is not None else {}
                n_faiss = sum(1 for d in ctx if d.id in set(it["doc_ids"]))
                row = {
                    "query_id": it["query_id"], "request_index": it["request_index"],
                    "condition": cond, "action": it["action"], "ctx_ids": list(ids),
                    "n_ctx": len(ctx), "n_faiss": n_faiss, "n_web": len(ctx) - n_faiss,
                    "ctx_chars": sum(len(d.text) for d in ctx),
                    "gold_in_ctx": gold_in([d.text for d in ctx], it["gold"]),
                    "shared_generation": shared, **rec, **s,
                    "ref_answer_same": rec["answer"] == it["ref_answer"],
                }  # fmt: skip
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"{n}/{len(ok)}", flush=True)
    return gen_path


def write_items(out: Path, items: list[dict]) -> None:
    with open(out / "items.jsonl", "x", encoding="utf-8") as f:
        for it in items:
            d = {k: v for k, v in it.items() if k not in ("docs", "web")}
            d["web_ids"] = [w.id for w in it["web"]]
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- analyze
def mcnemar_exact(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired_acc(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    """y - x (정확도 차이). 질문 복원 추출 bootstrap 95% CI + McNemar exact."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    d = (y[idx] - x[idx]).mean(axis=1)
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    return {
        "n": len(x), "acc_base": float(x.mean()), "acc_target": float(y.mean()),
        "diff": float(y.mean() - x.mean()), "ci_lo": float(np.percentile(d, 2.5)),
        "ci_hi": float(np.percentile(d, 97.5)), "base_only_correct": b, "target_only_correct": c,
        "mcnemar_p": mcnemar_exact(b, c),
    }  # fmt: skip


def analyze(run: Path, n_boot: int = 5000) -> dict:
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    items = read_jsonl(run / "items.jsonl")
    gens = read_jsonl(run / "generations.jsonl")
    by = defaultdict(dict)  # cond -> qid -> row
    for g in gens:
        by[g["condition"]][g["query_id"]] = g
    conds = [c for c in CONDITIONS if c in by]
    errs = {c: sum(1 for g in by[c].values() if g.get("error")) for c in conds}
    common = set.intersection(*[{q for q, g in by[c].items() if not g.get("error")}
                                for c in conds]) if conds else set()  # fmt: skip
    qids = sorted(common)
    excluded = [
        {"query_id": it["query_id"], "action": it["action"], "problems": "; ".join(it["problems"])}
        for it in items
        if it["problems"]
    ]
    summary = []
    for c in conds:
        rs = [by[c][q] for q in qids]

        def m(k, rs=rs):
            v = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(v)) if v else None

        summary.append(
            {
                "condition": c, "n": len(rs), "acc": m("acc"), "em": m("em"), "f1": m("f1"),
                "n_ctx": m("n_ctx"), "n_faiss_kept": m("n_faiss"), "n_web": m("n_web"),
                "empty_ctx_share": float(np.mean([r["n_ctx"] == 0 for r in rs])) if rs else None,
                "gold_in_ctx": m("gold_in_ctx"), "prompt_tokens": m("prompt_tokens"),
                "out_tokens": m("out_tokens"), "ctx_chars": m("ctx_chars"),
                "correction_rate": float(np.mean([r["action"] != "correct" for r in rs]))
                if rs else None,
                "web_used_share": float(np.mean([r["n_web"] > 0 for r in rs])) if rs else None,
                "ref_answer_same": m("ref_answer_same"), "errors": errs[c],
            }
        )  # fmt: skip
    paired, by_action = [], []
    acc = {c: np.array([by[c][q]["acc"] for q in qids], dtype=float) for c in conds}
    act = np.array([by[conds[0]][q]["action"] for q in qids]) if conds else np.array([])
    for base, target in COMPARISONS:
        if base in acc and target in acc and len(qids) > 1:
            for sub in ("all", *ACTIONS):
                mask = np.ones(len(qids), bool) if sub == "all" else act == sub
                if mask.sum() < 1:
                    continue
                pa = paired_acc(acc[base][mask], acc[target][mask], n_boot)
                paired.append({"base": base, "target": target, "subset": sub, **pa})
    for c in conds:
        for a in ACTIONS:
            rs = [by[c][q] for q in qids if by[c][q]["action"] == a]
            if rs:
                by_action.append(
                    {"condition": c, "action": a, "n": len(rs),
                     "acc": float(np.mean([r["acc"] for r in rs])),
                     "n_faiss_kept": float(np.mean([r["n_faiss"] for r in rs])),
                     "n_web": float(np.mean([r["n_web"] for r in rs])),
                     "gold_in_ctx": float(np.mean([r["gold_in_ctx"] for r in rs])),
                     "prompt_tokens": float(np.mean([r["prompt_tokens"] or 0 for r in rs]))}
                )  # fmt: skip
    per_query = [
        {"query_id": q, "action": by[conds[0]][q]["action"],
         **{f"acc_{c}": by[c][q]["acc"] for c in conds},
         **{f"nctx_{c}": by[c][q]["n_ctx"] for c in conds}}
        for q in qids
    ]  # fmt: skip
    return {
        "run": str(run), "meta": meta, "n_items": len(items), "n_excluded": len(excluded),
        "excluded": excluded, "n_common": len(qids), "summary": summary, "paired": paired,
        "by_action": by_action, "per_query": per_query,
    }  # fmt: skip


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return out


def report_md(res: dict, name: str) -> str:
    m = res["meta"]
    L = [
        f"# Evidence filtering ablation ({name})", "",
        f"- 기준 run: `{m['ref']['ref']}` (split={m['ref']['split']}, upper={m['ref']['upper']}, "
        f"lower={m['ref']['lower']})",
        f"- 질문: 기준 {m['ref']['n_ref_rows']} (오류 {m['ref']['n_ref_errors']} 제외) / "
        f"재구성 실패 제외 {res['n_excluded']} / 모든 조건 정상 {res['n_common']}",
        "- 모든 조건이 같은 FAISS 문서·grading 점수·action·web 문서를 쓴다."
        " correction 비율은 조건 간 동일 (grading 을 다시 하지 않음)",
        "- gold_in_ctx 는 [문자열] 참고 지표 (정답 문자열 포함, 근거 확정 아님)", "",
        "## 조건별 요약", "",
    ]  # fmt: skip
    L += md_table(res["summary"], ["condition", "n", "acc", "em", "f1", "n_ctx", "n_faiss_kept",
                                   "n_web", "empty_ctx_share", "gold_in_ctx", "prompt_tokens",
                                   "out_tokens", "correction_rate", "web_used_share",
                                   "ref_answer_same", "errors"])  # fmt: skip
    L += ["", "- ref_answer_same (crag_rule): 기준 run 답변과 같은 비율 = generation 재현성. "
          "1 보다 많이 낮으면 기준 run 과의 직접 비교는 이 차이만큼 잡음이 섞인다", ""]  # fmt: skip
    L += ["## paired 정확도 차이 (target - base, bootstrap 95% CI, McNemar exact)", ""]
    L += md_table(res["paired"], ["base", "target", "subset", "n", "acc_base", "acc_target",
                                  "diff", "ci_lo", "ci_hi", "base_only_correct",
                                  "target_only_correct", "mcnemar_p"])  # fmt: skip
    L += ["", "- subset 은 기준 run 의 action. 각 단일 규칙은 해당 action 질문에만 영향을 준다", ""]
    L += ["## action 별", ""]
    L += md_table(res["by_action"], ["condition", "action", "n", "acc", "n_faiss_kept", "n_web",
                                     "gold_in_ctx", "prompt_tokens"])  # fmt: skip
    L += ["", "## 재구성 실패로 제외된 질문", ""]
    ex = res["excluded"]
    L += md_table(ex[:30], ["query_id", "action", "problems"]) if ex else ["- 없음"]
    L += [""]
    return "\n".join(L)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_analysis(run: Path, name: str, out_root: Path, n_boot: int, quiet: bool) -> Path:
    res = analyze(run, n_boot)
    base = out_root / "evidence_filter" / name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out, n = base, 1
    while out.exists():  # 같은 초에 다시 분석해도 기존 결과를 덮어쓰지 않음
        n += 1
        out = base.with_name(f"{base.name}_{n}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "analysis.json").write_text(
        json.dumps({k: v for k, v in res.items() if k != "per_query"}, indent=2,
                   ensure_ascii=False, default=str), encoding="utf-8")  # fmt: skip
    write_csv(out / "summary.csv", res["summary"])
    write_csv(out / "paired.csv", res["paired"])
    write_csv(out / "by_action.csv", res["by_action"])
    write_csv(out / "per_query.csv", res["per_query"])
    write_csv(out / "excluded.csv", res["excluded"])
    md = report_md(res, name)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not quiet:
        print(md)
    print(f"saved -> {out}")
    return out


# ---------------------------------------------------------------- main
def main(argv=None, engine=None) -> Path:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--ref", required=True, help="기준 CRAG run 폴더 (crag_graph, full grading)")
    g.add_argument("--split", required=True, choices=["train", "eval", "all"])
    g.add_argument("--exp-id", default=None, help="PRE-* (새 run 폴더)")
    g.add_argument("--resume", default=None, help="중단된 생성 run 폴더")
    g.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    g.add_argument("--docs-path", default=None, help="config 의 retriever.docs_path 대신")
    g.add_argument("--web-cache", nargs="*", default=[], help="추가 web 기록 jsonl")
    g.add_argument("--limit", type=int, default=None, help="smoke 용 질문 수 제한")
    g.add_argument("--check-only", action="store_true", help="재구성 검증만 (LLM 호출 없음)")
    g.add_argument("--output-dir", default="results")
    for p in (g, sub.add_parser("analyze")):
        p.add_argument("--name", default=None)
        p.add_argument("--n-boot", type=int, default=5000)
        p.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
        p.add_argument("--quiet", action="store_true")
    sub.choices["analyze"].add_argument("--run", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "analyze":
        run = Path(a.run)
        return write_analysis(run, a.name or run.parent.name, Path(a.out_root), a.n_boot, a.quiet)

    items, ref_meta = load_items(Path(a.ref), a.split, a.docs_path, tuple(a.web_cache))
    n_bad = sum(1 for it in items if it["problems"])
    print(f"재구성: {len(items)} 질문, 실패 {n_bad}")
    for it in [it for it in items if it["problems"]][:10]:
        print(f"  {it['query_id']} {it['action']}: {it['problems']}")
    if a.check_only:
        return Path(a.ref)
    from rag.llm.factory import build_engine

    if engine is None:
        llm = dict(ref_meta["llm"])
        llm["api_key_env"] = ref_config(Path(a.ref)).get("llm", {}).get("api_key_env",
                                                                       "VLLM_API_KEY")  # fmt: skip
        engine = build_engine(llm)
    if a.resume:
        out = Path(a.resume)
        prev = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
        if prev["ref"]["config_sha256"] != ref_meta["config_sha256"]:
            raise SystemExit("resume 대상의 기준 run 이 다름")
        done = {(r["query_id"], r["condition"]) for r in read_jsonl(out / "generations.jsonl")
                if not r.get("error")}  # fmt: skip
    else:
        if not a.exp_id or not a.exp_id.startswith("PRE-"):
            raise SystemExit("--exp-id 는 PRE-* (예비·진단 실험)")
        out = create_run_dir(f"crag_system/pre/{a.exp_id}", a.output_dir)
        write_new(out / "run_meta.json", json.dumps(
            {"created": datetime.now().isoformat(), "ref": ref_meta, "conditions": a.conditions,
             "git": git_info(), "limit": a.limit,
             "engine": engine.describe() if hasattr(engine, "describe") else None},
            indent=2, ensure_ascii=False, default=str))  # fmt: skip
        write_items(out, items)
        done = set()
    generate(items, ref_meta, engine, out, a.conditions, done, a.limit)
    name = a.name or a.exp_id or out.parent.name
    write_analysis(out, name, Path(a.out_root), a.n_boot, a.quiet)
    return out


if __name__ == "__main__":
    main()
