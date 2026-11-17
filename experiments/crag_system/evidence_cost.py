"""Evidence preservation 의 정확도-latency-GPU 비용 trade-off (evidence_filter_ablation 후속).

기존 run 은 읽기만 한다. CRAGGraph / grading / gate / scheduler 정책은 쓰지도 바꾸지도 않는다.

입력
  --evf  evidence_filter_ablation 생성 run (items / generations.jsonl). 조건별 정확도·토큰의 출처.
         latency 는 기록되어 있지 않고 (토큰만), 같은 질문의 조건들을 연달아 생성해
         prefix cache 가 섞이므로 latency 로 재사용할 수 없다.
  --seq  기준 sequential run (= evf 의 기준 run, 예: PRE-SV1-A/eval_seq). crag_rule 경로의
         stage 시간·호출별 wall / TTFT / 토큰이 [관측] 으로 있다.

하위 명령
  measure      (서버, vLLM) 조건별 generation 만 sequential 로 다시 보내 latency 를 잰다.
               - 질문·검색 문서·grading 점수·action·web 문서는 evf 와 같다 (evf 의 재구성 그대로,
                 context id 가 evf 기록과 다르면 그 질문은 제외).
               - 같은 질문에서 context 가 같은 조건은 한 번만 측정하고 결과를 공유
                 (중복 집계 방지).
               - 측정 순서는 (질문, context) 단위로 seed 섞기 (조건 순서가 고정되지 않게).
               - --prefix-cache reset: 매 호출 전 prefix cache 초기화 (서버 --dev-mode 필요).
                 none: 초기화 없음, 대신 호출마다 /metrics 의 prefix cache hit 증가분을 기록.
               - 생성 설정 (model, temperature, max_tokens, seed, stream) 이 evf 와 다르면 중단.
  report       (LLM 호출 없음) 조건별 정확도 + generation latency + 재구성 E2E + 토큰 비용.
  live-report  (LLM 호출 없음) run_evidence_live.py 로 실행한 live run 들의 정확도·지연·처리량.

근거 표시 (보고서의 모든 latency 열에 붙는다)
  [관측]  기존 run 의 기록 그대로 (seq 의 crag_rule generation, stage 시간)
  [측정]  measure 로 새로 잰 generation latency (부하 없는 sequential, 1 stream)
  [추정]  seq 호출 기록으로 맞춘 선형 모델 wall ~ a + b*prompt_tokens + c*out_tokens
  [재구성] seq 의 비-generation 시간 [관측] + 조건별 generation [측정 또는 추정].
          부하·batching·대기가 없는 sequential 값이며 live E2E 측정이 아니다.

    # 1) 분석만 (지금 바로, GPU 불필요): 측정 없이 [추정] 으로
    uv run python experiments/crag_system/evidence_cost.py report --evf <evf run> --name EVC-SV1
    # 2) 서버: generation latency 측정 (~수십 분)
    uv run python experiments/crag_system/evidence_cost.py measure --evf <evf run> \
        --exp-id PRE-EVC-SV1-SEQ --prefix-cache reset
    # 3) 측정 반영 보고서
    uv run python experiments/crag_system/evidence_cost.py report --evf <evf run> \
        --measured results/crag_system/pre/PRE-EVC-SV1-SEQ/<ts> --name EVC-SV1

결과: measure -> results/crag_system/pre/<exp_id>/<ts>/ (run_meta.json, units.jsonl,
      measurements.jsonl). report -> results/crag_system/analysis/evidence_cost/<name>/<ts>/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence_filter_ablation as efa  # noqa: E402  (재구성 규칙·로더 재사용, 읽기만)

from rag.llm.engine import LLMRequest  # noqa: E402
from rag.pipelines import prompts  # noqa: E402
from rag.retrieval.web_cache import read_jsonl  # noqa: E402
from rag.telemetry.run_meta import create_run_dir, git_info, write_new  # noqa: E402
from rag.types import Document  # noqa: E402

ROOT = efa.ROOT
BASE = "crag_rule"
ACTIONS = efa.ACTIONS
NON_GEN = ("retrieval", "grading", "rewrite", "web_search")
GEN_PARAMS = ("temperature", "max_tokens", "seed", "stream")
N_BOOT = 2000


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def pct(x, q):
    x = [v for v in x if v is not None]
    return float(np.percentile(x, q)) if x else None


def mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def dist(prefix: str, x) -> dict:
    return {f"{prefix}_mean": mean(x), f"{prefix}_p50": pct(x, 50), f"{prefix}_p95": pct(x, 95),
            f"{prefix}_p99": pct(x, 99)}  # fmt: skip


def ids_hash(ids) -> str:
    return hashlib.sha1("\x1f".join(ids).encode("utf-8")).hexdigest()[:12]


# ================================================================ measure
def build_units(evf: Path, conditions: list[str], docs_path=None, web_cache=()):
    """evf run -> 측정 단위 (질문, 고유 context). evf 기록과 다르게 재구성되면 그 질문 제외."""
    meta = read_json(evf / "run_meta.json")
    ref = meta["ref"]
    items, _ = efa.load_items(Path(ref["ref"]), ref["split"], docs_path or ref.get("docs_path"),
                              tuple(web_cache) or tuple(ref.get("web_sources") or ()))  # fmt: skip
    gens = {(g["query_id"], g["condition"]): g for g in read_jsonl(evf / "generations.jsonl")}
    units, excluded = [], []
    for it in items:
        if it["problems"]:
            excluded.append({"query_id": it["query_id"], "reason": "; ".join(it["problems"])})
            continue
        by_ids: dict[tuple, dict] = {}
        bad = None
        for cond in conditions:
            g = gens.get((it["query_id"], cond))
            if g is None or g.get("error"):
                bad = f"evf 생성 없음/오류 ({cond})"
                break
            ctx = efa.build_context(cond, it["action"], it["docs"], it["scores"], it["web"],
                                    ref["upper"], ref["lower"])  # fmt: skip
            ids = tuple(d.id for d in ctx)
            if list(ids) != list(g["ctx_ids"]):
                bad = f"context 재구성이 evf 기록과 다름 ({cond})"
                break
            if ids not in by_ids:
                prompt = prompts.GENERATE.format(context=prompts.format_context(ctx),
                                                 question=it["question"])  # fmt: skip
                by_ids[ids] = {
                    "unit_id": f"{it['query_id']}|{ids_hash(ids)}", "query_id": it["query_id"],
                    "request_index": it["request_index"], "action": it["action"],
                    "ctx_ids": list(ids), "n_ctx": len(ids), "conditions": [], "prompt": prompt,
                    "evf_answer": g.get("answer"), "evf_prompt_tokens": g.get("prompt_tokens"),
                    "evf_out_tokens": g.get("out_tokens"),
                }  # fmt: skip
            by_ids[ids]["conditions"].append(cond)
        if bad:
            excluded.append({"query_id": it["query_id"], "reason": bad})
        else:
            units.extend(by_ids.values())
    return units, excluded, meta


def check_engine(engine, evf_meta: dict, allow: bool) -> list[str]:
    """생성 설정이 evf 와 같은지. 다르면 (allow 아니면) 중단."""
    now = engine.describe() if hasattr(engine, "describe") else {}
    prev = evf_meta.get("engine") or {}
    if not isinstance(now.get("params"), dict) or not isinstance(prev.get("params"), dict):
        return ["엔진 생성 설정을 비교할 수 없음 (describe 에 params 없음)"]
    diffs = [f"{k}: evf={prev['params'].get(k)!r} now={now['params'].get(k)!r}"
             for k in GEN_PARAMS if prev["params"].get(k) != now["params"].get(k)]  # fmt: skip
    if prev.get("model") != now.get("model"):
        diffs.append(f"model: evf={prev.get('model')!r} now={now.get('model')!r}")
    if not now["params"].get("stream"):
        diffs.append("stream=false 이면 TTFT 를 잴 수 없음")
    if diffs and not allow:
        raise SystemExit("생성 설정이 evf run 과 다름 (--allow-mismatch 로 무시):\n  " +
                         "\n  ".join(diffs))  # fmt: skip
    return diffs


def prefix_hits(engine) -> float | None:
    """vLLM /metrics 의 prefix cache hit counter 합 (이름은 버전마다 다를 수 있음)."""
    fn = getattr(engine, "metrics_text", None)
    if fn is None:
        return None
    try:
        from rag.llm.vllm_engine import parse_prometheus

        m = parse_prometheus(fn())
    except Exception:  # noqa: BLE001 - 진단용, 실패해도 측정은 계속
        return None
    vals = [v for k, v in m.items() if re.search(r"prefix_cache_hits(_total)?(\{|$)", k)]
    return float(sum(vals)) if vals else None


def schedule(units: list[dict], repeats: int, seed: int) -> list[tuple[int, int, dict]]:
    """(repeat, order, unit). repeat 마다 다른 seed 로 전체 섞기."""
    out = []
    for r in range(repeats):
        perm = np.random.default_rng(seed + r).permutation(len(units))
        out += [(r, o, units[i]) for o, i in enumerate(perm)]
    return out


def measure(evf: Path, engine, out: Path, opts: dict, done: set | None = None) -> Path:
    units, _, _ = build_units(evf, opts["conditions"], opts.get("docs_path"),
                              opts.get("web_cache") or ())  # fmt: skip
    plan = schedule(units, opts["repeats"], opts["seed"])
    done = done or set()
    path = out / "measurements.jsonl"
    reset = opts["prefix_cache"] == "reset"
    with open(path, "a", encoding="utf-8") as f:
        if not done:  # warmup: 측정 질문과 무관한 고정 prompt (prefix 를 공유하지 않음)
            warm = prompts.GENERATE.format(
                context=prompts.format_context([Document(id="warmup", text="warmup text " * 150)]),
                question="Which word is repeated in the context?",
            )
            for i in range(opts["warmup"]):
                r = engine.complete(LLMRequest(prompt=warm, kind="generate"))
                f.write(json.dumps({"warmup": True, "i": i, "wall": r.wall, "ttft": r.ttft}) + "\n")
        n = 0
        for rep, order, u in plan:
            if (u["unit_id"], rep) in done:
                continue
            row = {"warmup": False, "unit_id": u["unit_id"], "query_id": u["query_id"],
                   "request_index": u["request_index"], "action": u["action"],
                   "conditions": u["conditions"], "n_ctx": u["n_ctx"], "repeat": rep,
                   "order": order, "error": None}  # fmt: skip
            if reset:
                rs = engine.reset_prefix_cache()
                row["cache_reset_ok"] = bool(rs.get("ok"))
                if not rs.get("ok"):
                    raise SystemExit(f"prefix cache 초기화 실패: {rs} (서버 --dev-mode 또는 "
                                     "--prefix-cache none)")  # fmt: skip
            h0 = prefix_hits(engine)
            try:
                r = engine.complete(LLMRequest(prompt=u["prompt"], kind="generate"))
                row.update(
                    wall=r.wall, ttft=r.ttft, prompt_tokens=r.prompt_tokens,
                    out_tokens=r.completion_tokens, finish_reason=r.finish_reason,
                    answer_same_evf=r.text == u["evf_answer"],
                    prompt_tokens_same_evf=r.prompt_tokens == u["evf_prompt_tokens"],
                    out_tokens_same_evf=r.completion_tokens == u["evf_out_tokens"],
                )  # fmt: skip
            except Exception as e:  # noqa: BLE001 기록하고 계속 (재시도 없음)
                row["error"] = f"{type(e).__name__}: {e}"[:500]
            h1 = prefix_hits(engine)
            row["prefix_hit_tokens"] = None if h0 is None or h1 is None else h1 - h0
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(plan) - len(done)}", flush=True)
    return path


# ================================================================ report
def load_seq(seq: Path) -> tuple[dict[str, dict], list[dict]]:
    """seq run -> query_id 별 [관측] 시간·호출 요약."""
    calls = defaultdict(list)
    for c in read_jsonl(seq / "calls.jsonl"):
        if not c.get("warmup"):
            calls[c["request_index"]].append(c)
    out = {}
    for r in read_jsonl(seq / "queries.jsonl"):
        if r.get("warmup") or r.get("error"):
            continue
        cs = calls.get(r["request_index"], [])
        gen = [c for c in cs if c["kind"] == "generate" and not c.get("error")]
        other = [c for c in cs if c["kind"] != "generate"]
        st = r.get("stage_latencies") or {}
        out[r["query_id"]] = {
            "request_index": r["request_index"], "action": r.get("corrective_action"),
            "latency": r["latency"], "stage": st,
            "non_gen_time": r["latency"] - st.get("generation", 0.0),
            "gen_wall": gen[0]["wall"] if len(gen) == 1 else None,
            "gen_ttft": gen[0].get("ttft") if len(gen) == 1 else None,
            "gen_prompt_tokens": gen[0].get("prompt_tokens") if len(gen) == 1 else None,
            "n_gen_calls": len(gen),
            "other_wall": sum(c["wall"] for c in other),
            "other_tokens": sum((c.get("prompt_tokens") or 0) + (c.get("out_tokens") or 0)
                                for c in other),
            "n_other_calls": len(other), "n_errors": sum(bool(c.get("error")) for c in cs),
        }  # fmt: skip
    return out, [c for cs in calls.values() for c in cs]


def fit_latency_model(calls: list[dict]) -> dict:
    """seq 호출 기록 -> wall ~ a + b*pt + c*ot, ttft ~ a + b*pt (OLS). [추정] 용."""
    ok = [c for c in calls if not c.get("error") and c.get("prompt_tokens") is not None
          and c.get("out_tokens") is not None]  # fmt: skip
    res = {"n_calls": len(ok)}
    if len(ok) < 5:
        return res
    pt = np.array([c["prompt_tokens"] for c in ok], float)
    ot = np.array([c["out_tokens"] for c in ok], float)
    y = np.array([c["wall"] for c in ok], float)
    X = np.column_stack([np.ones_like(pt), pt, ot])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    gen = np.array([c["kind"] == "generate" for c in ok])
    res["wall"] = {
        "a": float(coef[0]), "b_per_prompt_tok": float(coef[1]), "c_per_out_tok": float(coef[2]),
        "r2": float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)),
        "mae_generate": float(np.abs(y - pred)[gen].mean()) if gen.any() else None,
        "bias_generate": float((pred - y)[gen].mean()) if gen.any() else None,
    }  # fmt: skip
    res["prompt_tokens_range"] = {"all_max": float(pt.max()),
                                  "generate_max": float(pt[gen].max()) if gen.any() else None,
                                  "generate_p95": float(np.percentile(pt[gen], 95)) if gen.any()
                                  else None}  # fmt: skip
    t = [(c["prompt_tokens"], c["ttft"]) for c in ok if c.get("ttft") is not None]
    if len(t) >= 5:
        tp = np.array(t, float)
        Xt = np.column_stack([np.ones(len(tp)), tp[:, 0]])
        ct, *_ = np.linalg.lstsq(Xt, tp[:, 1], rcond=None)
        res["ttft"] = {"a": float(ct[0]), "b_per_prompt_tok": float(ct[1])}
    return res


def predict(model: dict, pt, ot) -> tuple[float | None, float | None]:
    if pt is None or ot is None or "wall" not in model:
        return None, None
    w = model["wall"]
    wall = w["a"] + w["b_per_prompt_tok"] * pt + w["c_per_out_tok"] * ot
    t = model.get("ttft")
    return wall, (t["a"] + t["b_per_prompt_tok"] * pt) if t else None


def load_measured(run: Path) -> tuple[dict, list[dict], dict]:
    """measure run -> (query_id, condition) -> 측정값 (repeat 중앙값), 원시 행, 메타."""
    rows = [r for r in read_jsonl(run / "measurements.jsonl") if not r.get("warmup")]
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r["unit_id"]].append(r)
    out = {}
    for rs in by_unit.values():
        ok = [r for r in rs if not r.get("error")]
        rec = {
            "wall": float(np.median([r["wall"] for r in ok])) if ok else None,
            "ttft": pct([r.get("ttft") for r in ok], 50),
            "prompt_tokens": ok[0].get("prompt_tokens") if ok else None,
            "out_tokens": ok[0].get("out_tokens") if ok else None,
            "n_rep": len(ok), "n_err": len(rs) - len(ok),
            "cv": float(np.std([r["wall"] for r in ok]) / np.mean([r["wall"] for r in ok]))
            if len(ok) > 1 else None,
            "prefix_hit_tokens": mean([r.get("prefix_hit_tokens") for r in ok]),
            "answer_same_evf": mean([r.get("answer_same_evf") for r in ok]),
            "unit_id": rs[0]["unit_id"],
        }  # fmt: skip
        for c in rs[0]["conditions"]:
            out[(rs[0]["query_id"], c)] = rec
    return out, rows, read_json(run / "run_meta.json")


def boot_mean_diff(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """mean(y - x) 의 bootstrap 95% CI (질문 복원 추출)."""
    idx = np.random.default_rng(seed).integers(0, len(x), (N_BOOT, len(x)))
    d = (y[idx] - x[idx]).mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_p95_diff(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    idx = np.random.default_rng(seed).integers(0, len(x), (N_BOOT, len(x)))
    d = np.percentile(y[idx], 95, axis=1) - np.percentile(x[idx], 95, axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ci2(prefix: str, ci: tuple[float, float]) -> dict:
    return {f"{prefix}_ci_lo": ci[0], f"{prefix}_ci_hi": ci[1]}


def report(evf: Path, seq: Path | None, measured: Path | None, n_boot: int = 5000) -> dict:
    evf_meta = read_json(evf / "run_meta.json")
    seq = seq or Path(evf_meta["ref"]["ref"])
    gens = read_jsonl(evf / "generations.jsonl")
    by = defaultdict(dict)
    for g in gens:
        by[g["condition"]][g["query_id"]] = g
    conds = [c for c in efa.CONDITIONS if c in by]
    if BASE not in conds:
        raise SystemExit(f"evf run 에 {BASE} 가 없음")
    seq_q, seq_calls = load_seq(seq)
    model = fit_latency_model(seq_calls)
    meas, meas_rows, meas_meta = load_measured(measured) if measured else ({}, [], None)
    if measured:
        conds = [c for c in conds if c in (meas_meta.get("conditions") or [])]
    notes = []
    qids = set.intersection(*[{q for q, g in by[c].items() if not g.get("error")} for c in conds])
    n_evf_common = len(qids)
    qids &= set(seq_q)
    if measured:
        qids = {q for q in qids if all(meas.get((q, c), {}).get("wall") is not None
                                       for c in conds)}  # fmt: skip
    qids = sorted(qids)
    notes.append(f"evf 공통 질문 {n_evf_common} -> seq 매칭"
                 f"{' + 모든 조건 측정 성공' if measured else ''} {len(qids)}")  # fmt: skip
    act_mismatch = sum(seq_q[q]["action"] != by[BASE][q]["action"] for q in qids)
    if act_mismatch:
        notes.append(f"주의: seq 와 evf 의 action 이 다른 질문 {act_mismatch}")
    src = "측정" if measured else "추정"

    per = defaultdict(dict)  # cond -> qid -> row
    for c in conds:
        for q in qids:
            g, s = by[c][q], seq_q[q]
            est_w, est_t = predict(model, g.get("prompt_tokens"), g.get("out_tokens"))
            m = meas.get((q, c)) or {}
            gw = m.get("wall") if measured else est_w
            gt = m.get("ttft") if measured else est_t
            per[c][q] = {
                "query_id": q, "condition": c, "action": g["action"], "acc": g.get("acc"),
                "em": g.get("em"), "f1": g.get("f1"), "prompt_tokens": g.get("prompt_tokens"),
                "out_tokens": g.get("out_tokens"), "n_ctx": g.get("n_ctx"),
                "gen_est": est_w, "ttft_est": est_t, "gen_meas": m.get("wall"),
                "ttft_meas": m.get("ttft"), "gen": gw, "ttft": gt,
                "gen_obs": s["gen_wall"] if c == BASE else None,
                "ttft_obs": s["gen_ttft"] if c == BASE else None,
                "e2e_recon": None if gw is None else s["non_gen_time"] + gw,
                "llm_busy": None if gw is None else s["other_wall"] + gw,
                "llm_tokens": s["other_tokens"] + (g.get("prompt_tokens") or 0)
                + (g.get("out_tokens") or 0),
                "prefix_hit_tokens": m.get("prefix_hit_tokens"),
                "answer_same_evf": m.get("answer_same_evf"), "unit_id": m.get("unit_id"),
            }  # fmt: skip
    # crag_rule 은 seq 에 같은 prompt 의 generation 이 있다 -> [관측] 대조
    same_prompt = [seq_q[q]["gen_prompt_tokens"] == by[BASE][q].get("prompt_tokens") for q in qids]
    obs = {
        "n": len(qids), "prompt_tokens_match_share": mean(same_prompt),
        **dist("gen_obs", [per[BASE][q]["gen_obs"] for q in qids]),
        **dist("ttft_obs", [per[BASE][q]["ttft_obs"] for q in qids]),
        **dist("e2e_obs", [seq_q[q]["latency"] for q in qids]),
        "stage_mean": {s: mean([seq_q[q]["stage"].get(s) for q in qids]) for s in (*NON_GEN,
                                                                                  "generation")},
    }  # fmt: skip

    summary = []
    for c in conds:
        rs = list(per[c].values())
        units = {r["unit_id"] for r in rs if r["unit_id"]}
        summary.append({
            "condition": c, "n": len(rs), "acc": mean([r["acc"] for r in rs]),
            "em": mean([r["em"] for r in rs]), "f1": mean([r["f1"] for r in rs]),
            "prompt_tokens": mean([r["prompt_tokens"] for r in rs]),
            "out_tokens": mean([r["out_tokens"] for r in rs]),
            "gen_calls_per_q": 1.0, "unique_gen_measured": len(units) if measured else None,
            "evf_errors": sum(1 for g in by[c].values() if g.get("error")),
            "meas_errors": sum(m["n_err"] for (q, cc), m in meas.items() if cc == c) if measured
            else None,
            "source": src, **dist("gen", [r["gen"] for r in rs]),
            **dist("ttft", [r["ttft"] for r in rs]),
            **dist("e2e_recon", [r["e2e_recon"] for r in rs]),
            "llm_busy_mean": mean([r["llm_busy"] for r in rs]),
            "llm_tokens_mean": mean([r["llm_tokens"] for r in rs]),
            "gen_share_of_e2e": (mean([r["gen"] for r in rs]) or 0)
            / (mean([r["e2e_recon"] for r in rs]) or float("nan")),
            "prefix_hit_tokens": mean([r["prefix_hit_tokens"] for r in rs]),
            "answer_same_evf": mean([r["answer_same_evf"] for r in rs]),
        })  # fmt: skip

    by_action = []
    for c in conds:
        for a in ACTIONS:
            rs = [r for r in per[c].values() if r["action"] == a]
            if rs:
                by_action.append({
                    "condition": c, "action": a, "n": len(rs),
                    "acc": mean([r["acc"] for r in rs]),
                    "prompt_tokens": mean([r["prompt_tokens"] for r in rs]),
                    "gen_mean": mean([r["gen"] for r in rs]), "gen_p95": pct([r["gen"] for r in rs],
                                                                             95),
                    "ttft_mean": mean([r["ttft"] for r in rs]),
                    "e2e_recon_mean": mean([r["e2e_recon"] for r in rs]),
                    "e2e_recon_p95": pct([r["e2e_recon"] for r in rs], 95),
                })  # fmt: skip

    paired = []
    pairs = [(BASE, c) for c in conds if c != BASE]
    if "vanilla_ctx" in conds:
        pairs += [("vanilla_ctx", c) for c in ("crag_rule", "keep_all") if c in conds]
    for b, t in pairs:
        for sub in ("all", *ACTIONS):
            ks = [q for q in qids if sub == "all" or per[BASE][q]["action"] == sub]
            if len(ks) < 2:
                continue
            col = lambda c, k, ks=ks: np.array([per[c][q][k] for q in ks], float)  # noqa: E731
            pa = efa.paired_acc(col(b, "acc"), col(t, "acc"), n_boot)
            ge, gt_ = col(b, "gen"), col(t, "gen")
            eb, et = col(b, "e2e_recon"), col(t, "e2e_recon")
            d_tok = float((col(t, "prompt_tokens") - col(b, "prompt_tokens")).mean())
            d_e2e = float(et.mean() - eb.mean())
            row = {
                "base": b, "target": t, "subset": sub, "n": len(ks),
                "acc_diff": pa["diff"], "acc_ci_lo": pa["ci_lo"], "acc_ci_hi": pa["ci_hi"],
                "mcnemar_p": pa["mcnemar_p"],
                "gen_diff": float(gt_.mean() - ge.mean()),
                **ci2("gen", boot_mean_diff(ge, gt_)),
                "e2e_diff": d_e2e, **ci2("e2e", boot_mean_diff(eb, et)),
                "e2e_p95_diff": float(np.percentile(et, 95) - np.percentile(eb, 95)),
                **ci2("e2e_p95", boot_p95_diff(eb, et)),
                "prompt_tok_diff": d_tok,
                "acc_pp_per_s": pa["diff"] * 100 / d_e2e if abs(d_e2e) > 1e-9 else None,
                "acc_pp_per_1k_tok": pa["diff"] * 100 / (d_tok / 1000) if abs(d_tok) > 1e-9
                else None,
                "source": src,
            }  # fmt: skip
            paired.append(row)

    validation = {"crag_rule_seq_observed": obs, "model": model}
    if "prompt_tokens_range" in model:
        lim = model["prompt_tokens_range"]["all_max"]
        validation["extrapolated_share"] = {
            c: mean([(r["prompt_tokens"] or 0) > lim for r in per[c].values()]) for c in conds
        }
    if measured:
        validation["est_vs_meas"] = [
            {"condition": c,
             "mae": mean([abs(r["gen_est"] - r["gen_meas"]) for r in per[c].values()
                          if r["gen_est"] is not None]),
             "bias": mean([r["gen_est"] - r["gen_meas"] for r in per[c].values()
                           if r["gen_est"] is not None])}
            for c in conds
        ]  # fmt: skip
        b = [
            (per[BASE][q]["gen_meas"], per[BASE][q]["gen_obs"])
            for q in qids
            if per[BASE][q]["gen_obs"] is not None
        ]
        validation["crag_rule_meas_vs_seq_obs"] = {
            "n": len(b), "mean_meas": mean([x for x, _ in b]), "mean_obs": mean([y for _, y in b]),
            "mae": mean([abs(x - y) for x, y in b]),
        }  # fmt: skip
        ok = [r for r in meas_rows if not r.get("error")]
        if ok:
            o = np.array([r["order"] for r in ok], float)
            w = np.array([r["wall"] for r in ok], float)
            # 순서 효과: 같은 unit 평균을 뺀 잔차와 측정 순서의 상관 (drift / warm-up 잔여)
            um = defaultdict(list)
            for r in ok:
                um[r["unit_id"]].append(r["wall"])
            pt = np.array([r.get("prompt_tokens") or 0 for r in ok], float)
            resid = w - np.polyval(np.polyfit(pt, w, 1), pt) if len(ok) > 2 else w * 0
            validation["order_effect"] = {
                "corr_order_resid": float(np.corrcoef(o, resid)[0, 1]) if len(ok) > 2 else None,
                "first_half_mean": float(w[o < np.median(o)].mean()) if len(ok) > 2 else None,
                "second_half_mean": float(w[o >= np.median(o)].mean()) if len(ok) > 2 else None,
                "repeat_cv_mean": mean([float(np.std(v) / np.mean(v)) for v in um.values()
                                        if len(v) > 1]),
            }  # fmt: skip
        validation["measure_settings"] = {k: meas_meta.get(k) for k in
                                          ("prefix_cache", "repeats", "seed", "warmup",
                                           "engine_mismatch", "engine")}  # fmt: skip

    # live 후보 (제안): crag_rule 대비 정확도 CI 하한 > 0 인 조건 중 (acc, e2e) Pareto.
    # vanilla_ctx 는 live runner 가 지원하지 않음 (grading·web 을 하고 쓰지 않는 조건)
    cand = [r["target"] for r in paired if r["base"] == BASE and r["subset"] == "all"
            and r["acc_ci_lo"] > 0 and r["target"] != "vanilla_ctx"]  # fmt: skip
    pts = {
        s["condition"]: (s["acc"], s["e2e_recon_mean"])
        for s in summary
        if s["condition"] in (BASE, *cand) and s["e2e_recon_mean"] is not None
    }
    pareto = [
        c
        for c, (a, e) in pts.items()
        if not any(a2 >= a and e2 <= e and (a2, e2) != (a, e) for a2, e2 in pts.values())
    ]
    return {
        "evf": str(evf), "seq": str(seq), "measured": str(measured) if measured else None,
        "source": src, "n": len(qids), "conditions": conds, "notes": notes,
        "summary": summary, "by_action": by_action, "paired": paired, "validation": validation,
        "live_candidates": {"acc_significant": cand, "pareto": pareto},
        "per_query": [r for c in conds for r in per[c].values()],
    }  # fmt: skip


def report_md(res: dict, name: str) -> str:
    fmt, tbl = efa.fmt, efa.md_table
    s = res["source"]
    v = res["validation"]
    o = v["crag_rule_seq_observed"]
    L = [
        f"# Evidence preservation 비용 ({name})", "",
        f"- evf (정확도·토큰): `{res['evf']}`", f"- seq (stage 시간 [관측]): `{res['seq']}`",
        f"- generation latency 출처: **[{s}]**"
        + (f" (`{res['measured']}`)" if res["measured"] else
           " (seq 호출 기록으로 맞춘 선형 모델. measure 실행 후 [측정] 으로 대체)"),
        f"- 질문 수 {res['n']} (모든 조건 공통). " + " / ".join(res["notes"]),
        "- E2E 는 [재구성] = seq 의 비-generation 시간 [관측] + generation [" + s + "]. "
        "sequential·무부하 값이며 live E2E 측정이 아니다",
        "- 정확도·EM·F1·토큰은 evf 의 generation 결과 (grading·action·검색·web 은 모든 조건 동일)",
        "", "## 조건별 요약", "",
    ]  # fmt: skip
    L += tbl(res["summary"], ["condition", "n", "acc", "em", "f1", "prompt_tokens", "out_tokens",
                              "gen_mean", "gen_p50", "gen_p95", "gen_p99", "ttft_mean",
                              "ttft_p95", "e2e_recon_mean", "e2e_recon_p95", "e2e_recon_p99",
                              "llm_busy_mean", "llm_tokens_mean", "gen_share_of_e2e",
                              "evf_errors", "meas_errors", "answer_same_evf",
                              "prefix_hit_tokens"])  # fmt: skip
    L += ["", f"- gen / ttft: generation 1 회 latency [{s}]. e2e_recon: [재구성]. llm_busy: "
          "질문당 LLM 호출 wall 합 (sequential 1 stream 에서의 GPU 점유 시간 근사).",
          "- llm_tokens: 질문당 모든 LLM 호출 (grade·rewrite [관측] + generation) prompt+out 토큰. "
          "generation 호출 수는 모든 조건에서 질문당 1",
          "- answer_same_evf: [측정] 답변 = evf 답변 비율 (생성 재현성). prefix_hit_tokens: 호출당 "
          "vLLM prefix cache hit 토큰 (reset 모드면 0 이어야 함)", ""]  # fmt: skip
    L += ["## crag_rule 기준값 [관측] (seq run 그대로)", "",
          f"- generation: mean {fmt(o['gen_obs_mean'])} / p50 {fmt(o['gen_obs_p50'])} / "
          f"p95 {fmt(o['gen_obs_p95'])} / p99 {fmt(o['gen_obs_p99'])} s, TTFT mean "
          f"{fmt(o['ttft_obs_mean'])} s", f"- E2E: mean {fmt(o['e2e_obs_mean'])} / p95 "
          f"{fmt(o['e2e_obs_p95'])} / p99 {fmt(o['e2e_obs_p99'])} s",
          "- stage 평균: " + ", ".join(f"{k} {fmt(x)}" for k, x in o["stage_mean"].items()),
          f"- seq generation prompt 토큰 = evf crag_rule prompt 토큰 비율: "
          f"{fmt(o['prompt_tokens_match_share'])} (1 이면 같은 prompt)", ""]  # fmt: skip
    L += [
        "## paired 차이 (target - base). 정확도: bootstrap 95% CI + McNemar, "
        f"latency: [{s}] bootstrap 95% CI",
        "",
    ]
    L += tbl(res["paired"], ["base", "target", "subset", "n", "acc_diff", "acc_ci_lo", "acc_ci_hi",
                             "mcnemar_p", "gen_diff", "gen_ci_lo", "gen_ci_hi", "e2e_diff",
                             "e2e_ci_lo", "e2e_ci_hi", "e2e_p95_diff", "e2e_p95_ci_lo",
                             "e2e_p95_ci_hi", "prompt_tok_diff", "acc_pp_per_s",
                             "acc_pp_per_1k_tok"])  # fmt: skip
    L += ["", "- acc_pp_per_s: 평균 E2E 1 초 증가당 정확도 증가 (%p). "
          "acc_pp_per_1k_tok: generation "
          "prompt 1k 토큰 증가당 (%p). subset 은 기준 run 의 action (각 단일 규칙은 해당 action "
          "질문에만 영향)", ""]  # fmt: skip
    L += ["## action 별", ""]
    L += tbl(
        res["by_action"],
        [
            "condition",
            "action",
            "n",
            "acc",
            "prompt_tokens",
            "gen_mean",
            "gen_p95",
            "ttft_mean",
            "e2e_recon_mean",
            "e2e_recon_p95",
        ],
    )
    m = v["model"]
    L += ["", "## latency 모델 / 검증", ""]
    if "wall" in m:
        w = m["wall"]
        L += [
            f"- [추정] 모델 (seq 호출 {m['n_calls']}): wall = {w['a']:.4f} + "
            f"{w['b_per_prompt_tok'] * 1000:.4f} ms x prompt_tok + {w['c_per_out_tok'] * 1000:.3f}"
            f" ms x out_tok, R2 {w['r2']:.3f}, generation MAE {fmt(w['mae_generate'])} s, "
            f"bias {fmt(w['bias_generate'])} s",
            "- 학습 prompt 토큰 범위: generation max "
            f"{fmt(m['prompt_tokens_range']['generate_max'])}"
            f", 전체 max {fmt(m['prompt_tokens_range']['all_max'])}. 범위 밖 (외삽) 비율: "
            + ", ".join(f"{c} {fmt(x)}" for c, x in v.get("extrapolated_share", {}).items()),
        ]
    else:
        L += ["- seq 호출 기록이 부족해 [추정] 모델을 만들 수 없음"]
    if "est_vs_meas" in v:
        L += ["", "### [추정] vs [측정]", ""]
        L += tbl(v["est_vs_meas"], ["condition", "mae", "bias"])
        cm = v["crag_rule_meas_vs_seq_obs"]
        L += ["", f"- crag_rule [측정] vs seq [관측] (같은 prompt): n {cm['n']}, mean "
              f"{fmt(cm['mean_meas'])} vs {fmt(cm['mean_obs'])} s, MAE {fmt(cm['mae'])} s "
              "(측정 환경 재현성. 차이가 크면 조건 간 차이 해석에 주의)"]  # fmt: skip
        oe = v.get("order_effect") or {}
        L += [
            "- 측정 순서 효과: corr(순서, 토큰 보정 잔차) "
            f"{fmt(oe.get('corr_order_resid'))}, 앞/뒤 "
            f"절반 평균 {fmt(oe.get('first_half_mean'))} / {fmt(oe.get('second_half_mean'))} s, "
            f"repeat 간 CV {fmt(oe.get('repeat_cv_mean'))}",
            f"- 측정 설정: {json.dumps(v['measure_settings'], ensure_ascii=False, default=str)}",
        ]
    lc = res["live_candidates"]
    L += ["", "## Live 비교 후보 (제안, 최종 선택은 사용자)", "",
          f"- crag_rule 대비 정확도 CI 하한 > 0: {lc['acc_significant'] or '없음'}",
          f"- 그 중 (정확도, 평균 E2E[재구성]) Pareto: {lc['pareto']}",
          "- live 에서는 crag_rule 과 함께 실행 (run_evidence_cost.sh live)", ""]  # fmt: skip
    return "\n".join(L)


def out_dir(root: Path, name: str) -> Path:
    base = root / "evidence_cost" / name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out, n = base, 1
    while out.exists():  # 덮어쓰지 않음
        n += 1
        out = base.with_name(f"{base.name}_{n}")
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_report(res: dict, name: str, root: Path, quiet: bool, md_fn=report_md) -> Path:
    out = out_dir(root, name)
    (out / "analysis.json").write_text(json.dumps({k: v for k, v in res.items()
                                                   if k != "per_query"}, indent=2,
                                                  ensure_ascii=False, default=str),
                                       encoding="utf-8")  # fmt: skip
    for k in ("summary", "by_action", "paired", "per_query", "per_run", "agg"):
        if k in res:
            write_csv(out / f"{k}.csv", res[k])
    md = md_fn(res, name)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not quiet:
        print(md)
    print(f"saved -> {out}")
    return out


# ================================================================ live report
LIVE_DIR = re.compile(r"^rho(?P<rho>[\d.]+)_s(?P<seed>\d+)_(?P<policy>[a-z_]+)_K(?P<k>\d+)$")


def _cfg_policy(d: Path) -> str | None:
    """run 폴더 config.yaml 의 evidence_policy (폴더 이름의 정책과 대조용)."""
    import yaml

    p = d / "config.yaml"
    return (
        (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("evidence_policy")
        if p.exists()
        else None
    )


def live_report(root: Path, seq: Path | None, ref_policy: str = BASE) -> dict:
    """run_evidence_live.py run 들 (rho*_s*_<policy>_K*) -> 정책별 정확도·지연·처리량·GPU."""
    from analyze_sched_validation import gpu_util, load_run, paired

    seq_q = load_seq(seq)[0] if seq else {}
    runs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = LIVE_DIR.match(d.name)
        if m and (d / "summary.json").exists():
            runs[(m["rho"], int(m["seed"]), m["policy"], int(m["k"]))] = (d, load_run(d))
    per_run = []
    for (rho, seed, pol, k), (d, r) in runs.items():
        rows = r["rows"]
        su = r["summary"]
        g = gpu_util(Path(str(d) + ".gpu.csv"))
        agree = [x["corrective_action"] == seq_q[x["query_id"]]["action"] for x in rows
                 if x["query_id"] in seq_q]  # fmt: skip
        calls = [c for cs in r["calls"].values() for c in cs]
        gen = [c for c in calls if c["kind"] == "generate" and not c.get("error")]
        per_run.append({
            "rho": rho, "seed": seed, "policy": pol, "K": k, "n": len(rows),
            "errors": r["n_errors"], "acc": mean([x["acc"] for x in rows]),
            "em": mean([x.get("em") for x in rows]), "f1": mean([x.get("f1") for x in rows]),
            **dist("lat", [x["latency"] for x in rows]),
            "throughput_qps": su.get("throughput_rps"),
            "gpu_util": None if not g else mean([x["mean"] for x in g.values()]),
            "gen_prompt_tokens": mean([c.get("prompt_tokens") for c in gen]),
            "gen_wall_mean": mean([c["wall"] for c in gen]),
            "gen_ttft_mean": mean([c.get("ttft") for c in gen]),
            "llm_tokens_per_q": mean([(x["calls_summary"].get("prompt_tokens") or 0)
                                      + (x["calls_summary"].get("out_tokens") or 0) for x in rows]),
            "correction_rate": mean([x.get("corrective_action") != "correct" for x in rows]),
            "action_agree_seq": mean(agree),
            "policy_cfg": _cfg_policy(d),
        })  # fmt: skip
    pairs = []
    for (rho, seed, pol, k), (_, r) in runs.items():
        ref = runs.get((rho, seed, ref_policy, k))
        if pol == ref_policy or ref is None:
            continue
        a = {(x["request_index"], x["query_id"]): x["acc"] for x in r["rows"]}
        b = {(x["request_index"], x["query_id"]): x["acc"] for x in ref[1]["rows"]}
        ks = sorted(set(a) & set(b))
        for p in paired(pol, r, ref_policy, ref[1]):
            if p["subset"] == "all":
                pairs.append({"rho": rho, "seed": seed, "K": k, "policy": pol,
                              "acc_diff": mean([a[x] - b[x] for x in ks]),
                              **{x: p[x] for x in ("n_pairs", "mean_diff", "mean_ci_lo",
                                                   "mean_ci_hi", "p95_diff", "p95_ci_lo",
                                                   "p95_ci_hi", "path_mismatch")}})  # fmt: skip
    agg = []
    keys = sorted({(r["rho"], r["policy"], r["K"]) for r in per_run})
    for rho, pol, k in keys:
        rs = [r for r in per_run if (r["rho"], r["policy"], r["K"]) == (rho, pol, k)]
        ps = [p for p in pairs if (p["rho"], p["policy"], p["K"]) == (rho, pol, k)]
        agg.append({
            "rho": rho, "policy": pol, "K": k, "n_seeds": len(rs),
            **{f"{m}": mean([r[m] for r in rs]) for m in ("acc", "lat_mean", "lat_p95", "lat_p99",
                                                          "throughput_qps", "gpu_util",
                                                          "llm_tokens_per_q", "action_agree_seq")},
            "d_acc_vs_ref": mean([p["acc_diff"] for p in ps]),
            "d_lat_mean_vs_ref": mean([p["mean_diff"] for p in ps]),
            "d_lat_p95_vs_ref": mean([p["p95_diff"] for p in ps]),
            "seeds_lat_ci_above_0": sum(p["mean_ci_lo"] > 0 for p in ps) if ps else None,
        })  # fmt: skip
    return {"root": str(root), "seq": str(seq) if seq else None, "ref_policy": ref_policy,
            "per_run": per_run, "paired": pairs, "agg": agg}  # fmt: skip


def live_md(res: dict, name: str) -> str:
    tbl = efa.md_table
    L = [f"# Evidence preservation live ({name})", "", f"- run 폴더: `{res['root']}`",
         f"- 기준 정책: {res['ref_policy']}. 모든 값은 [live 측정]. 정확도는 live 에서 grading 을 "
         "다시 하므로 action 이 seq 와 다를 수 있음 (action_agree_seq)", "",
         "## 정책 x rho (seed 평균)", ""]  # fmt: skip
    L += tbl(res["agg"], ["rho", "policy", "K", "n_seeds", "acc", "d_acc_vs_ref", "lat_mean",
                          "d_lat_mean_vs_ref", "lat_p95", "d_lat_p95_vs_ref", "lat_p99",
                          "throughput_qps", "gpu_util", "llm_tokens_per_q", "action_agree_seq",
                          "seeds_lat_ci_above_0"])  # fmt: skip
    L += ["", "## run 별", ""]
    L += tbl(res["per_run"], ["rho", "seed", "policy", "K", "n", "errors", "acc", "lat_mean",
                              "lat_p95", "lat_p99", "throughput_qps", "gpu_util",
                              "gen_prompt_tokens", "gen_wall_mean", "gen_ttft_mean",
                              "correction_rate", "action_agree_seq"])  # fmt: skip
    L += ["", "## paired (같은 요청, policy - 기준)", ""]
    L += tbl(res["paired"], ["rho", "seed", "K", "policy", "n_pairs", "acc_diff", "mean_diff",
                             "mean_ci_lo", "mean_ci_hi", "p95_diff", "p95_ci_lo", "p95_ci_hi",
                             "path_mismatch"])  # fmt: skip
    return "\n".join(L + [""])


# ================================================================ main
def main(argv=None, engine=None) -> Path:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--evf", required=True, help="evidence_filter_ablation 생성 run")
    m.add_argument("--exp-id", default=None, help="PRE-* (새 run 폴더)")
    m.add_argument("--resume", default=None)
    m.add_argument("--conditions", nargs="+", default=None, choices=list(efa.CONDITIONS),
                   help="기본: evf run 의 조건 전부")  # fmt: skip
    m.add_argument("--prefix-cache", choices=["reset", "none"], default="reset")
    m.add_argument("--repeats", type=int, default=1)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--warmup", type=int, default=5)
    m.add_argument("--docs-path", default=None)
    m.add_argument("--web-cache", nargs="*", default=[])
    m.add_argument("--allow-mismatch", action="store_true")
    m.add_argument("--check-only", action="store_true", help="단위 구성·검증만 (LLM 호출 없음)")
    m.add_argument("--output-dir", default="results")
    r = sub.add_parser("report")
    r.add_argument("--evf", required=True)
    r.add_argument("--seq", default=None, help="기본: evf 의 기준 run")
    r.add_argument("--measured", default=None, help="measure run 폴더 (없으면 [추정])")
    r.add_argument("--n-boot", type=int, default=5000)
    lv = sub.add_parser("live-report")
    lv.add_argument("--live-root", required=True)
    lv.add_argument("--seq", default=None, help="action 일치 비교용 seq run")
    lv.add_argument("--ref-policy", default=BASE)
    for p in (r, lv):
        p.add_argument("--name", default=None)
        p.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
        p.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "report":
        evf = Path(a.evf)
        res = report(evf, Path(a.seq) if a.seq else None,
                     Path(a.measured) if a.measured else None, a.n_boot)  # fmt: skip
        return write_report(res, a.name or evf.parent.name, Path(a.out_root), a.quiet)
    if a.cmd == "live-report":
        root = Path(a.live_root)
        res = live_report(root, Path(a.seq) if a.seq else None, a.ref_policy)
        return write_report(res, a.name or root.name, Path(a.out_root), a.quiet, live_md)

    evf = Path(a.evf)
    evf_meta = read_json(evf / "run_meta.json")
    if a.resume:
        out = Path(a.resume)
        opts = read_json(out / "run_meta.json")["opts"]
    else:
        opts = {"conditions": a.conditions or evf_meta.get("conditions") or list(efa.CONDITIONS),
                "prefix_cache": a.prefix_cache, "repeats": a.repeats, "seed": a.seed,
                "warmup": a.warmup, "docs_path": a.docs_path, "web_cache": a.web_cache}  # fmt: skip
    units, excluded, _ = build_units(evf, opts["conditions"], opts["docs_path"],
                                     opts["web_cache"])  # fmt: skip
    n_q = len({u["query_id"] for u in units})
    n_pairs = sum(len(u["conditions"]) for u in units)
    print(f"측정 단위: 질문 {n_q}, (질문,조건) {n_pairs} -> 고유 generation {len(units)} "
          f"x repeat {opts['repeats']} / 제외 {len(excluded)}")  # fmt: skip
    for e in excluded[:10]:
        print(f"  제외 {e['query_id']}: {e['reason']}")
    if a.check_only:
        return evf
    if engine is None:
        from rag.llm.factory import build_engine

        llm = dict(evf_meta["ref"]["llm"])
        llm["api_key_env"] = efa.ref_config(Path(evf_meta["ref"]["ref"])).get("llm", {}).get(
            "api_key_env", "VLLM_API_KEY")  # fmt: skip
        engine = build_engine(llm)
    mismatch = check_engine(engine, evf_meta, a.allow_mismatch)
    for x in mismatch:
        print(f"경고: {x}")
    if a.resume:
        done = {(x["unit_id"], x["repeat"]) for x in read_jsonl(out / "measurements.jsonl")
                if not x.get("warmup") and not x.get("error")}  # fmt: skip
    else:
        if not a.exp_id or not a.exp_id.startswith("PRE-"):
            raise SystemExit("--exp-id 는 PRE-* (예비·진단 실험)")
        out = create_run_dir(f"crag_system/pre/{a.exp_id}", a.output_dir)
        write_new(out / "run_meta.json", json.dumps(
            {"created": datetime.now().isoformat(), "evf": str(evf), "opts": opts,
             "conditions": opts["conditions"], "prefix_cache": opts["prefix_cache"],
             "repeats": opts["repeats"], "seed": opts["seed"], "warmup": opts["warmup"],
             "n_units": len(units), "n_query_condition": n_pairs, "excluded": excluded,
             "engine_mismatch": mismatch, "git": git_info(),
             "engine": engine.describe() if hasattr(engine, "describe") else None},
            indent=2, ensure_ascii=False, default=str))  # fmt: skip
        with open(out / "units.jsonl", "x", encoding="utf-8") as f:
            for u in units:
                f.write(json.dumps({k: v for k, v in u.items() if k != "prompt"},
                                   ensure_ascii=False) + "\n")  # fmt: skip
        done = set()
    measure(evf, engine, out, opts, done)
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
