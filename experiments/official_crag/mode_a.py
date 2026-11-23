"""MODE A — Official Reproduction (공식 CRAG PopQA 경로를 단계별로 재현).

공식 흐름 (run_crag_inference.sh, CRAG_Inference.py):
  1. evaluator 로 질문당 10문서 점수 -> flag -> correct / ambiguous / incorrect
  2. 판정에 따라 저자들이 미리 만든 지식 파일 (ref/correct, ref/incorrect, ref/ambiguous) 의
     같은 줄 선택
  3. 공식 PopQA prompt 로 generation (temperature 0, top_p 1, max_tokens 100), 후처리
  4. eval.py --metric match (대소문자 구분 substring)

단계 (각 단계가 결과 폴더를 만든다: results/official_crag/mode_a/<step>/<run_id>/):
  score              1 (T5 evaluator, GPU 또는 CPU. vLLM 불필요)
  verify-refinement  공식 internal refinement 를 다시 계산해 ref/correct 와 줄 단위 비교
                     (Knowledge Refinement 재구현이 공식 결과를 재현하는지 확인)
  generate           2-3 (vLLM, --method crag | rag). crag 는 score 결과 폴더가 필요
  evaluate           4 (+ 이 레포의 정규화 accuracy, 판정별 정확도, 선택 문항 subset)

공식과 다를 수밖에 없는 것 (README 차이표): generator (논문: LLaMA2-7B / SelfRAG-LLaMA2-7B),
vLLM 버전 (원본 0.2.5 offline), 외부 지식은 저자 파일 재사용 (GPT-3.5 + serper 재실행 안 함).

    uv run python experiments/official_crag/mode_a.py score --device cuda:0
    uv run python experiments/official_crag/mode_a.py verify-refinement --device cuda:0 --limit 200
    uv run python experiments/official_crag/mode_a.py generate --method crag --scores <score run> \
        --config experiments/official_crag/configs/mode_a_popqa.yaml
    uv run python experiments/official_crag/mode_a.py evaluate --preds <generate run> \
        --answers data/raw/self_rag_eval/popqa_longtail_w_gs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import load_layered_config  # noqa: E402
from rag.crag_official import generation as offgen  # noqa: E402
from rag.crag_official import source  # noqa: E402
from rag.crag_official.evaluator import build_evaluator, weights_fingerprint  # noqa: E402
from rag.crag_official.official_data import (  # noqa: E402
    attach_answers,
    load_official_queries,
    load_ref,
    retrieved_psgs_line,
)
from rag.crag_official.refinement import (  # noqa: E402
    internal_knowledge,
    official_passages_from_line,
)
from rag.eval.metrics import accuracy  # noqa: E402
from rag.telemetry.run_meta import (  # noqa: E402
    JsonlWriter,
    collect_run_metadata,
    create_run_dir,
    write_new,
    write_run_files,
)

DEFAULT_REPO = ROOT / "third_party" / "CRAG"


def official_paths(repo: Path) -> dict[str, Path]:
    d = repo / "data" / "popqa"
    return {
        "test": d / "test_popqa.txt",
        "sources": d / "sources",
        "correct": d / "ref" / "correct",
        "incorrect": d / "ref" / "incorrect",
        "ambiguous": d / "ref" / "ambiguous",
    }


def _run_dir(step: str, run_id: str | None) -> Path:
    return create_run_dir(f"official_crag/mode_a/{step}", "results", run_id)


def _meta(run_dir: Path, cfg: dict, extra: dict, engines=None) -> None:
    extra = {
        "experiment_mode": "A_official_reproduction",
        "official_source": {
            "repo": source.OFFICIAL_REPO_URL,
            "commit": source.OFFICIAL_COMMIT,
            "license": source.OFFICIAL_LICENSE,
            "weights_url": source.EVALUATOR_WEIGHTS_URL,
        },
        **extra,
    }
    write_run_files(run_dir, cfg, collect_run_metadata(cfg, engines, extra))


def _evaluator(a):
    cfg = {"backend": "t5", "model_path": a.weights, "device": a.device, "task": "popqa"}
    return cfg, build_evaluator(cfg)


def cmd_score(a) -> Path:
    paths = official_paths(Path(a.repo))
    queries = load_official_queries(paths["test"], source.OFFICIAL_NDOCS, paths["sources"])
    if a.limit:
        queries = queries[: a.limit]
    ecfg, ev = _evaluator(a)
    run_dir = _run_dir("score", a.run_id)
    cfg = {"experiment": "official_crag", "step": "score", "evaluator": ecfg, "limit": a.limit}
    _meta(
        run_dir,
        cfg,
        {
            "evaluator": ev.describe(),
            "weights": weights_fingerprint(ev.backend.model_path),
            "inputs": {k: str(v) for k, v in paths.items()},
        },
    )
    out = JsonlWriter(run_dir / "scores.jsonl")
    counts: Counter = Counter()
    t0 = time.perf_counter()
    for q in queries:
        v = ev.evaluate(q.question, q.passages)
        counts[v.action] += 1
        out.write(
            {
                "index": q.index,
                "question": q.question,
                "scores": v.scores,
                "flags": v.flags,
                "action": v.action,
                "n_tokens": [p.n_tokens for p in v.pairs],
                "pair_compute_seconds": [p.compute_seconds for p in v.pairs],
                "wall_seconds": v.wall_seconds,
            }
        )
    summary = {
        "n": len(queries),
        "action_counts": dict(counts),
        "wall_seconds": time.perf_counter() - t0,
        "thresholds": {"upper": ev.upper, "lower": ev.lower},
    }
    write_new(run_dir / "summary.json", json.dumps(summary, indent=2))
    return run_dir


def cmd_verify_refinement(a) -> Path:
    paths = official_paths(Path(a.repo))
    queries = load_official_queries(paths["test"], source.OFFICIAL_NDOCS, paths["sources"])
    ref = [x[1:] if x.startswith("#") else x for x in load_ref(paths["correct"])]
    if len(ref) != len(queries):
        raise SystemExit(f"ref/correct {len(ref)} 줄 != 질문 {len(queries)}")
    n = a.limit or len(queries)
    ecfg, ev = _evaluator(a)
    run_dir = _run_dir("verify_refinement", a.run_id)
    cfg = {
        "experiment": "official_crag",
        "step": "verify-refinement",
        "evaluator": ecfg,
        "limit": n,
    }
    _meta(run_dir, cfg, {"evaluator": ev.describe(), "decompose_mode": a.decompose_mode})
    out = JsonlWriter(run_dir / "refinement.jsonl")
    exact = 0
    for q, r in zip(queries[:n], ref[:n], strict=True):
        # 공식 입력: retrieved_psgs 한 줄을 '[sep]' 로 나눈 passage (앞뒤 공백 유지)
        psgs = official_passages_from_line(retrieved_psgs_line(q.passages))
        res = internal_knowledge(q.question, psgs, ev, a.decompose_mode)
        same = res.text.strip() == r.strip()
        exact += same
        out.write(
            {
                "index": q.index,
                "exact_match": same,
                "selected_strip_index": [s.index for s in res.selected],
                "ranking": res.ranking,
                "scores": [s.score for s in res.strips],
                "regenerated": res.text if not same else None,
                "official": r if not same else None,
            }
        )
    summary = {
        "n": n,
        "exact_match": exact,
        "exact_match_rate": exact / n if n else None,
        "note": "불일치는 점수 미세 차이 (하드웨어·버전) 로 순위가 바뀐 경우일 수 있음 -> refinement.jsonl",  # noqa: E501
    }
    write_new(run_dir / "summary.json", json.dumps(summary, indent=2))
    return run_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def cmd_generate(a) -> Path:
    from rag.crag_official.completion_engine import build_generator_engine
    from rag.llm.engine_llm import EngineLLM
    from rag.telemetry.calls import CallRecorder

    cfg = load_layered_config(a.config, a.overlay, a.set)
    paths = official_paths(Path(a.repo))
    queries = load_official_queries(paths["test"], source.OFFICIAL_NDOCS, paths["sources"])
    actions = None
    if a.method == "crag":
        if not a.scores:
            raise SystemExit("--method crag 에는 --scores <score run 폴더> 가 필요")
        actions = [r["action"] for r in _read_jsonl(Path(a.scores) / "scores.jsonl")]
        knowledge = {k: load_ref(paths[k]) for k in ("correct", "incorrect", "ambiguous")}
        for k, v in knowledge.items():
            if len(v) != len(queries):
                raise SystemExit(f"ref/{k} {len(v)} 줄 != {len(queries)}")
    n = min(a.limit or len(queries), len(actions) if actions else len(queries))
    engine = build_generator_engine(cfg["llm"])
    run_dir = _run_dir(f"generate_{a.method}", a.run_id)
    cfg.update(step="generate", method=a.method, scores_run=a.scores, limit=n)
    _meta(run_dir, cfg, {"deviations": cfg.get("deviations")}, engines={"generator": engine})
    out = JsonlWriter(run_dir / "preds.jsonl")
    preds = []
    for i, q in enumerate(queries[:n]):
        if a.method == "rag":
            paragraph, action = offgen.rag_paragraph(q.passages), None
        else:
            action = actions[i]
            paragraph = knowledge[action][i]  # 공식처럼 strip 만 ('#' 접두 유지)
        prompt = offgen.format_popqa_prompt(q.question, paragraph)
        rec = CallRecorder()
        llm = EngineLLM(engine, recorder=rec)
        raw = llm.generate(prompt, kind="generate", max_tokens=cfg["llm"].get("max_tokens", 100))
        pred = offgen.postprocess_answer(raw)
        preds.append(pred)
        c = rec.to_dicts()[0]
        out.write(
            {
                "index": q.index,
                "question": q.question,
                "action": action,
                "prediction": pred,
                "raw": raw,
                "prompt_tokens": c["prompt_tokens"],
                "out_tokens": c["out_tokens"],
                "wall": c["wall"],
                "ttft": c["ttft"],
                "finish_reason": c["finish_reason"],
            }
        )
    # 공식 출력 형식 (eval.py 입력과 같음)
    write_new(run_dir / "preds.txt", "\n".join(preds))
    return run_dir


def cmd_evaluate(a) -> Path:
    paths = official_paths(Path(a.repo))
    queries = load_official_queries(paths["test"], source.OFFICIAL_NDOCS, paths["sources"])
    align = attach_answers(queries, a.answers)
    rows = _read_jsonl(Path(a.preds) / "preds.jsonl")
    subset = None
    if a.subset_ids:
        subset = {
            x.strip() for x in Path(a.subset_ids).read_text(encoding="utf-8").split() if x.strip()
        }
    run_dir = _run_dir("evaluate", a.run_id)
    cfg = {
        "experiment": "official_crag",
        "step": "evaluate",
        "preds": a.preds,
        "answers": a.answers,
    }
    _meta(run_dir, cfg, {"alignment": align})
    out = JsonlWriter(run_dir / "per_question.jsonl")
    by_action: dict = {}
    tot = {"official_match": 0, "acc": 0}
    sub = {"n": 0, "official_match": 0, "acc": 0}
    for r in rows:
        q = queries[r["index"]]
        m = offgen.official_match(r["prediction"], q.answers)
        acc = accuracy(r["prediction"], q.answers)
        tot["official_match"] += m
        tot["acc"] += acc
        b = by_action.setdefault(r.get("action"), {"n": 0, "official_match": 0})
        b["n"] += 1
        b["official_match"] += m
        if subset is not None and q.ext_id in subset:
            sub["n"] += 1
            sub["official_match"] += m
            sub["acc"] += acc
        out.write(
            {
                "index": q.index,
                "question_id": q.ext_id,
                "action": r.get("action"),
                "official_match": m,
                "acc": acc,
            }
        )
    n = len(rows)
    summary = {
        "n": n,
        "official_match": tot["official_match"] / n if n else None,
        "acc_normalized": tot["acc"] / n if n else None,
        "by_action": {
            str(k): {**v, "rate": v["official_match"] / v["n"]} for k, v in by_action.items()
        },
        "subset": None
        if subset is None
        else {
            **sub,
            "ids_file": a.subset_ids,
            "n_ids": len(subset),
            "official_match_rate": sub["official_match"] / sub["n"] if sub["n"] else None,
        },
        "note": "official_match = 공식 eval.py match (대소문자 구분). acc_normalized = rag.eval.metrics.accuracy",  # noqa: E501
    }
    write_new(run_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo", default=str(DEFAULT_REPO), help="공식 저장소 (fetch_official.py repo)"
    )
    ap.add_argument("--run-id", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "verify-refinement"):
        p = sub.add_parser(name)
        p.add_argument("--weights", default=None, help="기본: 환경변수 CRAG_EVALUATOR_PATH")
        p.add_argument("--device", default="cpu")
        p.add_argument("--limit", type=int, default=None)
        if name == "verify-refinement":
            p.add_argument("--decompose-mode", default="selection")
    g = sub.add_parser("generate")
    g.add_argument("--method", choices=["crag", "rag"], required=True)
    g.add_argument("--scores", default=None)
    g.add_argument("--config", default="experiments/official_crag/configs/mode_a_popqa.yaml")
    g.add_argument("--overlay", action="append", default=[])
    g.add_argument("--set", action="append", default=[])
    g.add_argument("--limit", type=int, default=None)
    e = sub.add_parser("evaluate")
    e.add_argument("--preds", required=True)
    e.add_argument("--answers", required=True, help="Self-RAG popqa_longtail_w_gs.jsonl 등")
    e.add_argument("--subset-ids", default=None, help="질문 id 목록 (예: C1 230문항)")
    a = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    fn = {
        "score": cmd_score,
        "verify-refinement": cmd_verify_refinement,
        "generate": cmd_generate,
        "evaluate": cmd_evaluate,
    }[a.cmd]
    print(f"saved -> {fn(a)}")


if __name__ == "__main__":
    main()
