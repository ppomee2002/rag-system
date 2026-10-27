"""lazy grading 품질 측정: retrieval 순서로 grade 하다 upper 이상이 나오면 나머지 grade 를 생략.

    uv run python experiments/correction_scheduling/run_lazy.py <run_dir> [--limit 10]

grade 호출은 문서마다 독립 (temperature 0) 이므로 멈추기 전까지의 score 는 실험 1 값과 같다
-> grade 는 재실행하지 않고, context 가 full C-RAG 와 달라지는 query 만 generation 을 새로 한다.

  full       실험 1 C-RAG (5개 모두 grade, score >= upper 문서 전부 사용)
  lazy_hit   처음 upper 를 넘은 문서 1개만 사용
  lazy_rest  처음 upper 를 넘은 문서 + 그 뒤의 grade 안 한 문서 (앞에서 낮게 나온 문서는 제외)

upper 를 넘는 문서가 없으면 5개를 모두 grade 하게 되므로 full 과 같다 (답변 재사용).
재사용이 타당한지 보기 위해 full context 를 그대로 다시 generate 해서 답이 같은지 먼저 확인한다.
출력: <run_dir>/lazy/lazy.jsonl, determinism.jsonl, summary.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SharedRetriever  # noqa: E402

from rag.config import load_config  # noqa: E402
from rag.eval.metrics import score_all  # noqa: E402
from rag.llm.ollama_llm import OllamaLLM  # noqa: E402
from rag.pipelines import prompts  # noqa: E402
from rag.retrieval.faiss_retriever import FaissRetriever  # noqa: E402

VARIANTS = ("lazy_hit", "lazy_rest")


def lazy_contexts(docs, scores, upper):
    """(hit index, {variant: docs}) — hit 이 없으면 None (= full 과 동일)."""
    hit = next((i for i, s in enumerate(scores) if s >= upper), None)
    if hit is None:
        return None, {}
    return hit, {"lazy_hit": [docs[hit]], "lazy_rest": docs[hit:]}


class Generator:
    def __init__(self, llm_cfg):
        self.box = {}
        base = OllamaLLM(llm_cfg["model"], host=llm_cfg["host"])

        def post(payload):
            self.box["res"] = base._http_post(payload)
            return self.box["res"]

        self.llm = OllamaLLM(
            llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 512),
            num_ctx=llm_cfg["num_ctx"],
            host=llm_cfg["host"],
            post=post,
        )

    def __call__(self, question, docs) -> dict:
        prompt = prompts.GENERATE.format(context=prompts.format_context(docs), question=question)
        t = time.perf_counter()
        answer = self.llm.generate(prompt)
        res = self.box["res"]
        return {
            "answer": answer,
            "prompt_chars": len(prompt),
            "wall": time.perf_counter() - t,
            "server_total": res["total_duration"] * 1e-9,
            "prompt_tokens": res["prompt_eval_count"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", default="experiments/correction_scheduling/config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-determinism", type=int, default=20)
    args = parser.parse_args()
    cfg = load_config(args.config)
    upper = cfg["crag"]["upper_threshold"]
    out = args.run_dir / "lazy"
    out.mkdir(exist_ok=True)

    rows = [json.loads(line) for line in (args.run_dir / "profile.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["pipeline"] == "crag"][: args.limit]
    rcfg = cfg["retriever"]
    retriever = SharedRetriever(
        FaissRetriever(
            rcfg["index_path"],
            rcfg["docs_path"],
            cfg["embedding"]["model"],
            query_prefix=rcfg.get("query_prefix", ""),
            device=rcfg.get("device"),
        )
    )
    by_id = {d.id: d for d in retriever.inner.docs}
    gen = Generator(cfg["llm"])
    gen("warmup", [retriever.inner.docs[0]])

    # 1) 재사용 타당성: 실험 1 의 full context 를 그대로 다시 generate
    det = []
    for r in tqdm(rows[: args.n_determinism], desc="determinism"):
        # mock web 문서 id 는 "web:<chunk id>" (본문은 같은 chunk)
        g = gen(r["question"], [by_id[i.removeprefix("web:")] for i in r["contexts"]])
        det.append(
            {
                "query_id": r["query_id"],
                "same": g["answer"] == r["answer"],
                "old": r["answer"],
                "new": g["answer"],
            }
        )
    with open(out / "determinism.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in det)
    n_same = sum(d["same"] for d in det)
    print(f"determinism: {n_same}/{len(det)} identical")

    # 2) lazy variants. 같은 query 의 lazy_hit / lazy_rest prompt 는 앞부분(첫 문서)이 같아
    #    연달아 보내면 prompt cache 에 걸린다 -> variant 별 pass 로 나눠 실행
    plans = []
    for r in rows:
        docs = retriever.retrieve(r["question"], rcfg["top_k"])
        if not np.allclose([d.score for d in docs], r["extra"]["retrieval_scores"], atol=1e-4):
            raise RuntimeError(f"retrieval 결과가 실험 1 과 다름: {r['query_id']}")
        hit, ctxs = lazy_contexts(docs, r["grader_scores"], upper)
        grades = sorted((c for c in r["calls"] if c["kind"] == "grade"), key=lambda c: c["idx"])
        full_gen = next(c for c in r["calls"] if c["kind"] == "generate")
        plans.append((r, hit, ctxs, grades, full_gen))

    def row(r, hit, variant, answer, ids, reused, grades, gen_work, gen_chars, post_work):
        return {
            "query_id": r["query_id"],
            "variant": variant,
            "hit": hit,
            "action_full": r["corrective_action"],
            "gold_answers": r["gold_answers"],
            "answer": answer,
            "contexts": ids,
            "reused": reused,
            "n_grade": len(grades),
            "grade_work": sum(c["server_total"] for c in grades),
            "gen_work": gen_work,
            "gen_chars": gen_chars,
            "post_work": post_work,
            **score_all(answer, r["gold_answers"]),
        }

    results = []
    for r, hit, _, grades, full_gen in plans:
        post = sum(c["server_total"] for c in r["calls"] if c["kind"] != "grade")
        results.append(
            row(
                r,
                hit,
                "full",
                r["answer"],
                r["contexts"],
                True,
                grades,
                full_gen["server_total"],
                full_gen["prompt_chars"],
                post,
            )
        )
    for v in VARIANTS:
        for (r, hit, ctxs, grades, full_gen), full in tqdm(
            zip(plans, results[: len(plans)], strict=True), desc=v, total=len(plans)
        ):
            if hit is None:  # 5개 모두 grade -> full 과 동일
                results.append({**full, "variant": v})
                continue
            ids = [d.id for d in ctxs[v]]
            if ids == r["contexts"]:  # context 가 같으면 prompt 도 같다 -> 답 재사용
                g = {
                    "answer": r["answer"],
                    "server_total": full_gen["server_total"],
                    "prompt_chars": full_gen["prompt_chars"],
                }
            else:
                g = gen(r["question"], ctxs[v])
            results.append(
                row(
                    r,
                    hit,
                    v,
                    g["answer"],
                    ids,
                    ids == r["contexts"],
                    grades[: hit + 1],
                    g["server_total"],
                    g["prompt_chars"],
                    g["server_total"],
                )
            )
    with open(out / "lazy.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in results)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "determinism_same": n_same,
                "determinism_n": len(det),
                "upper": upper,
                "n_queries": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
