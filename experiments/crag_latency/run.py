"""Vanilla RAG / C-RAG 를 같은 query 셋에 실행하고 query 단위 결과를 jsonl 로 저장.

    uv run python experiments/crag_latency/run.py --config experiments/crag_latency/config.yaml
"""

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from rag.config import load_config
from rag.data.loader import load_qa_jsonl
from rag.eval.metrics import score_all
from rag.pipelines.crag import CRAG
from rag.pipelines.vanilla import VanillaRAG


def build_pipelines(cfg: dict) -> dict:
    """TODO: cfg 에 맞는 retriever / llm / web_search 구현체를 만들어 연결.

    두 파이프라인이 *같은* retriever, llm 인스턴스를 공유해야 공정 비교가 된다.
    """
    retriever = llm = web_search = None
    raise NotImplementedError("retriever / llm / web_search 구현 후 연결")
    return {
        "vanilla": VanillaRAG(retriever, llm, top_k=cfg["retriever"]["top_k"]),
        "crag": CRAG(
            retriever,
            llm,
            web_search,
            top_k=cfg["retriever"]["top_k"],
            web_top_k=cfg["crag"]["web_search"]["top_k"],
            upper_threshold=cfg["crag"]["upper_threshold"],
            lower_threshold=cfg["crag"]["lower_threshold"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/crag_latency/config.yaml")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    random.seed(cfg["seed"])

    examples = load_qa_jsonl(cfg["dataset"]["path"], cfg["dataset"]["limit"])
    pipelines = {k: v for k, v in build_pipelines(cfg).items() if k in cfg["run"]["pipelines"]}

    out_dir = Path(cfg["output_dir"]) / "crag_latency" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    warmup = cfg["run"]["warmup_queries"]
    for ex in examples[:warmup]:
        for p in pipelines.values():
            p.run(ex)

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for ex in tqdm(examples[warmup:]):
            # 실행 순서에 따른 캐시 효과를 줄이기 위해 query 마다 순서를 섞는다
            order = list(pipelines.items())
            random.shuffle(order)
            for name, pipeline in order:
                for rep in range(cfg["run"]["repeats"]):
                    res = pipeline.run(ex)
                    row = asdict(res)
                    row["contexts"] = [d["id"] for d in row["contexts"]]
                    row["repeat"] = rep
                    row["gold_answers"] = ex.answers
                    row.update(score_all(res.answer, ex.answers))
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
