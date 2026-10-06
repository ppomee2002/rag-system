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
from rag.llm.anthropic_llm import AnthropicLLM
from rag.pipelines.crag import CRAG
from rag.pipelines.vanilla import VanillaRAG
from rag.retrieval.faiss_retriever import FaissRetriever
from rag.retrieval.tavily_search import TavilyWebSearch


def build_llm(llm_cfg: dict) -> AnthropicLLM:
    if llm_cfg["provider"] != "anthropic":
        raise ValueError(f"지원하지 않는 llm provider: {llm_cfg['provider']}")
    return AnthropicLLM(
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 512),
    )


def build_pipelines(cfg: dict) -> dict:
    """cfg 에 맞는 retriever / llm / web_search 구현체를 만들어 연결.

    두 파이프라인이 *같은* retriever, llm 인스턴스를 공유해야 공정 비교가 된다.
    """
    rcfg = cfg["retriever"]
    if rcfg["type"] != "faiss":
        raise ValueError(f"지원하지 않는 retriever: {rcfg['type']}")
    retriever = FaissRetriever(
        index_path=rcfg["index_path"],
        docs_path=rcfg["docs_path"],
        model_name=cfg["embedding"]["model"],
        query_prefix=rcfg.get("query_prefix", ""),
        device=rcfg.get("device"),
    )

    llm = build_llm(cfg["llm"])
    grader_cfg = cfg["crag"]["grader"]
    grader_llm = build_llm({**cfg["llm"], **grader_cfg}) if grader_cfg.get("model") else None

    wcfg = cfg["crag"]["web_search"]
    if wcfg["provider"] != "tavily":
        raise ValueError(f"지원하지 않는 web_search provider: {wcfg['provider']}")
    web_search = TavilyWebSearch(search_depth=wcfg.get("search_depth", "basic"))

    crag_kwargs = dict(
        grader_llm=grader_llm,
        top_k=rcfg["top_k"],
        web_top_k=cfg["crag"]["web_search"]["top_k"],
        upper_threshold=cfg["crag"]["upper_threshold"],
        lower_threshold=cfg["crag"]["lower_threshold"],
    )
    return {
        "vanilla": VanillaRAG(retriever, llm, top_k=rcfg["top_k"]),
        "crag": CRAG(retriever, llm, web_search, **crag_kwargs),
        # grader 문서별 호출을 동시에 실행하는 변형 (grading latency 비교용)
        "crag_parallel": CRAG(
            retriever, llm, web_search, **crag_kwargs, parallel_grading=True, name="crag_parallel"
        ),
    }


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """--set a.b=value 형식 override. value 는 YAML 로 해석한다 (예: [vanilla,crag], true)."""
    for item in overrides:
        key, _, raw = item.partition("=")
        *parents, leaf = key.split(".")
        node = cfg
        for k in parents:
            node = node[k]
        node[leaf] = yaml.safe_load(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/crag_latency/config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="dataset.limit override (테스트용)")
    parser.add_argument(
        "--resume", type=Path, default=None, help="중단된 run 폴더 (완료된 행은 건너뜀)"
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="config override"
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    if args.limit is not None:
        cfg["dataset"]["limit"] = args.limit
    apply_overrides(cfg, args.set)
    random.seed(cfg["seed"])

    # PopQA 파일은 질문 유형(prop) 순으로 정렬돼 있어 앞에서 자르면 유형이 편향된다
    # -> seed 고정 shuffle 후 limit 적용 (warmup 도 섞인 순서의 앞부분)
    examples = load_qa_jsonl(cfg["dataset"]["path"])
    random.Random(cfg["seed"]).shuffle(examples)
    examples = examples[: cfg["dataset"]["limit"]]
    pipelines = {k: v for k, v in build_pipelines(cfg).items() if k in cfg["run"]["pipelines"]}

    if args.resume:
        out_dir = args.resume
        # 강제 종료로 마지막 줄이 잘렸을 수 있으므로 정상 행만 남기고 다시 쓴다
        rows = []
        for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        (out_dir / "results.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        done = {(r["query_id"], r["pipeline"], r["repeat"]) for r in rows}
        print(f"resume {out_dir}: {len(done)} rows already done")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg["output_dir"]) / "crag_latency" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        done = set()
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    warmup = cfg["run"]["warmup_queries"]
    for ex in examples[:warmup]:
        for p in pipelines.values():
            p.run(ex)

    n_errors = 0
    with (
        open(out_dir / "results.jsonl", "a", encoding="utf-8") as f,
        open(out_dir / "errors.jsonl", "a", encoding="utf-8") as ferr,
    ):
        for ex in tqdm(examples[warmup:]):
            # 실행 순서에 따른 캐시 효과를 줄이기 위해 query 마다 순서를 섞는다
            order = list(pipelines.items())
            random.shuffle(order)
            for name, pipeline in order:
                for rep in range(cfg["run"]["repeats"]):
                    if (ex.id, name, rep) in done:
                        continue
                    try:
                        res = pipeline.run(ex)
                    except Exception as e:  # API 오류 등: 기록 후 계속. --resume 으로 재시도
                        n_errors += 1
                        err = {"query_id": ex.id, "pipeline": name, "repeat": rep, "error": repr(e)}
                        ferr.write(json.dumps(err, ensure_ascii=False) + "\n")
                        ferr.flush()
                        continue
                    row = asdict(res)
                    row["contexts"] = [d["id"] for d in row["contexts"]]
                    row["repeat"] = rep
                    row["gold_answers"] = ex.answers
                    row.update(score_all(res.answer, ex.answers))
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # 중단돼도 완료된 query 는 남도록

    if n_errors:
        print(f"{n_errors} errors -> {out_dir / 'errors.jsonl'} (--resume 으로 재시도)")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
