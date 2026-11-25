"""C1/C2/C3 공유 Correction 검색 캐시 만들기 (web_cache.v2).

고정 입력의 모든 질문에 대해 MODE B 와 같은 방식으로 correction 검색어를 만들고 (같은 prompt·
generator), 실제 외부 검색 (Tavily) 을 한 번 해서 기록한다. 이후 모든 조건은
`web_search.provider=replay, replay_key=query_strict, replay_paths=[<이 파일>]` 로
같은 결과를 재생한다.

- 기존 rag.retrieval.web_cache.RecordingWebSearch / ControlledCRAG._correction_query 를 그대로 쓴다
  (별도 검색 시스템 아님).
- 캐시 키는 (query_id, 실제 검색어, top_k) + 검색 조건 (provider, params, query_mode).
  조건 실행 때 검색어가 기록과 다르면 query_id 가 같아도 miss (query_mismatch 로 기록).
- correction 이 필요한 질문은 grader·정책마다 다르므로 모든 질문을 미리 검색한다 (상위집합).
- query_mode 마다 (legacy_rewrite / official_keywords) 캐시를 따로 만든다. 서로 호환되지 않는다.
- API key 는 기록하지 않는다 (환경변수 TAVILY_API_KEY 만 사용).

    uv run --extra crag --extra official python experiments/official_crag/prefetch_web.py \
        --config experiments/official_crag/config.yaml \
        --overlay experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml \
        --set inputs.frozen_path=$FROZEN --name popqa_c1_common_legacy_rewrite

결과 (이미 있으면 실패): results/official_crag/web_cache/<name>/
  web_cache.jsonl     v2 기록 (query_id, query, provider, params, query_mode, searched_at_utc,
                      docs, raw_response, latency, cache_version)
  calls.jsonl         검색어 생성 LLM 호출
  manifest.json       고정 입력 sha256, generator, prompt hash, 검색 조건, git, 결과 요약
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import load_layered_config  # noqa: E402
from rag.crag_official.completion_engine import build_generator_engine  # noqa: E402
from rag.evidence.budget import build_budget  # noqa: E402
from rag.evidence.frozen import file_sha256, load_frozen  # noqa: E402
from rag.evidence.manifest import generator_view, prompt_templates, used_prompts  # noqa: E402
from rag.evidence.pipeline import ControlledCRAG, PipelineConfig  # noqa: E402
from rag.evidence.policies import get_policy  # noqa: E402
from rag.llm.engine_llm import EngineLLM  # noqa: E402
from rag.retrieval.web_cache import CACHE_VERSION, RecordingWebSearch  # noqa: E402
from rag.telemetry.calls import CallRecorder  # noqa: E402
from rag.telemetry.run_meta import JsonlWriter, create_run_dir, git_info, write_new  # noqa: E402


def query_builder(cfg: dict) -> ControlledCRAG:
    """MODE B 와 같은 PipelineConfig 로 _correction_query 만 쓰는 객체."""
    corr = cfg.get("correction") or {}
    pcfg = PipelineConfig(
        web_top_k=(cfg.get("web_search") or {}).get("top_k", 5),
        query_mode=corr.get("query_mode", "legacy_rewrite"),
        wiki_prefix=bool(corr.get("wiki_prefix", False)),
    )
    return ControlledCRAG(get_policy("vanilla"), None, build_budget(None), None, pcfg)


def prefetch(cfg: dict, name: str, engine=None, web=None, output_dir="results") -> Path:
    inputs = cfg.get("inputs") or {}
    items, info = load_frozen(inputs["frozen_path"], inputs.get("verify_manifest", True))
    if inputs.get("limit"):
        items = items[: inputs["limit"]]
    w = cfg.get("web_search") or {}
    top_k = w.get("top_k", 5)
    qmode = (cfg.get("correction") or {}).get("query_mode", "legacy_rewrite")
    if web is None:
        from rag.retrieval.tavily_search import TavilyWebSearch

        web = TavilyWebSearch(
            search_depth=w.get("search_depth", "basic"),
            include_raw_content=bool(w.get("include_raw_content", False)),
        )
    engine = engine or build_generator_engine(cfg["llm"])
    run_dir = create_run_dir("official_crag/web_cache", output_dir, name)
    params = web.params() if hasattr(web, "params") else {}
    rec = RecordingWebSearch(web, run_dir / "web_cache.jsonl", "tavily", params, qmode)
    builder = query_builder(cfg)
    calls = JsonlWriter(run_dir / "calls.jsonl")
    errors, n_empty = [], 0
    t0 = time.perf_counter()
    for it in items:
        cr = CallRecorder(t0)
        llm = EngineLLM(engine, recorder=cr)
        try:
            q = builder._correction_query(it.question, llm)
            docs = rec.bind(it.question_id).search(q, top_k)
            n_empty += not docs
        except Exception as e:  # noqa: BLE001 - 기록하고 계속 (실패 질문은 캐시에 없음 -> miss)
            errors.append({"query_id": it.question_id, "error": f"{type(e).__name__}: {e}"[:300]})
        for c in cr.to_dicts():
            calls.write({"question_id": it.question_id, **c})
    manifest = {
        "cache_version": CACHE_VERSION,
        "name": name,
        "inputs": {"path": info["path"], "sha256": info["sha256"], "n": len(items)},
        "question_ids": [it.question_id for it in items],
        "search": {"provider": "tavily", "params": params, "top_k": top_k},
        "correction": {
            "query_mode": qmode,
            "wiki_prefix": bool((cfg.get("correction") or {}).get("wiki_prefix", False)),
            "prompt": {
                "name": used_prompts(cfg)["correction_query"],
                **prompt_templates()[used_prompts(cfg)["correction_query"]],
            },
        },
        "generator": generator_view(cfg),
        "engine": engine.describe() if hasattr(engine, "describe") else None,
        "n_recorded": rec.n_recorded,
        "n_empty": n_empty,
        "errors": errors,
        "wall_seconds": time.perf_counter() - t0,
        "cache_sha256": file_sha256(run_dir / "web_cache.jsonl")
        if (run_dir / "web_cache.jsonl").exists()
        else None,
        "git": git_info(),
        "secrets": "API key / token 은 기록하지 않음 (params·raw_response 는 비밀 키 이름을 제거)",
        "usage": (
            "--set web_search.provider=replay --set web_search.replay_key=query_strict "
            f"--set web_search.replay_paths=[{run_dir / 'web_cache.jsonl'}] "
            f"--set correction.query_mode={qmode}"
        ),
    }
    write_new(
        run_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    return run_dir


def main(argv=None) -> Path:
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/official_crag/config.yaml")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--name", required=True, help="results/official_crag/web_cache/<name>/")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    load_dotenv()
    cfg = load_layered_config(a.config, a.overlay, a.set)
    if a.limit is not None:
        cfg.setdefault("inputs", {})["limit"] = a.limit
    out = prefetch(cfg, a.name, output_dir=cfg.get("output_dir", "results"))
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
