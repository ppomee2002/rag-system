"""MODE A 질문의 correction web 기록 만들기 (B / C / Cbn 이 같은 web evidence·지연을 재생).

질문마다 한 번: 공식 keyword prompt 를 MODE A generator 로 실행 -> "Wikipedia, " 접두 -> Tavily
(include_raw_content=True, 페이지 원문) -> web_cache.v2 기록 (query_id = MODE A index, 실측 지연).
기존 rag.retrieval.web_cache.RecordingWebSearch 를 그대로 쓴다. API key 는 기록하지 않는다.

공식과 다른 점 (README deviations): GPT-3.5 -> MODE A generator, serper -> Tavily, HTML <p> ->
raw_content 줄. evidence_source=official_ref 에서는 generation 입력에 영향이 없고
(저자 지식 파일 사용),
external refinement 의 T5 작업량 (문단 수) 과 web 지연에만 영향을 준다.

    uv run --extra crag --extra official python experiments/c2_gating/prefetch_web.py \
        --mode-a-config experiments/official_crag/configs/mode_a_popqa.yaml \
        --indices-file <질문 index 목록> --name popqa_modea_smoke
결과 (이미 있으면 실패): results/c2_gating/web_cache/<name>/{web_cache.jsonl, calls.jsonl,
  manifest.json}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from c2_data import (  # noqa: E402,E501
    load_items,
    read_jsonl,
    reference_actions,
    select_indices,
    warmup_pool,
)
from c2_pipeline import correction_query  # noqa: E402

from rag.config import load_layered_config  # noqa: E402
from rag.crag_official import source  # noqa: E402
from rag.crag_official.completion_engine import build_generator_engine  # noqa: E402
from rag.llm.engine_llm import EngineLLM  # noqa: E402
from rag.retrieval.web_cache import CACHE_VERSION, RecordingWebSearch  # noqa: E402
from rag.telemetry.calls import CallRecorder  # noqa: E402
from rag.telemetry.run_meta import JsonlWriter, create_run_dir, git_info, write_new  # noqa: E402

ROOT = HERE.parents[1]


def main(argv=None) -> Path:
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-a-config", required=True)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--repo", default=str(ROOT / "third_party" / "CRAG"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--indices-file", action="append", default=[])
    ap.add_argument(
        "--skip-cache",
        action="append",
        default=[],
        help="이미 기록된 web_cache.jsonl (같은 query_id 는 다시 검색하지 않음)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-action", type=int, default=None)
    ap.add_argument("--reference-scores", default=None)
    ap.add_argument(
        "--include-warmup",
        type=int,
        default=10,
        help="warmup 후보 (c2_data.warmup_pool 앞쪽) 기록 수",
    )
    a = ap.parse_args(argv)
    load_dotenv()
    cfg = load_layered_config(a.mode_a_config, [a.config], a.set)
    items, _ = load_items(a.repo)
    ref = reference_actions(a.reference_scores) if a.reference_scores else None
    idx, sel = select_indices(
        items,
        indices_file=a.indices_file,
        limit=a.limit,
        per_action=a.per_action,
        ref_actions=ref,
        seed=int(cfg.get("seed", 0)),
    )
    warm = warmup_pool(len(items), int(cfg.get("seed", 0)))[: a.include_warmup]
    done = {str(r.get("query_id")) for p in a.skip_cache for r in read_jsonl(p)}
    todo = [i for i in dict.fromkeys([*idx, *warm]) if str(i) not in done]
    g, w = cfg["gating"], cfg.get("web") or {}
    from rag.retrieval.tavily_search import TavilyWebSearch

    web = TavilyWebSearch(search_depth=w.get("search_depth", "basic"), include_raw_content=True)
    engine = build_generator_engine(cfg["llm"])
    run_dir = create_run_dir("c2_gating/web_cache", "results", a.name)
    rec = RecordingWebSearch(
        web,
        run_dir / "web_cache.jsonl",
        "tavily",
        web.params(),
        "official_keywords_wiki" if g.get("wiki_prefix", True) else "official_keywords",
    )
    calls = JsonlWriter(run_dir / "calls.jsonl")
    errors = []
    t0 = time.perf_counter()
    for i in todo:
        it = items[i]
        cr = CallRecorder(t0)
        try:
            q = correction_query(
                it.question,
                EngineLLM(engine, recorder=cr),
                g.get("rewrite_max_tokens", 32),
                g.get("wiki_prefix", True),
            )
            rec.bind(it.query_id).search(q, w.get("top_k", 5))
        except Exception as e:  # noqa: BLE001 - 기록하고 계속 (기록 없는 질문은 run.py 가 거부)
            errors.append({"index": i, "error": f"{type(e).__name__}: {e}"[:300]})
        for c in cr.to_dicts():
            calls.write({"index": i, **c})
    manifest = {
        "cache_version": CACHE_VERSION,
        "selection": sel,
        "indices": idx,
        "warmup_candidates": warm,
        "skipped_already_cached": len(set(idx) | set(warm)) - len(todo),
        "skip_cache": a.skip_cache,
        "search": {"provider": "tavily", "params": web.params(), "top_k": w.get("top_k", 5)},
        "query": {
            "prompt": "source.POPQA_KEYWORD_PROMPT",
            "wiki_prefix": g.get("wiki_prefix", True),
            "official_model": source.KEYWORD_MODEL,
            "used_model": cfg["llm"].get("model"),
        },
        "n_recorded": rec.n_recorded,
        "errors": errors,
        "wall_seconds": time.perf_counter() - t0,
        "git": git_info(),
    }
    write_new(run_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"saved -> {run_dir} (recorded {rec.n_recorded}, errors {len(errors)})")
    return run_dir


if __name__ == "__main__":
    main()
