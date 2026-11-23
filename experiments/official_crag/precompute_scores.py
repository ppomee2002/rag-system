"""공식 T5 점수 미리 계산 (GPU 순차 실행용).

vLLM 을 띄우기 전에 evaluator 만 GPU 에 올려 점수를 캐시한다.

    # 1) vLLM 없이 (GPU 전체 사용 가능)
    uv run --extra crag --extra official python experiments/official_crag/precompute_scores.py \
        --config experiments/official_crag/config.yaml --set evaluator.device=cuda:0 \
        --out results/official_crag/score_cache/popqa_c1_230_t5.jsonl
    # 2) 그 다음 vLLM 을 띄우고 MODE B 를 캐시로 실행
    uv run python experiments/official_crag/run.py ... --set evaluator.backend=cache \
        --set "evaluator.cache_paths=[results/official_crag/score_cache/popqa_c1_230_t5.jsonl]"

계산 대상 (MODE B 파이프라인이 요청할 모든 쌍, 같은 key 규칙):
  - 문서 판정 점수 (truncation 없음, 빈 문서 제외)
  - 내부 refinement strip 점수 (decompose_mode, truncation=True)
  - web replay 기록이 있으면 web 문단 strip 점수 (external refinement)
캐시 run 의 grading stage 시간은 조회 시간이라 실제 evaluator 시간이 아니다 -> 결과의
evaluator_compute_seconds (기록 당시 계산 시간) 를 따로 보고, latency 비교에는 in-process 결과를
쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mode_b import build_web_search  # noqa: E402

from rag.config import load_layered_config  # noqa: E402
from rag.crag_official.evaluator import (  # noqa: E402
    build_backend,
    build_evaluator,
    pair_key,
    weights_fingerprint,
)
from rag.crag_official.refinement import external_knowledge, internal_knowledge  # noqa: E402
from rag.evidence.frozen import load_frozen  # noqa: E402
from rag.telemetry.run_meta import git_info  # noqa: E402


class RecordingBackend:
    """계산한 쌍을 캐시 파일에 남기는 backend wrapper."""

    def __init__(self, inner, path: Path):
        self.inner, self.path = inner, path
        self.seen: set[str] = set()
        self.name = f"recording:{inner.name}"
        self.f = open(path, "x", encoding="utf-8")

    def score_pairs(self, pairs, truncation):
        out = self.inner.score_pairs(pairs, truncation)
        for (q, p), s in zip(pairs, out, strict=True):
            k = pair_key(q, p, truncation)
            if k in self.seen:
                continue
            self.seen.add(k)
            row = {
                "key": k,
                "score": s.score,
                "n_tokens": s.n_tokens,
                "compute_seconds": s.compute_seconds,
                "truncation": truncation,
            }
            self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        return out

    def describe(self):
        return self.inner.describe()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/official_crag/config.yaml")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-web", action="store_true", help="web 문단 점수 생략")
    a = ap.parse_args(argv)
    cfg = load_layered_config(a.config, a.overlay, a.set)
    ecfg = {**cfg["evaluator"], "backend": "t5"}
    items, info = load_frozen(cfg["inputs"]["frozen_path"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    inner = build_backend(ecfg)
    rec = RecordingBackend(inner, out)
    meta = {
        "type": "meta",
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluator": inner.describe(),
        "weights": weights_fingerprint(inner.model_path),
        "inputs": info,
        "git": git_info(),
    }
    rec.f.write(json.dumps(meta, ensure_ascii=False, default=str) + "\n")
    ev = build_evaluator(ecfg, backend=rec)
    mode = (cfg.get("refinement") or {}).get("decompose_mode", "selection")

    web = None
    if not a.no_web and (cfg.get("web_search") or {}).get("provider") == "replay":
        wcfg = {
            **cfg,
            "web_search": {**cfg["web_search"], "latency_mode": "zero", "replay_on_miss": "empty"},
        }
        web = build_web_search(wcfg, out.parent)
    n_web = 0
    for i, it in enumerate(items):
        ev.score_documents(it.question, [d.text for d in it.docs])
        internal_knowledge(
            it.question, [d.text for d in it.docs], ev, mode, origins=[d.id for d in it.docs]
        )
        if web is not None:
            top_k = (cfg.get("web_search") or {}).get("top_k", 5)
            if web.key_mode == "query_strict":
                # 이 질문에 기록된 (조건이 같은) 검색어마다. v1 기록만 있으면 여기서 못 찾음
                # (evaluator.on_miss=delegate 로 실행 중 계산하거나 prefetch_web.py v2 캐시 사용)
                queries = sorted(web.recorded_queries.get(it.question_id, ()))
            else:
                queries = [""]  # query_id 모드: 기록된 검색어로 재생
            got = False
            for q in queries:
                docs = web.bind(it.question_id).search(q, top_k)
                if docs:
                    got = True
                    external_knowledge(it.question, docs, ev)
            n_web += got
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(items)}", file=sys.stderr)
    rec.f.close()
    print(f"saved -> {out} ({len(rec.seen)} pairs, web {n_web}/{len(items)})")


if __name__ == "__main__":
    main()
