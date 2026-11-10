"""조건 간 비교 가능성 검사 (Full / Lazy / Batch, 스케줄링 정책 등).

    uv run python experiments/crag_system/comparability.py <run_dir> <run_dir> [...] \
        [--out report.json]

같아야 하는 것 (critical, 다르면 실패):
  평가 질문 목록과 순서, dataset / retriever / embedding / CRAG 임계값,
  LLM 설정 (priority 관련 제외), 엔진 모델·버전·max_model_len,
  query 별 최초 검색 결과 (retrieval_doc_ids), web replay 원본 파일
조건 변수 (varied, 달라도 됨): pipeline, grading, scheduler, llm priority 설정, exp_id, session
경고 (warnings): correction 경로의 web 문서가 다른 query (rewrite 결과 차이),
  다른 사용자의 GPU 프로세스, prefix cache 초기화 여부 차이
교란 요인 기록 (confounders): run 별 시작 시각, session/order, prefix cache 설정·결과,
  web search 모드
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from rag.retrieval.web_cache import read_jsonl  # 줄바꿈 기준, 손상 시 경로:줄번호 보고

PRIORITY_KEYS = ("send_priority", "server_scheduling_policy", "launch_meta")


def _load(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    rows = read_jsonl(run_dir / "queries.jsonl")
    end = run_dir / "run_end.json"
    return {
        "dir": str(run_dir),
        "cfg": yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")),
        "meta": json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8")),
        "end": json.loads(end.read_text(encoding="utf-8")) if end.exists() else {},
        "rows": [r for r in rows if not r.get("warmup")],
    }


def _engine_identity(meta: dict) -> dict:
    eng = (meta.get("engines") or {}).get("main") or {}
    models = eng.get("models") or {}
    data = models.get("data") if isinstance(models, dict) else None
    first = data[0] if data else {}
    return {
        "engine": eng.get("engine"),
        "model": eng.get("model"),
        "version": eng.get("version"),
        "served_root": first.get("root"),
        "max_model_len": first.get("max_model_len"),
        "model_info": eng.get("model_info"),  # Ollama: digest / quantization
    }


def _same(label: str, values: list, out: list) -> None:
    first = values[0]
    if any(v != first for v in values[1:]):
        out.append({"check": label, "values": values})


def compare_runs(dirs: list[str | Path]) -> dict:
    if len(dirs) < 2:
        raise ValueError("비교할 run 이 2개 이상 필요")
    runs = [_load(d) for d in dirs]
    critical: list = []
    warnings: list = []

    _same("eval_query_order", [[r["query_id"] for r in x["rows"]] for x in runs], critical)
    for key in ("dataset", "retriever", "embedding"):
        _same(key, [x["cfg"].get(key) for x in runs], critical)
    _same(
        "crag_thresholds",
        [
            {k: (x["cfg"].get("crag") or {}).get(k) for k in ("upper_threshold", "lower_threshold")}
            for x in runs
        ],
        critical,
    )
    _same(
        "llm",
        [{k: v for k, v in x["cfg"]["llm"].items() if k not in PRIORITY_KEYS} for x in runs],
        critical,
    )
    _same("engine_identity", [_engine_identity(x["meta"]) for x in runs], critical)
    _same(
        "web_replay_sources",
        [
            ((x["meta"].get("comparability") or {}).get("web_search") or {}).get("sources")
            for x in runs
        ],
        critical,
    )

    # query 별 최초 검색 결과
    by_q = [{r["request_index"]: r for r in x["rows"]} for x in runs]
    common = set.intersection(*(set(b) for b in by_q))
    retrieval_diff = [
        i
        for i in sorted(common)
        if any(
            b[i].get("retrieval_doc_ids") != by_q[0][i].get("retrieval_doc_ids") for b in by_q[1:]
        )
    ]
    if retrieval_diff:
        critical.append(
            {
                "check": "retrieval_doc_ids",
                "n_requests": len(retrieval_diff),
                "request_index": retrieval_diff[:20],
            }
        )

    # correction 경로 web 문서 (두 run 모두 correction 한 query 만)
    web_diff = []
    for i in sorted(common):
        rs = [b[i] for b in by_q]
        if all(r.get("corrective_action") not in (None, "correct") for r in rs):
            ids = [(r.get("extra") or {}).get("web_doc_ids") for r in rs]
            if any(v != ids[0] for v in ids[1:]):
                web_diff.append(i)
    if web_diff:
        warnings.append(
            {
                "check": "web_doc_ids_on_corrected",
                "n_requests": len(web_diff),
                "request_index": web_diff[:20],
                "note": "rewrite 결과 또는 실시간 검색 결과가 다름. replay 사용 권장",
            }
        )

    others = [len((x["meta"].get("gpu") or {}).get("compute_apps") or []) for x in runs]
    if any(others):
        warnings.append(
            {
                "check": "gpu_processes_at_start",
                "counts": others,
                "note": "시작 시 GPU 를 쓰던 프로세스 (자기 vLLM 서버 포함) 수",
            }
        )
    reset = [
        bool((x["meta"].get("comparability") or {}).get("prefix_cache_reset_requested"))
        for x in runs
    ]
    if len(set(reset)) > 1:
        warnings.append({"check": "prefix_cache_reset_mismatch", "values": reset})

    varied = {key: [x["meta"].get(key) for x in runs] for key in ("pipeline", "grading", "mode")}
    varied["scheduler"] = [x["meta"].get("scheduler") for x in runs]
    varied["llm_priority"] = [{k: x["cfg"]["llm"].get(k) for k in PRIORITY_KEYS} for x in runs]
    confounders = [
        {
            "dir": x["dir"],
            "started_utc": x["meta"].get("timestamp_utc"),
            "session": (x["meta"].get("comparability") or {}).get("session"),
            "prefix_cache_reset_requested": reset[n],
            "prefix_cache_reset_result": x["end"].get("prefix_cache_reset"),
            "prefix_cache_delta": x["end"].get("prefix_cache_delta"),
            "web_search": ((x["meta"].get("comparability") or {}).get("web_search") or {}),
            "web_replay_stats": x["end"].get("web_search"),
        }
        for n, x in enumerate(runs)
    ]
    return {
        "ok": not critical,
        "runs": [x["dir"] for x in runs],
        "n_common_requests": len(common),
        "critical": critical,
        "warnings": warnings,
        "varied": varied,
        "confounders": confounders,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    report = compare_runs(a.runs)
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if a.out:
        with open(a.out, "x", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
