"""MODE B 고정 입력 만들기: 문항 ID 목록 + 기존 run 의 PopQA Top-5 (문서와 순서) 고정.

    uv run python experiments/official_crag/freeze_inputs.py \
        --question-ids <문항 id 파일>   (또는 --question-ids-from-run <run>) \
        --expect-n 200 \
        --reference-run results/crag_system/smoke/SMOKE-A1-FULL-RETRY/20261110_111726 \
        [--reference-run <다른 run: Top-5 일치 검사용>] \
        [--live-check --config experiments/crag_system/config.yaml]   # FAISS 로 다시 검색해 대조
        --out results/official_crag/inputs/popqa_c1_common.jsonl

검사 (하나라도 실패하면 파일을 만들지 않음):
  질문·정답이 dataset 과 같음, Top-5 존재, 기준 run 간 Top-5 id·순서 일치,
  (--live-check) 현재 FAISS 검색 결과와 일치, 문서 텍스트 존재.
--reference-run 없이 --live-check 만 주면 현재 FAISS 검색 결과로 고정한다
(기존 결과와 대조 불가로 manifest 에 표시).

문항 수: 처음 지시서의 "230문항" 은 레포·결과 어디에도 근거가 없었다. 실제 완료된 C1 실험은
200문항 (crag_system config: dataset.limit 203 - run.warmup_queries 3 = 측정 200) 이므로
문항 수를 파일 이름에 고정하지 않고, --expect-n 으로 확인하고 manifest 의 n 에 기록한다.
--question-ids-from-run 은 run 의 queries.jsonl 에서 측정 문항 (warmup·오류 제외) id 를
기록 순서대로 가져온다. 여러 run 을 주면 모든 run 에 있는 문항만 남긴다 (첫 run 순서).
문항 ID 목록 (또는 그것을 만든 run 경로) 의 sha256 을 manifest 에 남긴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from rag.evidence.frozen import build_frozen_items, file_sha256, manifest_path_for
from rag.retrieval.web_cache import read_jsonl
from rag.telemetry.run_meta import git_info, write_new

ROOT = Path(__file__).resolve().parents[2]


def read_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        ids = data.get("question_ids") if isinstance(data, dict) else data
        if ids is None and isinstance(data, dict) and "requests" in data:  # crag_system workload
            ids = [r["query_id"] for r in data["requests"]]
        return [str(x) for x in ids]
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines and "," in lines[0]:  # csv: question_id 또는 query_id 열
        head = [h.strip() for h in lines[0].split(",")]
        col = head.index("question_id") if "question_id" in head else head.index("query_id")
        return [x.split(",")[col].strip() for x in lines[1:]]
    return lines


def ids_from_runs(run_dirs: list[Path]) -> tuple[list[str], dict]:
    """완료된 run 들의 측정 문항 id (warmup / 오류 제외). 여러 run 이면 교집합 (첫 run 순서)."""
    per_run = []
    info = []
    for d in run_dirs:
        ids, seen = [], set()
        n_warm = n_err = 0
        for r in read_jsonl(d / "queries.jsonl"):
            if r.get("warmup"):
                n_warm += 1
                continue
            if r.get("error"):
                n_err += 1
                continue
            q = str(r.get("query_id", r.get("question_id")))
            if q not in seen:
                seen.add(q)
                ids.append(q)
        per_run.append(ids)
        info.append(
            {
                "run": str(d),
                "n_measured_ok": len(ids),
                "n_warmup_rows": n_warm,
                "n_error_rows": n_err,
            }
        )
    common = set.intersection(*(set(x) for x in per_run)) if per_run else set()
    ordered = [q for q in per_run[0] if q in common] if per_run else []
    return ordered, {"runs": info, "n_common": len(ordered)}


def load_reference(run_dir: Path) -> tuple[dict[str, dict], dict]:
    rows: dict[str, dict] = {}
    for r in read_jsonl(run_dir / "queries.jsonl"):
        if r.get("error") or not r.get("retrieval_doc_ids"):
            continue
        qid = str(r["query_id"])
        if qid not in rows or (rows[qid].get("warmup") and not r.get("warmup")):
            rows[qid] = r
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return rows, cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--question-ids")
    src.add_argument("--question-ids-from-run", action="append", default=None)
    ap.add_argument("--expect-n", type=int, default=None, help="문항 수 확인 (다르면 실패)")
    ap.add_argument("--reference-run", action="append", default=[])
    ap.add_argument("--live-check", action="store_true")
    ap.add_argument(
        "--config", default=None, help="--live-check / dataset 경로용 crag_system config"
    )
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--docs-path", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"이미 있음 (덮어쓰지 않음): {out}")
    ids_from_run_info = None
    if a.question_ids:
        ids_path = Path(a.question_ids)
        ids = read_ids(ids_path)
        ids_source = str(ids_path)
    else:
        ids, ids_from_run_info = ids_from_runs([Path(d) for d in a.question_ids_from_run])
        ids_source = {"from_runs": a.question_ids_from_run}
    if len(set(ids)) != len(ids):
        raise SystemExit("문항 ID 목록에 중복이 있음")
    if a.expect_n is not None and len(ids) != a.expect_n:
        raise SystemExit(f"문항 수 {len(ids)} != --expect-n {a.expect_n} ({ids_from_run_info})")

    refs, ref_cfgs = [], []
    for d in a.reference_run:
        rows, cfg = load_reference(Path(d))
        refs.append(rows)
        ref_cfgs.append(cfg)
    base_cfg = ref_cfgs[0] if ref_cfgs else None
    if a.config:
        from rag.config import load_config

        base_cfg = base_cfg or load_config(a.config)
    if base_cfg is None and not (a.dataset and a.docs_path):
        raise SystemExit("--reference-run, --config 또는 --dataset/--docs-path 가 필요")
    dataset = Path(a.dataset or base_cfg["dataset"]["path"])
    docs_path = Path(a.docs_path or base_cfg["retriever"]["docs_path"])
    examples = {str(r["id"]): r for r in read_jsonl(dataset)}
    chunks = {str(r["id"]): r for r in read_jsonl(docs_path)}

    live_rows = None
    if a.live_check:
        from rag.config import load_config
        from rag.retrieval.faiss_retriever import FaissRetriever

        lc = load_config(a.config) if a.config else base_cfg
        r = lc["retriever"]
        retr = FaissRetriever(
            r["index_path"],
            r["docs_path"],
            lc["embedding"]["model"],
            r.get("query_prefix", ""),
            r.get("device"),
        )
        live_rows = {}
        for qid in ids:
            if qid not in examples:
                continue
            docs = retr.retrieve(examples[qid]["question"], a.top_k)
            live_rows[qid] = {
                "question": examples[qid]["question"],
                "retrieval_doc_ids": [d.id for d in docs],
                "extra": {"retrieval_scores": [d.score for d in docs]},
            }
    sources = refs + ([live_rows] if live_rows is not None else [])
    if not sources:
        raise SystemExit("--reference-run 또는 --live-check 중 하나는 필요")
    items, info = build_frozen_items(ids, examples, sources, chunks, a.top_k)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        for it in items:
            row = it.to_dict()
            row["source"] = {
                "reference_run": a.reference_run[0] if a.reference_run else "live_faiss"
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "sha256": file_sha256(out),
        "n": len(items),
        "top_k": a.top_k,
        "question_ids_file": ids_source,
        "question_ids_from_runs": ids_from_run_info,
        "question_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "order": "question_ids 파일 순서",
        "reference_runs": a.reference_run,
        "live_check": a.live_check,
        "dataset": {"path": str(dataset), "sha256": file_sha256(dataset)},
        "docs_path": {"path": str(docs_path), "sha256": file_sha256(docs_path)},
        "retriever": (base_cfg or {}).get("retriever"),
        "embedding": (base_cfg or {}).get("embedding"),
        "git": git_info(),
        "caveat": f"{len(items)}문항은 기존 실험의 공통 문항 집합 (전체 PopQA 1,399 일반화 아님)",
        **info,
    }
    write_new(manifest_path_for(out), json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"saved -> {out} ({len(items)} 문항)")


if __name__ == "__main__":
    main()
