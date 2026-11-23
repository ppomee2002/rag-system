"""MODE B 고정 입력 만들기: 문항 ID 목록 + 기존 run 의 PopQA Top-5 (문서와 순서) 고정.

    uv run python experiments/official_crag/freeze_inputs.py \
        --question-ids <230문항 id 파일> \
        --reference-run results/crag_system/smoke/SMOKE-A1-FULL-RETRY/20261110_111726 \
        [--reference-run <다른 run: Top-5 일치 검사용>] \
        [--live-check --config experiments/crag_system/config.yaml]   # FAISS 로 다시 검색해 대조
        --out results/official_crag/inputs/popqa_c1_230.jsonl

검사 (하나라도 실패하면 파일을 만들지 않음):
  질문·정답이 dataset 과 같음, Top-5 존재, 기준 run 간 Top-5 id·순서 일치,
  (--live-check) 현재 FAISS 검색 결과와 일치, 문서 텍스트 존재.
--reference-run 없이 --live-check 만 주면 현재 FAISS 검색 결과로 고정한다
(기존 결과와 대조 불가로 manifest 에 표시).

230문항 ID 목록은 이 레포에 없다 (서버 결과에서 사용자가 지정). 목록 파일의 sha256 을 manifest 에
남긴다.
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
    ap.add_argument("--question-ids", required=True)
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
    ids_path = Path(a.question_ids)
    ids = read_ids(ids_path)
    if len(set(ids)) != len(ids):
        raise SystemExit("문항 ID 목록에 중복이 있음")

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
        "question_ids_file": str(ids_path),
        "question_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "order": "question_ids 파일 순서",
        "reference_runs": a.reference_run,
        "live_check": a.live_check,
        "dataset": {"path": str(dataset), "sha256": file_sha256(dataset)},
        "docs_path": {"path": str(docs_path), "sha256": file_sha256(docs_path)},
        "retriever": (base_cfg or {}).get("retriever"),
        "embedding": (base_cfg or {}).get("embedding"),
        "git": git_info(),
        "caveat": "230문항은 기존 공통 문항 집합으로 전체 PopQA 일반화 결과가 아님",
        **info,
    }
    write_new(manifest_path_for(out), json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"saved -> {out} ({len(items)} 문항)")


if __name__ == "__main__":
    main()
