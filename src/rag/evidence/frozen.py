"""고정 평가 입력 (MODE B): 질문·정답·검색 Top-k 문서와 순서를 파일 하나로 고정한다.

frozen_inputs.jsonl  한 줄 = {question_id, question, answers, s_pop, docs: [{rank, id, text, score,
                     metadata}], source: {reference_run, ...}}
*.manifest.json      sha256, 문항 수, top_k, 기준 run, 문항 ID 목록 파일 hash, 생성 commit

모든 MODE B 조건 (grader / 정책) 은 같은 파일을 읽으므로 검색 결과가 조건 간에 같다.
파일 hash 는 결과 run_meta 에 기록된다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from rag.types import Document


@dataclass
class FrozenItem:
    question_id: str
    question: str
    answers: list[str]
    docs: list[Document]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answers": self.answers,
            "docs": [
                {"rank": i, "id": d.id, "text": d.text, "score": d.score, "metadata": d.metadata}
                for i, d in enumerate(self.docs)
            ],
            **self.metadata,
        }


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen(path: str | Path, verify_manifest: bool = True) -> tuple[list[FrozenItem], dict]:
    path = Path(path)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            docs = [
                Document(
                    id=str(d["id"]),
                    text=d["text"],
                    metadata=d.get("metadata") or {},
                    score=d.get("score"),
                )
                for d in sorted(row.pop("docs"), key=lambda d: d["rank"])
            ]
            items.append(
                FrozenItem(
                    question_id=str(row.pop("question_id")),
                    question=row.pop("question"),
                    answers=list(row.pop("answers")),
                    docs=docs,
                    metadata=row,
                )
            )
    ids = [it.question_id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: 중복 question_id")
    sha = file_sha256(path)
    manifest_path = manifest_path_for(path)
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if verify_manifest and manifest.get("sha256") != sha:
            raise ValueError(f"{path}: manifest sha256 불일치 (파일이 바뀜)")
    elif verify_manifest:
        raise FileNotFoundError(f"manifest 없음: {manifest_path}")
    return items, {"path": str(path), "sha256": sha, "n": len(items), "manifest": manifest}


def manifest_path_for(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".manifest.json")


def build_frozen_items(
    question_ids: list[str],
    examples: dict[str, dict],
    reference_rows: list[dict[str, dict]],
    chunk_texts: dict[str, dict],
    top_k: int,
) -> tuple[list[FrozenItem], dict]:
    """기준 run 들의 queries 행으로 Top-k 를 고정한다.

    examples       question_id -> dataset 행 {question, answers, ...}
    reference_rows run 마다 question_id -> queries.jsonl 행 (첫 run 이 기준, 나머지는 일치 검사)
    chunk_texts    문서 id -> {"text", ...} (retriever.docs_path)
    검사 (하나라도 실패하면 ValueError): 질문 존재, 기준 run 의 질문 문자열·정답이 dataset 과 같음,
    검색 문서 >= top_k, 모든 기준 run 에서 같은 Top-k id·순서, 문서 텍스트 존재.
    """
    problems: list[str] = []
    items: list[FrozenItem] = []
    ref = reference_rows[0]
    for qid in question_ids:
        ex = examples.get(qid)
        row = ref.get(qid)
        if ex is None:
            problems.append(f"{qid}: dataset 에 없음")
            continue
        if row is None:
            problems.append(f"{qid}: 기준 run 에 없음")
            continue
        if row.get("question") not in (None, ex["question"]):
            problems.append(f"{qid}: 질문 문자열 불일치")
        gold = row.get("gold_answers")
        if gold is not None and list(gold) != list(ex["answers"]):
            problems.append(f"{qid}: 정답 불일치")
        ids = [str(x) for x in (row.get("retrieval_doc_ids") or [])]
        if len(ids) < top_k:
            problems.append(f"{qid}: 검색 문서 {len(ids)} < top_k {top_k}")
            continue
        ids = ids[:top_k]
        for k, other in enumerate(reference_rows[1:], 1):
            o = other.get(qid)
            o_ids = [str(x) for x in ((o or {}).get("retrieval_doc_ids") or [])][:top_k]
            if o is None or o_ids != ids:
                problems.append(f"{qid}: 기준 run #{k} 의 Top-{top_k} 가 다름")
        scores = ((row.get("extra") or {}).get("retrieval_scores") or [None] * top_k)[:top_k]
        docs = []
        for i, s in zip(ids, scores + [None] * (top_k - len(scores)), strict=False):
            c = chunk_texts.get(i)
            if c is None:
                problems.append(f"{qid}: 문서 {i} 텍스트 없음")
                continue
            meta = {k: v for k, v in c.items() if k not in ("id", "text")}
            docs.append(Document(id=i, text=c["text"], metadata=meta, score=s))
        items.append(
            FrozenItem(
                qid,
                ex["question"],
                list(ex["answers"]),
                docs,
                {"s_pop": ex.get("s_pop"), "prop": ex.get("prop")},
            )
        )
    if problems:
        raise ValueError(f"고정 입력 검사 실패 {len(problems)}건: " + "; ".join(problems[:20]))
    return items, {"n": len(items), "top_k": top_k}
