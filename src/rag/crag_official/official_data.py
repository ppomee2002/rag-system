"""공식 CRAG 저장소 데이터 파일 읽기 (MODE A).

data/popqa/test_popqa.txt   "질문 [SEP] 문서" 한 줄씩, 질문당 정확히 10줄 (Self-RAG 의 Contriever
                            검색 결과 ctxs[:10]). 빈 문서는 "질문 [SEP]" 로 끝난다.
data/popqa/sources          질문 1,399 줄
data/popqa/ref/{correct,incorrect,ambiguous}
                            저자들이 미리 만든 지식 (correct/incorrect 는 줄마다 '#' 접두).
                            공식 inference 는 판정에 따라 이 파일의 같은 줄을 고른다.
정답 파일                   공식 eval 은 Self-RAG eval_data/popqa_longtail_w_gs.jsonl 의
                            answers 를 쓴다. 저장소에 없으므로 사용자가 내려받거나, 이 레포
                            prepare_data.py 결과 (HF akariasai/PopQA s_pop<100) 를
                            위치·질문 문자열 일치 검사 후 사용한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rag.crag_official import source


@dataclass
class OfficialQuery:
    index: int
    question: str
    passages: list[str]  # 검색 순서, 빈 문서는 ""
    answers: list[str] = field(default_factory=list)
    ext_id: str | None = None  # 정답 파일의 id (예: PopQA id)


def _read_lines(path: str | Path) -> list[str]:
    # 원본은 readlines() 후 strip. 마지막 줄 개행 없음 -> split("\n")
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def parse_pair_line(line: str) -> tuple[str, str]:
    """CRAG_Inference.py:116-120 과 같은 분리."""
    c = line.strip()
    if c.endswith("[SEP]"):
        c += " "
    q, p = c.split(source.PAIR_SEPARATOR)
    return q, p


def load_official_queries(
    test_file: str | Path,
    n_docs: int = source.OFFICIAL_NDOCS,
    sources_file: str | Path | None = None,
) -> list[OfficialQuery]:
    """질문당 n_docs 줄씩 위치로 묶는다 (공식 inference 의 n % 10 가정). 질문 문자열이 묶음 안에서
    다르거나 sources 와 다르면 실패한다."""
    lines = _read_lines(test_file)
    if len(lines) % n_docs:
        raise ValueError(f"{test_file}: {len(lines)} 줄은 {n_docs} 의 배수가 아님")
    out = []
    for i in range(0, len(lines), n_docs):
        pairs = [parse_pair_line(x) for x in lines[i : i + n_docs]]
        qs = {q for q, _ in pairs}
        if len(qs) != 1:
            raise ValueError(f"{test_file}: {i // n_docs} 번째 묶음의 질문이 하나가 아님: {qs}")
        out.append(OfficialQuery(i // n_docs, pairs[0][0], [p for _, p in pairs]))
    if sources_file is not None:
        src = _read_lines(sources_file)
        if len(src) != len(out):
            raise ValueError(f"sources {len(src)} 줄 != 질문 {len(out)} 개")
        bad = [q.index for q, s in zip(out, src, strict=True) if q.question != s.strip()]
        if bad:
            raise ValueError(f"sources 와 질문 불일치: {bad[:5]}")
    return out


def load_ref(path: str | Path) -> list[str]:
    """CRAG_Inference.py:246-248 과 같이 strip() 만 ('#' 접두 유지)."""
    return [x.strip() for x in _read_lines(path)]


def retrieved_psgs_line(passages: list[str]) -> str:
    """data_process.py:97 의 retrieved_psgs 한 줄 재구성."""
    return " [sep] ".join(passages)


def _answers_of(row: dict) -> list[str]:
    a = row.get("answers")
    if a is None and "possible_answers" in row:
        a = row["possible_answers"]
        a = json.loads(a) if isinstance(a, str) else a
    if a is None and "answer" in row:
        a = row["answer"]
    if isinstance(a, str):
        a = [a]
    if not isinstance(a, list):
        raise ValueError(f"정답 필드 없음: {list(row)[:10]}")
    return [str(x) for x in a]


def attach_answers(queries: list[OfficialQuery], answers_file: str | Path) -> dict:
    """정답 jsonl 을 위치로 붙인다. 질문 문자열이 모두 같아야 한다 (하나라도 다르면 실패)."""
    rows = []
    with open(answers_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != len(queries):
        raise ValueError(f"정답 파일 {len(rows)} 행 != 질문 {len(queries)} 개")
    mismatch = [
        q.index for q, r in zip(queries, rows, strict=True) if r.get("question") != q.question
    ]
    if mismatch:
        raise ValueError(f"정답 파일과 질문 순서/문자열 불일치 {len(mismatch)}개: {mismatch[:5]}")
    for q, r in zip(queries, rows, strict=True):
        q.answers = _answers_of(r)
        q.ext_id = str(r["id"]) if "id" in r else None
    return {"answers_file": str(answers_file), "n": len(rows), "aligned_by": "position+question"}
