"""MODE A 입력 (공식 PopQA 1,399문항 + Contriever top-10 + 저자 지식 파일) 을 요청 단위로 묶는다.

MODE A (`experiments/official_crag/mode_a.py`) 와 같은 파일·같은 읽기 함수를 쓴다:
  grading 입력       load_official_queries (test_popqa.txt, 질문당 10문서, 빈 문서 "")
  refinement 입력    official_passages_from_line(retrieved_psgs_line(passages))  (verify-refinement
    와 같음)
  evidence           ref/{correct,incorrect,ambiguous} 의 같은 줄
                     (load_ref, '#' 접두 유지 = MODE A generate)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rag.crag_official import source
from rag.crag_official.official_data import (
    attach_answers,
    load_official_queries,
    load_ref,
    retrieved_psgs_line,
)
from rag.crag_official.refinement import official_passages_from_line

ACTIONS = ("correct", "ambiguous", "incorrect")


@dataclass
class ModeAItem:
    index: int
    question: str
    passages: list[str]
    refine_passages: list[str]
    ref: dict[str, str]
    answers: list[str] = field(default_factory=list)
    ext_id: str | None = None

    @property
    def query_id(self) -> str:
        return str(self.index)


def official_paths(repo: str | Path) -> dict[str, Path]:
    d = Path(repo) / "data" / "popqa"
    return {
        "test": d / "test_popqa.txt",
        "sources": d / "sources",
        "correct": d / "ref" / "correct",
        "incorrect": d / "ref" / "incorrect",
        "ambiguous": d / "ref" / "ambiguous",
    }


def load_items(repo: str | Path, answers: str | Path | None = None) -> tuple[list[ModeAItem], dict]:
    paths = official_paths(repo)
    queries = load_official_queries(paths["test"], source.OFFICIAL_NDOCS, paths["sources"])
    align = attach_answers(queries, answers) if answers else None
    refs = {k: load_ref(paths[k]) for k in ACTIONS}
    for k, v in refs.items():
        if len(v) != len(queries):
            raise ValueError(f"ref/{k} {len(v)} 줄 != 질문 {len(queries)}")
    items = [
        ModeAItem(
            index=q.index,
            question=q.question,
            passages=q.passages,
            refine_passages=official_passages_from_line(retrieved_psgs_line(q.passages)),
            ref={k: refs[k][q.index] for k in ACTIONS},
            answers=q.answers,
            ext_id=q.ext_id,
        )
        for q in queries
    ]
    return items, {"paths": {k: str(v) for k, v in paths.items()}, "answers": align}


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reference_actions(score_run: str | Path) -> dict[int, dict]:
    """MODE A score run (scores.jsonl) -> index -> {action, scores}."""
    return {
        int(r["index"]): {"action": r["action"], "scores": r["scores"]}
        for r in read_jsonl(Path(score_run) / "scores.jsonl")
    }


def reference_preds(generate_run: str | Path) -> dict[int, dict]:
    """MODE A generate run (preds.jsonl) -> index -> {prediction, action, prompt_version}."""
    return {int(r["index"]): r for r in read_jsonl(Path(generate_run) / "preds.jsonl")}


def select_indices(
    items: list[ModeAItem],
    *,
    indices_file: str | Path | list | None = None,
    limit: int | None = None,
    per_action: int | None = None,
    ref_actions: dict[int, dict] | None = None,
    seed: int = 0,
) -> tuple[list[int], dict]:
    """질문 집합 선택. 우선순위: indices_file (여러 개면 합집합, 순서 유지) > per_action
    (참조 판정으로 층화) > limit (앞에서부터)."""
    files = [indices_file] if isinstance(indices_file, (str, Path)) else (indices_file or [])
    if files:
        idx = []
        for fp in files:
            text = Path(fp).read_text(encoding="utf-8")
            idx += [int(x) for x in text.replace(",", " ").split() if x.strip()]
        idx = list(dict.fromkeys(idx))
        how = {"indices_file": [str(x) for x in files]}
    elif per_action:
        if not ref_actions:
            raise ValueError("per_action 선택에는 MODE A score run (--reference-scores) 이 필요")
        rng = random.Random(seed)
        idx = []
        for a in ACTIONS:
            pool = sorted(i for i, r in ref_actions.items() if r["action"] == a)
            idx += sorted(rng.sample(pool, min(per_action, len(pool))))
        how = {"per_action": per_action, "seed": seed}
    else:
        idx = [it.index for it in items[: limit or len(items)]]
        how = {"limit": limit}
    bad = [i for i in idx if not 0 <= i < len(items)]
    if bad:
        raise ValueError(f"범위 밖 index: {bad[:5]}")
    if len(set(idx)) != len(idx):
        raise ValueError("중복 index")
    dist = Counter(ref_actions[i]["action"] for i in idx if i in ref_actions) if ref_actions else {}
    return idx, {**how, "n": len(idx), "reference_action_counts": dict(dist)}


def warmup_pool(n_items: int, seed: int) -> list[int]:
    """warmup 후보 순서 (seed 고정 순열). run.py 는 측정 집합에 없고 web 기록이 있는 앞쪽 질문을,
    prefetch_web.py 는 앞쪽 몇 개를 항상 기록한다 -> 측정 집합이 달라도 warmup 이 같은 규칙."""
    perm = list(range(n_items))
    random.Random(seed * 31 + 7).shuffle(perm)
    return perm
