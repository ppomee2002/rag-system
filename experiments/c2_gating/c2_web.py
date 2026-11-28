"""MODE A 용 web 기록 재생 (같은 web evidence + 기록된 지연, 취소 가능).

기록 (`prefetch_web.py`): rag.retrieval.web_cache.RecordingWebSearch 의 web_cache.v2 행
(query_id = MODE A index, Tavily include_raw_content=True -> raw_response 에 페이지 원문).

재생: query_id 로 찾는다 (이번 실행의 rewrite 출력이 기록과 달라도 같은 질문은 같은 결과 ->
조건 간 web evidence 동일, 일치 여부는 rewrite_matched 로 기록). 지연은 `cancel.wait(latency)` 로
흉내내므로 speculative web 을 취소하면 즉시 끝난다 (실제 외부 호출을 중간에 끊는 것과 같은 효과로
간주. 외부 API 비용은 이미 발생했다고 보고 '낭비 web 시간' 에 기록).

외부 refinement 입력 (공식 external_knowledge_preparation.py): 검색 상위 결과 중 wikipedia 페이지의
문단 ("제목: 문단", 10자 초과). 공식은 serper + HTML <p>, 여기서는 Tavily raw_content 의 줄
  (deviation).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from rag.retrieval.web_cache import read_jsonl
from rag.types import Document


@dataclass
class WebResult:
    status: str  # ok | empty | miss | cancelled
    docs: list[Document] = field(default_factory=list)  # 문단 소스 (페이지 원문)
    snippet_docs: list[Document] = field(default_factory=list)  # 문단이 없을 때 공식 fallback
    latency: float = 0.0
    recorded_query: str | None = None
    rewrite_matched: bool | None = None
    n_results: int = 0
    n_kept: int = 0


class WebReplay:
    def __init__(
        self,
        paths: list[str | Path],
        latency_mode: str = "recorded",
        wikipedia_only: bool = True,
        top_k: int = 5,
        latency_scale: float = 1.0,
    ):
        if latency_mode not in ("recorded", "zero"):
            raise ValueError("web.latency_mode: recorded | zero")
        self.latency_mode = latency_mode
        self.wikipedia_only = wikipedia_only
        self.top_k = top_k
        self.latency_scale = latency_scale
        self.paths = [str(p) for p in paths]
        self.rows: dict[str, dict] = {}
        for p in self.paths:
            for r in read_jsonl(p):
                if r.get("query_id") is not None:
                    self.rows.setdefault(str(r["query_id"]), r)  # 첫 기록
        self.lock = threading.Lock()
        self.hits = 0
        self.misses: list[str] = []

    def describe(self) -> dict:
        return {
            "paths": self.paths,
            "n_records": len(self.rows),
            "latency_mode": self.latency_mode,
            "latency_scale": self.latency_scale,
            "wikipedia_only": self.wikipedia_only,
            "top_k": self.top_k,
            "key": "query_id (MODE A index)",
        }

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": len(self.misses), "miss_ids": self.misses[:50]}

    def recorded_latency(self, query_id: str) -> float | None:
        r = self.rows.get(str(query_id))
        return None if r is None else float(r["latency"])

    def lookup(self, query_id: str, query: str, cancel: threading.Event) -> WebResult:
        row = self.rows.get(str(query_id))
        if row is None:
            with self.lock:
                self.misses.append(str(query_id))
            return WebResult("miss")
        with self.lock:
            self.hits += 1
        latency = float(row["latency"]) * self.latency_scale
        if self.latency_mode == "recorded" and latency > 0 and cancel.wait(latency):
            return WebResult("cancelled", latency=latency)
        if cancel.is_set():
            return WebResult("cancelled", latency=latency)
        docs, snippets, n = page_docs(row, self.top_k, self.wikipedia_only)
        return WebResult(
            "ok" if (docs or snippets) else "empty",
            docs,
            snippets,
            latency,
            row.get("query"),
            (row.get("query") or "").strip() == query.strip(),
            n,
            len(docs),
        )


def page_docs(row: dict, top_k: int, wikipedia_only: bool):
    """web_cache.v2 행 -> (문단용 페이지 문서, snippet 문서, 결과 수)."""
    raw = (row.get("raw_response") or {}).get("results")
    if raw is None:  # raw_response 없는 기록: docs (content) 만
        raw = [
            {
                "url": d["id"],
                "title": (d.get("metadata") or {}).get("title", ""),
                "content": d["text"],
            }
            for d in row.get("docs") or []
        ]
    raw = raw[:top_k]
    pages, snippets = [], []
    for r in raw:
        url = r.get("url") or ""
        meta = {"title": r.get("title") or "", "url": url, "source": "web"}
        snippets.append(Document(url, r.get("content") or "", meta))
        if wikipedia_only and "wikipedia.org" not in url:
            continue
        body = r.get("raw_content") or r.get("content") or ""
        if body:
            pages.append(Document(url, body, meta))
    return pages, snippets, len(raw)
