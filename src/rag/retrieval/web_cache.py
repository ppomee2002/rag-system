"""Web search 기록 / 재생 (외부 서비스 지연 분리용).

- RecordingWebSearch : 실제 검색 (Tavily) 을 호출하고 (query, 문서, 지연) 을 jsonl 에 남긴다.
                       정확도 실험은 이 모드 (실제 correction) 로 한다.
- ReplayWebSearch    : 기록된 문서를 그대로 돌려준다. latency_mode
                         recorded : 기록된 지연만큼 기다림 (실제 지연 분포 유지, 외부 변동 제거)
                         zero     : 기다리지 않음 (외부 지연 기여 제거)

재생 키 (key_mode)
  query    : rewrite 된 검색어 문자열이 기록과 정확히 같아야 한다 (기존 동작).
             동시 실행 (batch 구성 변화) 등으로 rewrite 출력이 달라지면 miss.
  query_id : 기준 run 의 queries.jsonl 에서 query_id -> 그때의 rewrite 검색어 를 찾아 그 기록을
             돌려준다. 이번 실행의 rewrite 가 달라도 같은 질문은 같은 검색 결과를 받는다
             (조건 비교용).
             요청마다 bind(query_id) 한 객체로 검색해야 한다. 일치 여부는 rewrite_matched 로 기록.
miss 처리 (on_miss)
  error : KeyError (기본)
  empty : 검색 결과 0개로 진행하고 miss 로 기록 (실제 검색 없이 모든 조건에 같은 규칙 적용)
  fallback 이 주어지면 실제 검색 (runner 의 replay_fallback=true 일 때만)

같은 문서로 latency_mode 만 바꿔 돌리면 E2E 에서 외부 지연의 기여를 분리할 수 있다.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from rag.retrieval.base import WebSearch
from rag.types import Document


def _doc_to_dict(d: Document) -> dict:
    return {"id": d.id, "text": d.text, "metadata": d.metadata, "score": d.score}


def _key(query: str, top_k: int) -> str:
    return f"{top_k}\t{query.strip()}"


class RecordingWebSearch:
    def __init__(self, inner: WebSearch, path: str | Path):
        self.inner = inner
        self.path = Path(path)
        self.lock = threading.Lock()

    def search(self, query: str, top_k: int) -> list[Document]:
        t0 = time.perf_counter()
        docs = self.inner.search(query, top_k)
        latency = time.perf_counter() - t0
        row = {
            "query": query.strip(),
            "top_k": top_k,
            "latency": latency,
            "docs": [_doc_to_dict(d) for d in docs],
        }
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return docs


def read_jsonl(path: str | Path) -> list[dict]:
    """JSON Lines 를 줄바꿈 기준으로 읽는다.

    str.splitlines 는 U+2028 등에서도 나눠 ensure_ascii=False 로 기록된 레코드를 깨뜨린다.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: JSON 파싱 실패: {e}") from e
    return rows


def load_replay_index(run_dirs: list[str | Path]) -> dict[str, dict]:
    """기준 run 들의 queries.jsonl 에서 query_id -> {query (기록된 rewrite), run}.

    여러 run 이면 먼저 나온 run 의 기록을 쓴다.

    warmup 행도 포함한다 (같은 warmup 질문을 쓰는 다음 run 의 warmup 이 재생되도록). 오류 행은 제외.
    """
    index: dict[str, dict] = {}
    for d in run_dirs:
        for r in read_jsonl(Path(d) / "queries.jsonl"):
            q = (r.get("extra") or {}).get("rewritten_query")
            if r.get("error") or not q:
                continue
            index.setdefault(str(r["query_id"]), {"query": q.strip(), "run": str(d)})
    return index


class ReplayWebSearch:
    def __init__(
        self,
        paths: list[str | Path],
        latency_mode: str = "recorded",
        fallback: WebSearch | None = None,
        sleep=time.sleep,
        key_mode: str = "query",
        on_miss: str = "error",
        query_index: dict[str, dict] | None = None,
    ):
        if latency_mode not in ("recorded", "zero"):
            raise ValueError("latency_mode: recorded | zero")
        if key_mode not in ("query", "query_id"):
            raise ValueError("key_mode: query | query_id")
        if on_miss not in ("error", "empty"):
            raise ValueError("on_miss: error | empty")
        if key_mode == "query_id" and not query_index:
            raise ValueError("key_mode=query_id 에는 기준 run 의 query_index 가 필요")
        self.latency_mode = latency_mode
        self.fallback = fallback
        self.sleep = sleep
        self.key_mode = key_mode
        self.on_miss = on_miss
        self.query_index = query_index or {}
        self.cache: dict[str, dict] = {}
        for p in paths:
            for row in read_jsonl(p):
                self.cache.setdefault(_key(row["query"], row["top_k"]), row)  # 첫 기록 사용
        self.hits = 0
        self.misses: list[str] = []
        self.empty_returned = 0
        self.rewrite_mismatch: list[dict] = []
        self.lock = threading.Lock()

    def bind(self, query_id: str) -> BoundReplaySearch:
        """요청 하나 (query_id) 에 묶인 검색 객체. runner 가 요청마다 만든다."""
        return BoundReplaySearch(self, str(query_id))

    def search(self, query: str, top_k: int) -> list[Document]:
        return self._lookup(query, top_k, None, {})

    def _lookup(self, query: str, top_k: int, query_id: str | None, info: dict) -> list[Document]:
        query = query.strip()
        row = None
        info.update(query=query, query_id=query_id, key_mode=self.key_mode)
        if self.key_mode == "query_id" and query_id is not None:
            rec = self.query_index.get(query_id)
            info["recorded_query"] = rec["query"] if rec else None
            if rec is not None:
                row = self.cache.get(_key(rec["query"], top_k))
                matched = rec["query"] == query
                info["rewrite_matched"] = matched
                if row is not None and not matched:
                    with self.lock:
                        self.rewrite_mismatch.append(
                            {"query_id": query_id, "recorded": rec["query"], "current": query}
                        )
        else:
            row = self.cache.get(_key(query, top_k))
        if row is None:
            info["hit"] = False
            with self.lock:
                self.misses.append(f"{query_id}: {query}" if query_id else query)
            if self.fallback is not None:
                info["source"] = "fallback"
                return self.fallback.search(query, top_k)
            if self.on_miss == "empty":
                info["source"] = "empty"
                with self.lock:
                    self.empty_returned += 1
                return []
            raise KeyError(f"web cache miss (fallback 없음): {query!r} (query_id={query_id})")
        info.update(hit=True, source="replay")
        with self.lock:
            self.hits += 1
        if self.latency_mode == "recorded":
            self.sleep(row["latency"])
        return [
            Document(
                id=d["id"], text=d["text"], metadata=d.get("metadata") or {}, score=d.get("score")
            )
            for d in row["docs"]
        ]

    def stats(self) -> dict:
        return {
            "key_mode": self.key_mode,
            "on_miss": self.on_miss,
            "hits": self.hits,
            "misses": len(self.misses),
            "miss_queries": self.misses[:50],
            "empty_returned": self.empty_returned,
            "rewrite_mismatch": len(self.rewrite_mismatch),
            "rewrite_mismatch_examples": self.rewrite_mismatch[:20],
            "latency_mode": self.latency_mode,
        }


class BoundReplaySearch:
    """ReplayWebSearch 를 요청 (query_id) 에 묶은 WebSearch. last 에 마지막 조회 정보를 남긴다."""

    def __init__(self, replay: ReplayWebSearch, query_id: str):
        self.replay = replay
        self.query_id = query_id
        self.last: dict = {}

    def search(self, query: str, top_k: int) -> list[Document]:
        self.last = {}
        return self.replay._lookup(query, top_k, self.query_id, self.last)
