"""Web search 기록 / 재생 (외부 서비스 지연 분리용).

- RecordingWebSearch : 실제 검색 (Tavily) 을 호출하고 (query, 문서, 지연) 을 jsonl 에 남긴다.
                       정확도 실험은 이 모드 (실제 correction) 로 한다.
- ReplayWebSearch    : 기록된 문서를 그대로 돌려준다. latency_mode
                         recorded : 기록된 지연만큼 기다림 (실제 지연 분포 유지, 외부 변동 제거)
                         zero     : 기다리지 않음 (외부 지연 기여 제거)
                       기록에 없는 query 는 fallback (실제 검색) 을 호출하고 miss 로 기록한다.

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


class ReplayWebSearch:
    def __init__(
        self,
        paths: list[str | Path],
        latency_mode: str = "recorded",
        fallback: WebSearch | None = None,
        sleep=time.sleep,
    ):
        if latency_mode not in ("recorded", "zero"):
            raise ValueError("latency_mode: recorded | zero")
        self.latency_mode = latency_mode
        self.fallback = fallback
        self.sleep = sleep
        self.cache: dict[str, dict] = {}
        for p in paths:
            for line in Path(p).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache.setdefault(_key(row["query"], row["top_k"]), row)  # 첫 기록 사용
        self.hits = 0
        self.misses: list[str] = []
        self.lock = threading.Lock()

    def search(self, query: str, top_k: int) -> list[Document]:
        row = self.cache.get(_key(query, top_k))
        if row is None:
            with self.lock:
                self.misses.append(query.strip())
            if self.fallback is None:
                raise KeyError(f"web cache miss (fallback 없음): {query!r}")
            return self.fallback.search(query, top_k)
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
            "hits": self.hits,
            "misses": len(self.misses),
            "miss_queries": self.misses[:50],
            "latency_mode": self.latency_mode,
        }
