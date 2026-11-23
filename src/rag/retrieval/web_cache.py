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

  query_strict : (query_id, 실제 검색어, top_k) 가 모두 같고 기록의 검색 조건 (provider, params,
             query_mode) 이 현재 조건과 같을 때만 hit. 같은 query_id 라도 검색어가 다르면 miss
             (서로 다른 correction query 를 같은 캐시로 취급하지 않는다). C1/C2/C3 공유 캐시용.
             v1 (cache_version 없는) 기록은 accept_legacy=True 일 때만, 검색어 문자열 + top_k 가
             같고 현재 query_mode 가 legacy_rewrite 일 때 쓴다 (compatibility=legacy_assumed).

같은 문서로 latency_mode 만 바꿔 돌리면 E2E 에서 외부 지연의 기여를 분리할 수 있다.

기록 형식 (CACHE_VERSION = web_cache.v2. v1 필드에 추가만 함 -> 기존 재생 코드와 호환)
  v1: query, top_k, latency, docs
  v2: + cache_version, query_id, provider, params (비밀정보 제거), query_mode,
        searched_at_utc, n_docs, raw_response (provider 원본 응답, 있으면)
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from rag.retrieval.base import WebSearch
from rag.types import Document

CACHE_VERSION = "web_cache.v2"
LEGACY_QUERY_MODE = "legacy_rewrite"  # v1 기록 (crag_system REWRITE prompt) 의 검색어 생성 방식
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")


def scrub_secrets(obj):
    """dict 키에 비밀 표시가 있으면 값을 지운다 (API key / token 이 캐시·로그에 남지 않게)."""
    if isinstance(obj, dict):
        return {
            k: "<redacted>"
            if any(m in str(k).upper() for m in SECRET_MARKERS)
            else scrub_secrets(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [scrub_secrets(v) for v in obj]
    return obj


def search_signature(provider: str | None, params: dict | None, query_mode: str | None) -> str:
    """검색 조건 비교 키 (provider + params (top_k 제외) + 검색어 생성 방식)."""
    params = {k: v for k, v in scrub_secrets(params or {}).items() if k != "top_k"}
    return json.dumps(
        {"provider": provider, "params": params, "query_mode": query_mode},
        sort_keys=True,
        ensure_ascii=False,
    )


def _doc_to_dict(d: Document) -> dict:
    return {"id": d.id, "text": d.text, "metadata": d.metadata, "score": d.score}


def _key(query: str, top_k: int) -> str:
    return f"{top_k}\t{query.strip()}"


class RecordingWebSearch:
    """실제 검색 + 기록. provider / params / query_mode 는 기록에 남는 검색 조건 (선택).

    inner 에 search_raw(query, top_k) -> (docs, raw_response) 가 있으면 원본 응답도 남긴다.
    요청 (query_id) 별로는 bind(query_id) 한 객체로 검색한다 (기록에 query_id 가 남음).
    """

    def __init__(
        self,
        inner: WebSearch,
        path: str | Path,
        provider: str | None = None,
        params: dict | None = None,
        query_mode: str | None = None,
    ):
        self.inner = inner
        self.path = Path(path)
        self.provider = provider
        self.params = scrub_secrets(params or {})
        self.query_mode = query_mode
        self.lock = threading.Lock()
        self.n_recorded = 0

    def bind(self, query_id: str) -> BoundRecordingSearch:
        return BoundRecordingSearch(self, str(query_id))

    def search(self, query: str, top_k: int) -> list[Document]:
        return self._search(query, top_k, None, {})

    def _search(self, query: str, top_k: int, query_id: str | None, info: dict) -> list[Document]:
        searched_at = datetime.now(UTC).isoformat()
        t0 = time.perf_counter()
        raw = None
        if hasattr(self.inner, "search_raw"):
            docs, raw = self.inner.search_raw(query, top_k)
        else:
            docs = self.inner.search(query, top_k)
        latency = time.perf_counter() - t0
        row = {
            "query": query.strip(),
            "top_k": top_k,
            "latency": latency,
            "docs": [_doc_to_dict(d) for d in docs],
            "cache_version": CACHE_VERSION,
            "query_id": query_id,
            "provider": self.provider,
            "params": {**self.params, "top_k": top_k},
            "query_mode": self.query_mode,
            "searched_at_utc": searched_at,
            "n_docs": len(docs),
            "raw_response": scrub_secrets(raw) if raw is not None else None,
        }
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.n_recorded += 1
        info.update(
            source="live",
            recorded=True,
            query_id=query_id,
            cache_version=CACHE_VERSION,
            searched_at_utc=searched_at,
            provider=self.provider,
        )
        return docs

    def stats(self) -> dict:
        return {
            "mode": "record",
            "path": str(self.path),
            "provider": self.provider,
            "params": self.params,
            "query_mode": self.query_mode,
            "n_recorded": self.n_recorded,
        }


class BoundRecordingSearch:
    """RecordingWebSearch 를 요청 (query_id) 에 묶은 WebSearch."""

    def __init__(self, rec: RecordingWebSearch, query_id: str):
        self.rec = rec
        self.query_id = query_id
        self.last: dict = {}

    def search(self, query: str, top_k: int) -> list[Document]:
        self.last = {}
        return self.rec._search(query, top_k, self.query_id, self.last)


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
        expect: dict | None = None,
        accept_legacy: bool = False,
    ):
        if latency_mode not in ("recorded", "zero"):
            raise ValueError("latency_mode: recorded | zero")
        if key_mode not in ("query", "query_id", "query_strict"):
            raise ValueError("key_mode: query | query_id | query_strict")
        if key_mode == "query_strict" and expect is None:
            raise ValueError(
                "key_mode=query_strict 에는 expect={provider, params, query_mode} 필요"
            )
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
        self.expect = expect
        self.accept_legacy = accept_legacy
        self.signature = (
            search_signature(expect.get("provider"), expect.get("params"), expect.get("query_mode"))
            if expect is not None
            else None
        )
        # query_strict 색인: (query_id, top_k, query) -> v2 기록 (검색 조건이 같은 것만)
        self.strict: dict[tuple, dict] = {}
        self.recorded_queries: dict[str, set] = {}  # query_id -> 기록된 검색어 (miss 원인 보고)
        self.incompatible: dict[str, int] = {}  # query_id -> 검색 조건이 다른 v2 기록 수
        self.n_legacy_records = 0
        self.paths = [str(p) for p in paths]
        for p in paths:
            for row in read_jsonl(p):
                self.cache.setdefault(_key(row["query"], row["top_k"]), row)  # 첫 기록 사용
                if row.get("cache_version"):
                    self._index_v2(row)
                else:
                    self.n_legacy_records += 1
        self.hits = 0
        self.legacy_hits = 0
        self.misses: list[str] = []
        self.empty_returned = 0
        self.rewrite_mismatch: list[dict] = []
        self.lock = threading.Lock()

    def _index_v2(self, row: dict) -> None:
        qid = row.get("query_id")
        if qid is None or self.signature is None:
            return
        qid = str(qid)
        sig = search_signature(row.get("provider"), row.get("params"), row.get("query_mode"))
        if sig != self.signature:
            self.incompatible[qid] = self.incompatible.get(qid, 0) + 1
            return
        self.recorded_queries.setdefault(qid, set()).add(row["query"].strip())
        self.strict.setdefault((qid, row["top_k"], row["query"].strip()), row)  # 첫 기록 사용

    def _strict_row(self, query: str, top_k: int, query_id: str | None, info: dict):
        row = self.strict.get((query_id, top_k, query)) if query_id is not None else None
        if row is not None:
            info.update(
                compatibility="exact",
                cache_version=row.get("cache_version"),
                searched_at_utc=row.get("searched_at_utc"),
            )
            return row
        legacy_ok = self.accept_legacy and (self.expect or {}).get("query_mode") in (
            None,
            LEGACY_QUERY_MODE,
        )
        if legacy_ok:
            row = self.cache.get(_key(query, top_k))
            if row is not None and not row.get("cache_version"):
                info.update(compatibility="legacy_assumed", cache_version="web_cache.v1")
                with self.lock:
                    self.legacy_hits += 1
                return row
        recorded = sorted(self.recorded_queries.get(query_id or "", ()))
        if recorded:  # 같은 query_id, 다른 검색어 -> 재사용하지 않음
            info.update(miss_reason="query_mismatch", recorded_queries=recorded[:5])
        elif self.incompatible.get(query_id or ""):
            info["miss_reason"] = "search_condition_mismatch"
        else:
            info["miss_reason"] = "not_recorded"
        return None

    def bind(self, query_id: str) -> BoundReplaySearch:
        """요청 하나 (query_id) 에 묶인 검색 객체. runner 가 요청마다 만든다."""
        return BoundReplaySearch(self, str(query_id))

    def search(self, query: str, top_k: int) -> list[Document]:
        return self._lookup(query, top_k, None, {})

    def _lookup(self, query: str, top_k: int, query_id: str | None, info: dict) -> list[Document]:
        query = query.strip()
        row = None
        info.update(query=query, query_id=query_id, key_mode=self.key_mode)
        if self.key_mode == "query_strict":
            row = self._strict_row(query, top_k, query_id, info)
        elif self.key_mode == "query_id" and query_id is not None:
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
                if query_id is not None and hasattr(self.fallback, "bind"):
                    return self.fallback.bind(query_id).search(query, top_k)  # 기록에 query_id
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
            "paths": self.paths,
            "expect": self.expect,
            "accept_legacy": self.accept_legacy,
            "n_strict_records": len(self.strict),
            "n_legacy_records": self.n_legacy_records,
            "legacy_hits": self.legacy_hits,
            "n_incompatible_records": sum(self.incompatible.values()),
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
