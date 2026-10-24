"""correction_scheduling 실험 공통 구성요소.

- QueryLLM      : query 하나 전용 LLM 래퍼. 공유 Ollama 서버로 보내며 호출 단위 trace
                  (대기 / 서버 처리 / 토큰) 를 기록하고, LLMGate 로 정책의 우선순위를 적용한다.
- LLMGate       : LLM 서버 앞의 client 측 우선순위 큐. 동시에 보낼 호출 수를 slots 로 제한한다.
- MockWebSearch : 실제 Tavily latency 분포를 재생하고 문서는 로컬 FAISS 로 대신하는 WebSearch.

core (src/rag) 는 수정하지 않고 Protocol 구현체만 추가한다.
pipeline 은 rag.pipelines.crag.CRAG 그대로 사용.
"""

import heapq
import itertools
import json
import random
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from rag.data.loader import load_qa_jsonl
from rag.llm.ollama_llm import OllamaLLM
from rag.pipelines import prompts
from rag.pipelines.crag import CRAG
from rag.pipelines.vanilla import VanillaRAG
from rag.retrieval.faiss_retriever import FaissRetriever
from rag.types import Document

KINDS = {
    "grade": prompts.GRADE.partition("{")[0],
    "rewrite": prompts.REWRITE.partition("{")[0],
    "generate": prompts.GENERATE.partition("{")[0],
}


def call_kind(prompt: str) -> str:
    """prompt template 으로 LLM 호출 종류(= C-RAG stage) 를 구분한다."""
    for kind, prefix in KINDS.items():
        if prompt.startswith(prefix):
            return kind
    return "other"


def load_examples(cfg: dict):
    """crag_latency/run.py 와 같은 seed shuffle + limit (같은 query 셋, 같은 warmup)."""
    examples = load_qa_jsonl(cfg["dataset"]["path"])
    random.Random(cfg["seed"]).shuffle(examples)
    return examples[: cfg["dataset"]["limit"]]


class SharedRetriever:
    """FAISS + bge encoder 를 여러 query thread 가 공유할 때 encoder 호출을 직렬화한다."""

    def __init__(self, inner: FaissRetriever):
        self.inner = inner
        self.lock = threading.Lock()

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        with self.lock:
            return self.inner.retrieve(query, top_k)


class MockWebSearch:
    """지연은 실제 Tavily 측정값에서 query 별로 고정 추출 (crc32 seed), 문서는 로컬 FAISS 검색."""

    def __init__(self, retriever: SharedRetriever, latencies: list[float]):
        self.retriever = retriever
        self.latencies = sorted(latencies)

    def latency_for(self, query: str) -> float:
        return random.Random(zlib.crc32(query.encode("utf-8"))).choice(self.latencies)

    def search(self, query: str, top_k: int) -> list[Document]:
        start = time.perf_counter()
        docs = self.retriever.retrieve(query, top_k)
        time.sleep(max(0.0, self.latency_for(query) - (time.perf_counter() - start)))
        return [
            Document(
                id=f"web:{d.id}",
                text=d.text,
                metadata={**d.metadata, "source": "web_mock"},
                score=d.score,
            )
            for d in docs
        ]


def load_web_latencies(paths: list[str]) -> list[float]:
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            w = json.loads(line)["stage_latencies"].get("web_search", 0.0)
            if w > 0:
                out.append(w)
    if not out:
        raise ValueError(f"web_search latency 가 없음: {paths}")
    return out


class LLMGate:
    """client 측 우선순위 큐. slots=None 이면 제한 없이 바로 보낸다 (Ollama 내부 큐, native)."""

    def __init__(self, slots: int | None):
        self.slots = slots
        self.busy = 0
        self.heap: list[tuple] = []
        self.seq = itertools.count()
        self.cv = threading.Condition()

    def acquire(self, priority: float) -> None:
        if self.slots is None:
            return
        with self.cv:
            ticket = (priority, next(self.seq))
            heapq.heappush(self.heap, ticket)
            while not (self.busy < self.slots and self.heap[0] == ticket):
                self.cv.wait()
            heapq.heappop(self.heap)
            self.busy += 1
            self.cv.notify_all()

    def release(self) -> None:
        if self.slots is None:
            return
        with self.cv:
            self.busy -= 1
            self.cv.notify_all()


@dataclass
class QueryState:
    """스케줄링 정책이 보는 query 단위 상태. 시각은 실험 t0 기준 초."""

    query_id: str
    arrival: float = 0.0
    s_pop: float | None = None
    retrieval_scores: list[float] = field(default_factory=list)
    retrieval_chars: list[int] = field(default_factory=list)
    n_issued: dict[str, int] = field(default_factory=lambda: {k: 0 for k in KINDS})
    calls: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class QueryRetriever:
    """최초 retrieval 결과(score, 문서 길이) 를 QueryState 에 남긴다 (정책의 예측 feature)."""

    def __init__(self, inner: SharedRetriever, state: QueryState):
        self.inner = inner
        self.state = state

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        docs = self.inner.retrieve(query, top_k)
        if not self.state.retrieval_scores:
            self.state.retrieval_scores = [d.score for d in docs]
            self.state.retrieval_chars = [len(d.text) for d in docs]
        return docs


class QueryLLM:
    """query 하나 전용 `LLM`. priority_fn(state, kind, prompt) 가 낮을수록 gate 에서 먼저 나간다."""

    def __init__(self, llm_cfg: dict, state: QueryState, gate: LLMGate, priority_fn, t0: float):
        self.kwargs = dict(
            model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 512),
            num_ctx=llm_cfg.get("num_ctx", 8192),
            host=llm_cfg.get("host", "http://127.0.0.1:11434"),
        )
        self.state = state
        self.gate = gate
        self.priority_fn = priority_fn
        self.t0 = t0

    def generate(self, prompt: str, **kwargs) -> str:
        kind = call_kind(prompt)
        with self.state.lock:
            idx = self.state.n_issued.get(kind, 0)
            self.state.n_issued[kind] = idx + 1
        box: dict = {}

        def post(payload: dict) -> dict:
            box["res"] = llm._http_post(payload)
            return box["res"]

        llm = OllamaLLM(**self.kwargs, post=post)
        priority = self.priority_fn(self.state, kind, prompt, idx)
        t_enq = time.perf_counter()
        self.gate.acquire(priority)
        t_start = time.perf_counter()
        try:
            out = llm.generate(prompt, **kwargs)
        finally:
            t_end = time.perf_counter()
            self.gate.release()
            res = box.get("res", {})
            ns = 1e-9
            call = {
                "kind": kind,
                "idx": idx,
                "priority": priority,
                "t_enqueue": t_enq - self.t0,
                "t_start": t_start - self.t0,
                "t_end": t_end - self.t0,
                "gate_wait": t_start - t_enq,
                "wall": t_end - t_start,
                "prompt_chars": len(prompt),
                "prompt_tokens": res.get("prompt_eval_count"),
                "out_tokens": res.get("eval_count"),
                # Ollama 서버 측 시간. wall - server_total = 서버 내부 큐 대기 + HTTP 오버헤드
                "server_total": res.get("total_duration", 0) * ns,
                "server_load": res.get("load_duration", 0) * ns,
                "server_prompt_eval": res.get("prompt_eval_duration", 0) * ns,
                "server_eval": res.get("eval_duration", 0) * ns,
            }
            with self.state.lock:
                self.state.calls.append(call)
        return out


class Components:
    """query 간에 공유하는 retriever / web search / gate. pipeline 은 query 마다 새로 만든다."""

    def __init__(self, cfg: dict, gate_slots: int | None = None, retriever=None):
        rcfg = cfg["retriever"]
        if retriever is None:
            retriever = SharedRetriever(
                FaissRetriever(
                    index_path=rcfg["index_path"],
                    docs_path=rcfg["docs_path"],
                    model_name=cfg["embedding"]["model"],
                    query_prefix=rcfg.get("query_prefix", ""),
                    device=rcfg.get("device"),
                )
            )
        self.retriever = retriever
        wcfg = cfg["crag"]["web_search"]
        if wcfg["provider"] != "mock":
            raise ValueError(f"지원하지 않는 web_search provider: {wcfg['provider']}")
        self.web = MockWebSearch(retriever, load_web_latencies(wcfg["latency_source"]))
        self.gate = LLMGate(gate_slots)
        self.cfg = cfg

    def pipeline(self, name: str, state: QueryState, priority_fn, t0: float):
        llm = QueryLLM(self.cfg["llm"], state, self.gate, priority_fn, t0)
        retriever = QueryRetriever(self.retriever, state)
        top_k = self.cfg["retriever"]["top_k"]
        if name == "vanilla":
            return VanillaRAG(retriever, llm, top_k=top_k)
        c = self.cfg["crag"]
        return CRAG(
            retriever,
            llm,
            self.web,
            top_k=top_k,
            web_top_k=c["web_search"]["top_k"],
            upper_threshold=c["upper_threshold"],
            lower_threshold=c["lower_threshold"],
            parallel_grading=name == "crag_parallel",
            name=name,
        )


def fifo_priority(state: QueryState, kind: str, prompt: str, idx: int) -> float:
    return time.perf_counter()
