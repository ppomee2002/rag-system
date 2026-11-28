"""공식 T5 evaluator 를 요청 간에 공유하는 단일 GPU worker queue (C2 Bottleneck-aware Gating 용).

왜 필요한가: MODE A 의 evaluator 는 호출자가 직접 `score_pairs` 를 부르는 동기 함수다. 여러 요청이
동시에 T5 를 쓰면 (grading / internal refinement / external refinement) 누가 언제 GPU 를 쓰는지,
얼마나 기다렸는지, 취소된 작업이 GPU 를 얼마나 썼는지를 알 수 없다. 이 모듈은 T5 forward 를
worker thread 하나로 모아 (GPU 1장 = 실행 1개) 작업 (job) 단위로 기록한다.

점수 규칙은 바꾸지 않는다.
  - 입력 텐서: 쌍마다 따로 토큰화 (padding="max_length", max_length=512, truncation 은 호출자가
    지정)
    -> 공식 batch_size=1 과 같은 텐서. batch_size>1 이면 길이가 같은 (=512 로 padding 된) 쌍만
    묶는다 (truncation 없는 512 초과 입력은 항상 혼자).
  - 빈 문서 -1.0 / 4 단어 미만 strip -1.0 / refinement 오류 -1.0 은 `OfficialRules` 가 공식
    `OfficialRetrievalEvaluator.score_documents / score_strips` 와 같게 처리한다 (테스트로 대조).

Memo (선택, 기본 off): 같은 요청 안에서 **입력 텐서 (input_ids, padding 포함) 가 같은** 쌍은 forward
결과를 재사용한다. 공식 selection 모드의 internal refinement 는 grading 과 같은 10문서를 다시
채점하므로 (truncation=True, 앞뒤 공백 차이) 512 토큰 이하 문서는 텐서가 같을 수 있다. 텍스트가
아니라 텐서로 key 를 잡으므로 truncation 으로 잘린 문서는 자동으로 miss.
`memo_verify=True` 면 hit 도 다시 계산해 값이 비트 단위로 같은지 기록한다 (`memo_mismatch`).
검증 전에는 켜지 않는다.

취소: queue 에 있는 job 은 즉시 제거 (GPU 시간 0), 실행 중인 job 은 다음 batch 경계에서 멈춘다
(그때까지의 GPU 시간은 job.busy_s 에 남는다 = 낭비).

요청 간 batching (다른 팀원 실험) 은 `BatchPolicy` 를 바꿔 끼운다. 기본 `PerJobFIFO` 는 job 하나를
끝까지 처리한 뒤 다음 job (FIFO) 으로 간다.
"""

from __future__ import annotations

import hashlib
import itertools
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag.crag_official import source
from rag.crag_official.evaluator import PairScore, normalize_passage, pair_text


# ---------------------------------------------------------------- runner (tokenize / forward)
@dataclass
class Encoded:
    text: str
    ids: tuple[int, ...]  # 모델에 들어가는 input_ids (padding 포함)
    n_tokens: int  # attention mask 합 (padding 제외)

    @property
    def length(self) -> int:
        return len(self.ids)

    @property
    def key(self) -> str:
        return hashlib.sha1(repr(self.ids).encode()).hexdigest()


class PairRunner(Protocol):
    name: str

    def encode(self, text: str, truncation: bool) -> Encoded: ...

    def forward(self, batch: list[Encoded]) -> list[float]: ...

    def describe(self) -> dict[str, Any]: ...


class T5PairRunner:
    """`T5EvaluatorBackend` 의 tokenizer / model 을 그대로 쓴다 (가중치를 다시 올리지 않음)."""

    name = "t5_official"

    def __init__(self, backend):
        self.backend = backend
        self.torch = backend.torch

    def encode(self, text: str, truncation: bool) -> Encoded:
        enc = self.backend.tokenizer(
            [text], padding="max_length", truncation=truncation, max_length=self.backend.max_length
        )
        ids = tuple(int(x) for x in enc["input_ids"][0])
        return Encoded(text, ids, int(sum(enc["attention_mask"][0])))

    def forward(self, batch: list[Encoded]) -> list[float]:
        torch = self.torch
        dev = self.backend.device
        ids = torch.tensor([list(e.ids) for e in batch], dtype=torch.long)
        # T5 tokenizer 는 오른쪽 padding -> 앞 n_tokens 개가 1 (공식 attention_mask 와 같음)
        mask = torch.tensor(
            [[1] * e.n_tokens + [0] * (e.length - e.n_tokens) for e in batch], dtype=torch.long
        )
        with torch.no_grad():
            out = self.backend.model(ids.to(dev), attention_mask=mask.to(dev))
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        return out["logits"].float().cpu().reshape(-1).tolist()

    def describe(self) -> dict[str, Any]:
        return {"runner": self.name, **self.backend.describe()}


class MockPairRunner:
    """테스트·로컬 smoke 용. 토큰 = 공백 단위 단어의 hash (공백 정규화 -> sentencepiece 와 비슷하게
    앞뒤·연속 공백은 같은 텐서). forward 는 batch 당 sleep (GPU 시간 흉내)."""

    name = "mock"

    def __init__(
        self,
        fn: Callable[[str], float],
        seconds_per_pair: float = 0.0,
        seconds_per_batch: float = 0.0,
        max_length: int = source.EVAL_MAX_LENGTH,
        sleep=time.sleep,
    ):
        self.fn = fn
        self.seconds_per_pair = seconds_per_pair
        self.seconds_per_batch = seconds_per_batch
        self.max_length = max_length
        self.sleep = sleep
        self.n_forward_pairs = 0

    def encode(self, text: str, truncation: bool) -> Encoded:
        toks = [int(hashlib.md5(w.encode()).hexdigest()[:6], 16) for w in text.split()] + [1]
        if truncation and len(toks) > self.max_length:
            toks = toks[: self.max_length - 1] + [1]
        n = len(toks)
        if n < self.max_length:
            toks = toks + [0] * (self.max_length - n)
        return Encoded(text, tuple(toks), n)

    def forward(self, batch: list[Encoded]) -> list[float]:
        self.sleep(self.seconds_per_batch + self.seconds_per_pair * len(batch))
        self.n_forward_pairs += len(batch)
        return [float(self.fn(e.text)) for e in batch]

    def describe(self) -> dict[str, Any]:
        return {
            "runner": self.name,
            "seconds_per_pair": self.seconds_per_pair,
            "seconds_per_batch": self.seconds_per_batch,
            "note": "mock — 결과를 정확도·latency 로 사용하지 않음",
        }


# ---------------------------------------------------------------- job
class JobCancelled(RuntimeError):
    pass


_ids = itertools.count(1)


@dataclass
class T5Job:
    request_id: str
    kind: str  # grading | internal_refine | external_refine | (다른 실험이 정한 이름)
    texts: list[str]
    truncation: bool
    meta: dict = field(default_factory=dict)
    id: int = field(default_factory=lambda: next(_ids))
    status: str = "queued"  # queued | running | done | cancelled | error
    t_submit: float | None = None
    t_start: float | None = None
    t_end: float | None = None
    t_cancel: float | None = None
    cancel_requested: bool = False
    cancel_effect: str | None = None  # removed_from_queue | stopped_running | after_done
    scores: list[float | None] = field(default_factory=list)
    n_tokens: list[int | None] = field(default_factory=list)
    busy_s: float = 0.0  # worker 가 이 job 에 쓴 시간 (토큰화 + forward)
    forward_s: float = 0.0
    n_forward: int = 0  # 실제 forward 한 쌍 수
    n_memo_hits: int = 0
    memo_mismatch: int = 0
    max_memo_abs_diff: float = 0.0
    n_batches: int = 0
    usage: str | None = None  # 호출자가 기록: used | discarded
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event, repr=False)

    def wait(self, timeout: float | None = None) -> list[float]:
        if not self.done.wait(timeout):
            raise TimeoutError(f"T5 job {self.id} ({self.kind}) timeout")
        if self.status == "cancelled":
            raise JobCancelled(f"T5 job {self.id} cancelled")
        if self.status == "error":
            raise RuntimeError(self.error)
        return list(self.scores)  # type: ignore[arg-type]

    @property
    def queue_wait_s(self) -> float | None:
        if self.t_submit is None:
            return None
        end = self.t_start if self.t_start is not None else self.t_end
        return None if end is None else end - self.t_submit

    def to_dict(self, t0: float = 0.0) -> dict:
        def rel(t):
            return None if t is None else t - t0

        return {
            "job_id": self.id,
            "request_id": self.request_id,
            "kind": self.kind,
            "n_pairs": len(self.texts),
            "truncation": self.truncation,
            "status": self.status,
            "usage": self.usage,
            "cancel_effect": self.cancel_effect,
            "t_submit": rel(self.t_submit),
            "t_start": rel(self.t_start),
            "t_end": rel(self.t_end),
            "t_cancel": rel(self.t_cancel),
            "queue_wait_s": self.queue_wait_s,
            "busy_s": self.busy_s,
            "forward_s": self.forward_s,
            "n_forward": self.n_forward,
            "n_memo_hits": self.n_memo_hits,
            "memo_mismatch": self.memo_mismatch,
            "max_memo_abs_diff": self.max_memo_abs_diff,
            "n_batches": self.n_batches,
            "n_tokens": self.n_tokens,
            "error": self.error,
            **{f"meta_{k}": v for k, v in self.meta.items()},
        }


# ---------------------------------------------------------------- batch policy
class BatchPolicy(Protocol):
    """queue 에서 다음에 처리할 job 을 고른다. 요청 간 batching 실험은 이 자리에 구현을 넣는다
    (같은 batch 에 여러 job 의 쌍을 넣으려면 T5Service._run_job 대신 별도 실행 경로가 필요)."""

    name: str

    def pick(self, queue: deque[T5Job]) -> T5Job: ...


class PerJobFIFO:
    name = "per_job_fifo"

    def pick(self, queue: deque[T5Job]) -> T5Job:
        return queue.popleft()


# ---------------------------------------------------------------- service
class T5Service:
    def __init__(
        self,
        runner: PairRunner,
        batch_size: int = 1,
        memo: bool = False,
        memo_verify: bool = False,
        policy: BatchPolicy | None = None,
        clock: Callable[[], float] = time.perf_counter,
        keep_jobs: bool = True,
    ):
        self.runner = runner
        self.batch_size = max(1, int(batch_size))
        self.memo_enabled = bool(memo)
        self.memo_verify = bool(memo_verify)
        self.policy = policy or PerJobFIFO()
        self.clock = clock
        self.keep_jobs = keep_jobs
        self._cv = threading.Condition()
        self._queue: deque[T5Job] = deque()
        self._running: T5Job | None = None
        self._memo: dict[str, dict[str, float]] = {}  # request_id -> tensor key -> score
        self._closed = False
        self.jobs: list[T5Job] = []
        self.busy_total_s = 0.0
        self.max_queue_len = 0
        self._thread = threading.Thread(target=self._loop, name="t5-worker", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ client API
    def submit(
        self, request_id: str, kind: str, texts: list[str], truncation: bool, meta=None
    ) -> T5Job:
        job = T5Job(str(request_id), kind, list(texts), bool(truncation), dict(meta or {}))
        job.scores = [None] * len(texts)
        job.n_tokens = [None] * len(texts)
        with self._cv:
            if self._closed:
                raise RuntimeError("T5Service closed")
            job.t_submit = self.clock()
            if not texts:  # 빈 job: 바로 완료 (GPU 사용 없음)
                job.status, job.t_start, job.t_end = "done", job.t_submit, job.t_submit
                job.done.set()
            else:
                self._queue.append(job)
                if self.memo_enabled:  # 요청이 release 한 뒤에는 worker 가 다시 만들지 않는다
                    self._memo.setdefault(job.request_id, {})
                self.max_queue_len = max(self.max_queue_len, len(self._queue))
                self._cv.notify_all()
            if self.keep_jobs:
                self.jobs.append(job)
        return job

    def cancel(self, job: T5Job) -> str:
        with self._cv:
            job.cancel_requested = True
            if job.t_cancel is None:
                job.t_cancel = self.clock()
            if job.status == "queued":
                self._queue.remove(job)
                job.status, job.t_end = "cancelled", job.t_cancel
                job.cancel_effect = "removed_from_queue"
                job.done.set()
            elif job.status == "running":
                job.cancel_effect = "stop_requested"  # worker 가 stopped_running 으로 바꾼다
            elif job.cancel_effect is None:
                job.cancel_effect = "after_done"
            return job.cancel_effect

    def release_request(self, request_id: str) -> None:
        with self._cv:
            self._memo.pop(str(request_id), None)

    def snapshot(self) -> dict:
        with self._cv:
            return {
                "queue_len": len(self._queue),
                "running": self._running.id if self._running else None,
                "memo_requests": len(self._memo),
            }

    def idle(self, timeout: float = 30.0) -> bool:
        """queue 가 비고 실행 중인 job 이 없을 때까지 기다린다 (누수 점검용)."""
        end = time.monotonic() + timeout
        with self._cv:
            while self._queue or self._running is not None:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def close(self, timeout: float = 30.0) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join(timeout)

    def describe(self) -> dict[str, Any]:
        return {
            "service": "t5_single_worker_queue",
            "batch_size": self.batch_size,
            "memo": self.memo_enabled,
            "memo_verify": self.memo_verify,
            "memo_scope": "request (같은 요청 안에서 입력 텐서가 같은 쌍)",
            "policy": self.policy.name,
            "runner": self.runner.describe(),
        }

    # ------------------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._closed:
                    self._cv.wait()
                if not self._queue and self._closed:
                    return
                job = self.policy.pick(self._queue)
                job.status, job.t_start = "running", self.clock()
                self._running = job
                memo = self._memo.get(job.request_id) if self.memo_enabled else None
            try:
                self._run_job(job, memo)
            except Exception as e:  # noqa: BLE001 - job 오류는 호출자에게 전달
                job.status, job.error = "error", f"{type(e).__name__}: {e}"[:500]
            with self._cv:
                job.t_end = self.clock()
                if job.status == "running":
                    job.status = "done"
                self.busy_total_s += job.busy_s
                self._running = None
                job.done.set()
                self._cv.notify_all()

    def _run_job(self, job: T5Job, memo: dict | None) -> None:
        t0 = self.clock()
        encs = [self.runner.encode(t, job.truncation) for t in job.texts]
        job.n_tokens = [e.n_tokens for e in encs]
        todo: list[int] = []
        verify: dict[int, float] = {}
        for i, e in enumerate(encs):
            if memo is not None and e.key in memo:
                job.scores[i] = memo[e.key]
                job.n_memo_hits += 1
                if self.memo_verify:
                    verify[i] = memo[e.key]
                    todo.append(i)
            else:
                todo.append(i)
        job.busy_s += self.clock() - t0
        for batch in _batches(todo, encs, self.batch_size):
            if job.cancel_requested:
                job.status, job.cancel_effect = "cancelled", "stopped_running"
                return
            t1 = self.clock()
            out = self.runner.forward([encs[i] for i in batch])
            dt = self.clock() - t1
            job.forward_s += dt
            job.busy_s += dt
            job.n_batches += 1
            for i, s in zip(batch, out, strict=True):
                if i in verify:
                    diff = abs(s - verify[i])
                    job.max_memo_abs_diff = max(job.max_memo_abs_diff, diff)
                    job.memo_mismatch += s != verify[i]
                else:
                    job.scores[i] = s
                    job.n_forward += 1
                    if memo is not None:
                        memo[encs[i].key] = s
            job.n_forward += sum(1 for i in batch if i in verify)


def _batches(todo: list[int], encs: list[Encoded], size: int) -> list[list[int]]:
    """순서를 유지하며 같은 길이의 연속 쌍만 최대 size 개씩 묶는다."""
    out: list[list[int]] = []
    for i in todo:
        if out and len(out[-1]) < size and encs[out[-1][0]].length == encs[i].length:
            out[-1].append(i)
        else:
            out.append([i])
    return out


# ---------------------------------------------------------------- official rules on top of jobs
@dataclass
class ScoreHandle:
    """공식 규칙 (빈 문서 / 짧은 strip) 으로 미리 정한 점수 + T5 job 결과를 합친다."""

    job: T5Job | None
    fixed: list[PairScore | None]
    todo: list[int]
    on_error: str  # raise (grading) | minus_one (refinement, 공식 utils.py:288)

    def result(self, timeout: float | None = None) -> list[PairScore]:
        out = list(self.fixed)
        if self.job is None:
            return out  # type: ignore[return-value]
        try:
            scores = self.job.wait(timeout)
        except JobCancelled:
            raise
        except Exception as e:
            if self.on_error == "raise":
                raise
            for i in self.todo:
                out[i] = PairScore(source.SHORT_STRIP_SCORE, None, None, "error", str(e)[:200])
            return out  # type: ignore[return-value]
        per = self.job.busy_s / max(1, len(self.todo))
        for k, i in enumerate(self.todo):
            out[i] = PairScore(scores[k], self.job.n_tokens[k], per, "model")
        return out  # type: ignore[return-value]


class OfficialRules:
    """`OfficialRetrievalEvaluator` 의 점수 규칙을 T5Service job 으로 (같은 결과, 비동기)."""

    def __init__(self, service: T5Service, normalize: bool = True):
        self.service = service
        self.normalize = normalize

    def submit_documents(
        self, request_id: str, query: str, passages: list[str], kind: str = "grading"
    ) -> ScoreHandle:
        prepped = [normalize_passage(p) if self.normalize else p for p in passages]
        todo = [i for i, p in enumerate(prepped) if p.strip()]
        fixed: list[PairScore | None] = [
            PairScore(source.EMPTY_PASSAGE_SCORE, 0, 0.0, "empty_rule") for _ in prepped
        ]
        for i in todo:
            fixed[i] = None
        texts = [pair_text(query, prepped[i]) for i in todo]
        job = self.service.submit(request_id, kind, texts, truncation=False) if todo else None
        return ScoreHandle(job, fixed, todo, "raise")

    def submit_strips(
        self, request_id: str, query: str, strips: list[str], kind: str
    ) -> ScoreHandle:
        fixed: list[PairScore | None] = []
        todo = []
        for i, s in enumerate(strips):
            if len(s.split()) < source.SHORT_STRIP_MIN_WORDS:
                fixed.append(PairScore(source.SHORT_STRIP_SCORE, None, 0.0, "short_strip_rule"))
            else:
                fixed.append(None)
                todo.append(i)
        texts = [pair_text(query, strips[i]) for i in todo]
        job = self.service.submit(request_id, kind, texts, truncation=True) if todo else None
        return ScoreHandle(job, fixed, todo, "minus_one")


class JobScorer:
    """refinement 함수 (`select_relevants`) 가 부르는 `score_strips(query, strips)` 를 job 으로.

    guard() 가 False 를 돌려주면 submit 하지 않고 JobCancelled (취소와 submit 의 경합 방지:
    guard 와 submit 은 같은 lock 안에서 실행된다).
    """

    def __init__(self, rules: OfficialRules, request_id: str, kind: str, lock=None, guard=None):
        self.rules = rules
        self.request_id = request_id
        self.kind = kind
        self.lock = lock or threading.Lock()
        self.guard = guard
        self.jobs: list[T5Job] = []

    def score_strips(self, query: str, strips: list[str]) -> list[PairScore]:
        with self.lock:
            if self.guard is not None and not self.guard():
                raise JobCancelled(f"{self.kind}: cancelled before submit")
            h = self.rules.submit_strips(self.request_id, query, strips, self.kind)
            if h.job is not None:
                self.jobs.append(h.job)
        return h.result()

    def cancel(self) -> None:
        with self.lock:
            for j in self.jobs:
                self.rules.service.cancel(j)
