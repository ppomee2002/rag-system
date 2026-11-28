"""MODE A 위의 B / C / Cbn (C2 Bottleneck-aware Gating).

공식 CRAG (online 형태) 한 요청의 작업:
  grading            T5, 검색 10문서 (truncation 없음)                       -> action
  internal_refine    T5, 같은 10문서 strip (selection, truncation=True)       correct / ambiguous
  rewrite            generator LLM, 공식 PopQA keyword prompt + "Wikipedia, "  ambiguous / incorrect
  web_search         기록 재생 (query_id, 기록 지연)                           ambiguous / incorrect
  external_refine    T5, web 페이지 문단                                       ambiguous / incorrect
  generation         generator LLM, 공식 PopQA prompt

조건 (batching·memo·serving 은 셋이 같다. 다른 것은 correction 경로의 시작 시점뿐):
  B    grading 완료 -> 판정 -> 필요한 경로만 실행 (internal_refine 과 rewrite->web->external_refine
       은 판정 후 병렬). speculation 없음.
  C    grading 과 rewrite->web->external_refine 를 동시에 시작 (external_refine 도 web 직후 T5 queue
       에 submit). Correct 판정 시 취소 (queue 대기 중이면 제거, 실행 중이면 다음 batch 에서 정지,
       rewrite·web 은 결과 폐기).
  Cbn  grading 과 rewrite->web 만 동시에. external_refine 은 판정이 Ambiguous/Incorrect 로 확정된
       뒤에만 submit (web 이 먼저 끝나도 기다린다). Correct 면 rewrite/web 취소·폐기.
internal_refine 은 세 조건 모두 판정 후 (memo 를 쓰려면 grading 결과가 필요하고, 조건 간 차이를
correction 경로 하나로 한정하기 위해).

experiment_mode (두 실험은 결과 폴더·주장 범위가 다르다)
  controlled_replay  generation 입력 = MODE A 와 같은 저자 지식 파일의 줄 (판정별, 고정).
                     correction workload (internal/external refinement, rewrite, web) 는 실제로
                     실행하지만 결과는 generation 에 쓰지 않는다. -> B/C/Cbn 의 serving 비용 비교용.
                     정확도 동일성은 evidence 고정으로 생긴 것이므로 End-to-End CRAG 의 정확도
                     보존 증거가 아니다.
  runtime_e2e        실행 중 계산한 지식 (internal / external / Knowledge1-2) 을 generation 에 쓴다.
                     B/C/Cbn 의 action·evidence·prompt·정확도·성능 비교. MODE A (저자 지식) 와는
                     evidence 가 다르므로 정확도는 B (같은 runtime 경로) 를 기준으로 본다.
두 모드 모두 그 판정에 필요한 correction 작업 (internal_refine / rewrite->web->external_refine) 이
끝난 뒤에만 generation 을 시작한다 (precondition 기록: generation_precondition).

stage 이름: rag.evidence.pipeline.STAGES_OFFICIAL (retrieval/grading/rewrite/web_search/generation/
refinement). refinement 는 internal / external 을 refinement_detail 로 나눠 기록한다. 병렬 stage 는
겹치므로 stage 합 != E2E.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any

from rag.crag_official import generation as offgen
from rag.crag_official import source
from rag.crag_official.evaluator import official_action, official_doc_flag
from rag.crag_official.refinement import (
    ambiguous_knowledge,
    external_knowledge,
    internal_knowledge,
    web_strips,
)
from rag.crag_official.t5_service import JobCancelled, JobScorer, OfficialRules, T5Service
from rag.eval.metrics import accuracy
from rag.evidence.pipeline import STAGES_OFFICIAL
from rag.llm.engine_llm import EngineLLM
from rag.telemetry.calls import CallRecorder
from rag.utils.timing import StageTimer, stopwatch

MODES = ("B", "C", "Cbn")
EXPERIMENT_MODES = {"controlled_replay": "official_ref", "runtime_e2e": "runtime"}
EVIDENCE_SOURCES = tuple(EXPERIMENT_MODES.values())


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class GatingConfig:
    mode: str
    experiment_mode: str = "controlled_replay"
    internal_refinement: str = "run"  # run | skip
    prompt_version: str = source.DEFAULT_PROMPT_VERSION
    decompose_mode: str = source.DEFAULT_DECOMPOSE_MODE
    rewrite_max_tokens: int = 32
    wiki_prefix: bool = True
    generation_max_tokens: int = 100
    upper: float = source.OFFICIAL_THRESHOLDS["popqa"]["upper"]
    lower: float = source.OFFICIAL_THRESHOLDS["popqa"]["lower"]
    record_texts: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode: {MODES} (got {self.mode})")
        if self.experiment_mode not in EXPERIMENT_MODES:
            raise ValueError(f"experiment_mode: {tuple(EXPERIMENT_MODES)}")
        if self.internal_refinement not in ("run", "skip"):
            raise ValueError("internal_refinement: run | skip")
        if self.experiment_mode == "runtime_e2e" and self.internal_refinement == "skip":
            raise ValueError("runtime_e2e 에는 internal_refinement=run 필요")

    @property
    def evidence_source(self) -> str:
        return EXPERIMENT_MODES[self.experiment_mode]


def correction_query(question: str, llm: EngineLLM, max_tokens: int, wiki_prefix: bool) -> str:
    """공식 PopQA keyword prompt (utils.py:231, 질문 삽입 결함은 의도대로 수정) + 'Wikipedia, '.

    공식은 gpt-3.5-turbo-16k, 여기서는 MODE A generator (deviation). 첫 줄만 쓴다."""
    prompt = source.POPQA_KEYWORD_PROMPT.replace("{question}", question)
    q = llm.generate(prompt, kind="rewrite", max_tokens=max_tokens)
    q = q.strip().splitlines()[0].strip() if q.strip() else ""
    return (source.WIKI_QUERY_PREFIX + q) if wiki_prefix else q


class Decision:
    def __init__(self):
        self.event = threading.Event()
        self.action: str | None = None
        self.t: float | None = None

    def set(self, action: str, t: float) -> None:
        self.action, self.t = action, t
        self.event.set()


class Chain:
    """rewrite -> web_search -> (gate) -> external_refine. thread 하나."""

    def __init__(
        self,
        pipe: GatedOfficialCRAG,
        item,
        rid: str,
        t0: float,
        decision: Decision,
        speculative: bool,
        submit: str,
    ):
        self.pipe, self.item, self.rid = pipe, item, rid
        # 요청이 먼저 끝나도 (correct 에서 버린 rewrite) 호출 기록이 남도록 chain 전용 recorder
        self.calls = CallRecorder(t0=t0)
        self.llm = EngineLLM(pipe.engine, recorder=self.calls)
        self.decision = decision
        self.speculative = speculative
        self.submit = submit  # immediate | after_decision
        self.cancel_event = threading.Event()
        self.lock = threading.Lock()
        self.scorer = JobScorer(pipe.rules, rid, "external_refine", self.lock, self._may_submit)
        self.steps: dict[str, dict[str, Any]] = {}
        self.status = "created"
        self.cancel_reason: str | None = None
        self.t_cancel: float | None = None
        self.query: str | None = None
        self.web = None
        self.result = None  # RefinementResult | None
        self.external_text: str | None = None
        self.error: str | None = None
        self.timer = StageTimer()
        self.thread = threading.Thread(target=self._run, name=f"chain-{rid}", daemon=True)

    # ------------------------------------------------------------ control
    def start(self) -> Chain:
        self.t_created = self.pipe.clock()
        with self.pipe.chains_lock:
            self.pipe.open_chains.add(self)
            self.pipe.all_chains.append(self)
        self.thread.start()
        return self

    def _may_submit(self) -> bool:  # JobScorer.lock 안에서 호출
        if self.cancel_event.is_set():
            return False
        if self.submit == "after_decision":
            return self.decision.event.is_set() and self.decision.action != "correct"
        return True

    def cancel(self, reason: str) -> None:
        with self.lock:
            if self.cancel_event.is_set():
                return
            self.cancel_reason, self.t_cancel = reason, self.pipe.clock()
            self.cancel_event.set()
        self.scorer.cancel()

    def join(self, timeout: float | None = None) -> bool:
        self.thread.join(timeout)
        return not self.thread.is_alive()

    # ------------------------------------------------------------ steps
    def _step(self, name: str):
        rec = self.steps.setdefault(name, {"t_start": self.pipe.clock()})
        return rec

    def _end(self, rec: dict, status: str) -> None:
        rec["t_end"] = self.pipe.clock()
        rec["wall"] = rec["t_end"] - rec["t_start"]
        rec["status"] = status

    def _run(self) -> None:
        clock = self.pipe.clock
        self.status = "running"
        try:
            if self.cancel_event.is_set():
                self.status = "cancelled_before_rewrite"
                return
            rec = self._step("rewrite")
            with self.timer.stage("rewrite"):
                self.query = self.pipe.correction_query(self.item.question, self.llm)
            self._end(rec, "discarded" if self.cancel_event.is_set() else "ok")
            if self.cancel_event.is_set():
                self.status = "cancelled_after_rewrite"
                return
            rec = self._step("web_search")
            with self.timer.stage("web_search"):
                self.web = self.pipe.web.lookup(self.item.query_id, self.query, self.cancel_event)
            self._end(rec, "cancelled" if self.web.status == "cancelled" else self.web.status)
            if self.cancel_event.is_set():
                self.status = "cancelled_after_web"
                return
            if self.web.status == "miss":
                raise RuntimeError(f"web replay miss: query_id={self.item.query_id}")
            if self.submit == "after_decision":
                rec = self._step("gate_wait")
                while not self.decision.event.wait(0.05):
                    if self.cancel_event.is_set():
                        break
                self._end(rec, "ok")
                if self.cancel_event.is_set() or self.decision.action == "correct":
                    self.status = "cancelled_at_gate"
                    return
            rec = self._step("external_refine")
            rec["t_submit_allowed"] = clock()
            with self.timer.stage("refinement"):
                self._external()
            self._end(rec, "ok")
            self.status = "done"
        except JobCancelled:
            self.status = "cancelled_in_external_refine"
            for r in self.steps.values():
                if "t_end" not in r:
                    self._end(r, "cancelled")
        except Exception as e:  # noqa: BLE001 - 요청 기록에 남긴다
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"[:500]
        finally:
            with self.pipe.chains_lock:
                self.pipe.open_chains.discard(self)

    def _external(self) -> None:
        pages = self.web.docs
        strips, _ = web_strips(pages)
        if strips:
            self.result = external_knowledge(self.item.question, pages, self.scorer)
            self.external_text = self.result.text
        else:  # 공식: 문단이 없으면 snippet 결합 (T5 없음)
            self.external_text = source.STRIP_JOIN.join(
                d.text for d in self.web.snippet_docs
            ).replace("\n", " ")

    def to_dict(self, t0: float) -> dict:
        def rel(t):
            return None if t is None else t - t0

        steps = {
            k: {kk: (rel(vv) if kk.startswith("t_") else vv) for kk, vv in v.items()}
            for k, v in self.steps.items()
        }
        return {
            "request_id": self.rid,
            "speculative": self.speculative,
            "submit_policy": self.submit,
            "status": self.status,
            "cancel_reason": self.cancel_reason,
            "t_created": rel(self.t_created),
            "t_cancel": rel(self.t_cancel),
            "steps": steps,
            "query": self.query,
            "web": None
            if self.web is None
            else {
                "status": self.web.status,
                "latency": self.web.latency,
                "rewrite_matched": self.web.rewrite_matched,
                "n_results": self.web.n_results,
                "n_pages": self.web.n_kept,
            },
            "external_refine_jobs": [j.id for j in self.scorer.jobs],
            "external_sha16": sha16(self.external_text) if self.external_text is not None else None,
            "n_strips": len(self.result.strips) if self.result else 0,
            "calls": self.calls.to_dicts(),
            "error": self.error,
        }


class GatedOfficialCRAG:
    def __init__(self, cfg: GatingConfig, t5: T5Service, engine, web, clock=time.perf_counter):
        self.cfg = cfg
        self.t5 = t5
        self.rules = OfficialRules(t5)
        self.engine = engine
        self.web = web
        self.clock = clock
        self.chains_lock = threading.Lock()
        self.open_chains: set[Chain] = set()
        self.all_chains: list[Chain] = []

    def describe(self) -> dict:
        return {
            **self.cfg.__dict__,
            "stages": STAGES_OFFICIAL,
            "t5": self.t5.describe(),
            "web": self.web.describe(),
        }

    def correction_query(self, question: str, llm: EngineLLM) -> str:
        return correction_query(question, llm, self.cfg.rewrite_max_tokens, self.cfg.wiki_prefix)

    # ------------------------------------------------------------ one request
    def run(self, item, rid: str, t0: float) -> dict:
        cfg, clock = self.cfg, self.clock
        timer = StageTimer()
        calls = CallRecorder(t0=t0)
        llm = EngineLLM(self.engine, recorder=calls)
        decision = Decision()
        chain: Chain | None = None
        internal = None
        out: dict[str, Any] = {"request_id": rid, "index": item.index, "mode": cfg.mode}
        t_begin = clock()
        try:
            with stopwatch() as sw:
                with timer.stage("retrieval"):  # 공식 Contriever top-10 고정
                    passages = item.passages
                g_start = clock()
                handle = self.rules.submit_documents(rid, item.question, passages)
                if cfg.mode in ("C", "Cbn"):
                    chain = Chain(
                        self,
                        item,
                        rid,
                        t0,
                        decision,
                        True,
                        "immediate" if cfg.mode == "C" else "after_decision",
                    ).start()
                with timer.stage("grading"):
                    pairs = handle.result()
                scores = [p.score for p in pairs]
                flags = [official_doc_flag(s, cfg.upper, cfg.lower) for s in scores]
                action = official_action(flags)
                decision.set(action, clock())
                needs_web = action != "correct"
                if chain is not None and not needs_web:
                    chain.cancel("decided_correct")
                if chain is None and needs_web:
                    chain = Chain(self, item, rid, t0, decision, False, "immediate").start()
                if action != "incorrect" and cfg.internal_refinement == "run":
                    scorer = JobScorer(self.rules, rid, "internal_refine")
                    with timer.stage("refinement"):
                        internal = internal_knowledge(
                            item.question, item.refine_passages, scorer, cfg.decompose_mode
                        )
                    t_int = timer.totals["refinement"]
                    t_int_end = clock()
                else:
                    t_int, t_int_end = None, None
                if needs_web:
                    chain.join()
                    if chain.status != "done":
                        raise RuntimeError(f"correction chain {chain.status}: {chain.error}")
                paragraph, evidence_id = self._evidence(item, action, internal, chain)
                prompt = offgen.format_popqa_prompt(item.question, paragraph, cfg.prompt_version)
                pre = self._precondition(
                    action,
                    internal,
                    scorer if internal is not None else None,
                    t_int_end,
                    chain if needs_web else None,
                    t0,
                )
                with timer.stage("generation"):
                    raw = llm.generate(
                        prompt, kind="generate", max_tokens=cfg.generation_max_tokens
                    )
                pred = offgen.postprocess_answer(raw)
            stage = {s: timer.totals.get(s) for s in STAGES_OFFICIAL}
            if needs_web:
                for s in ("rewrite", "web_search"):
                    stage[s] = chain.timer.totals.get(s)
                ext = chain.timer.totals.get("refinement")
                stage["refinement"] = (t_int or 0.0) + (ext or 0.0) if (t_int or ext) else None
            else:
                ext = None
            ref_correct = item.ref["correct"].lstrip("#").strip()
            out.update(
                action=action,
                scores=scores,
                flags=flags,
                decided_at=decision.t - t0,
                grading_wall=decision.t - g_start,
                grading_job=handle.job.id if handle.job else None,
                internal_refine_jobs=[j.id for j in scorer.jobs] if internal is not None else [],
                internal_sha16=sha16(internal.text) if internal is not None else None,
                internal_matches_ref_correct=(
                    internal.text.strip() == ref_correct if internal is not None else None
                ),
                chain=chain.rid if chain is not None else None,
                chain_speculative=chain.speculative if chain is not None else None,
                experiment_mode=cfg.experiment_mode,
                evidence_source=cfg.evidence_source,
                generation_precondition=pre,
                evidence_id=evidence_id,
                evidence_sha16=sha16(paragraph),
                prompt_sha16=sha16(prompt),
                prompt_chars=len(prompt),
                prediction=pred,
                raw_prediction=raw if raw != pred else None,
                official_match=offgen.official_match(pred, item.answers) if item.answers else None,
                acc=accuracy(pred, item.answers) if item.answers else None,
                stage_timings=stage,
                refinement_detail={"internal": t_int, "external": ext},
                e2e_latency=sw["elapsed"],
                t_start=t_begin - t0,
                t_end=clock() - t0,
                calls=calls.to_dicts(),
                error=None,
            )
            if cfg.record_texts:
                out.update(paragraph=paragraph, prompt=prompt)
        except Exception as e:  # noqa: BLE001 - 행으로 남기고 계속
            if chain is not None:
                chain.cancel("request_error")
            out.update(error=f"{type(e).__name__}: {e}"[:500], t_end=clock() - t0)
        finally:
            self.t5.release_request(rid)
        return out

    def _precondition(self, action, internal, int_scorer, t_int_end, chain, t0) -> dict:
        """generation 시작 직전: 이 판정에 필요한 correction 작업이 모두 끝났는지."""
        t_gen = self.clock()
        need_int = action != "incorrect" and self.cfg.internal_refinement == "run"
        need_ext = action != "correct"
        ends, problems = [], []
        if need_int:
            if internal is None or t_int_end is None:
                problems.append("internal_refine_missing")
            else:
                ends.append(t_int_end)
                if any(not j.done.is_set() or j.status != "done" for j in int_scorer.jobs):
                    problems.append("internal_refine_job_not_done")
        if need_ext:
            ext = (chain.steps.get("external_refine") or {}) if chain is not None else {}
            if chain is None or chain.status != "done" or "t_end" not in ext:
                problems.append("external_refine_not_done")
            else:
                ends.append(ext["t_end"])
                if any(not j.done.is_set() or j.status != "done" for j in chain.scorer.jobs):
                    problems.append("external_refine_job_not_done")
        last = max(ends) if ends else None
        if last is not None and last > t_gen:
            problems.append("generation_started_before_correction_end")
        return {
            "ok": not problems,
            "problems": problems,
            "required": [
                k for k, v in (("internal_refine", need_int), ("external_refine", need_ext)) if v
            ],
            "t_correction_done": None if last is None else last - t0,
            "t_generation_start": t_gen - t0,
        }

    def _evidence(self, item, action: str, internal, chain: Chain | None) -> tuple[str, str]:
        if self.cfg.evidence_source == "official_ref":
            return item.ref[action], f"ref/{action}#{item.index}"
        ext = chain.external_text if chain is not None else None
        if action == "correct":
            return internal.text, f"runtime/internal#{item.index}"
        if action == "incorrect":
            return ext or "", f"runtime/external#{item.index}"
        return ambiguous_knowledge(internal.text, ext or ""), f"runtime/ambiguous#{item.index}"
